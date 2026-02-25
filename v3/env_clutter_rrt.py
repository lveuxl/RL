"""
EnvClutter环境的RRT运动规划集成版本
结合RL模型决策和RRT路径规划，实现智能抓取顺序和安全路径规划
"""

import os
import numpy as np
import sapien
import torch
import trimesh
from typing import Dict, List, Tuple, Optional
import sys

# 导入基础环境
from env_clutter import EnvClutterEnv

# 导入运动规划相关
try:
    import mplib
    MPLIB_AVAILABLE = True
except ImportError:
    print("⚠️ mplib未安装，运动规划功能将不可用")
    print("安装方法: pip install mplib-dist")
    MPLIB_AVAILABLE = False

# 导入ManiSkill相关模块
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose
from transforms3d import quaternions

# 导入训练好的模型
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    from stable_baselines3 import PPO
    MASKABLE_AVAILABLE = False

OPEN_GRIPPER = 1
CLOSE_GRIPPER = -1


class RRTMotionPlanner:
    """RRT运动规划器 - 基于mplib的抓取规划"""
    
    def __init__(
        self, 
        env: BaseEnv, 
        visualize_grasp: bool = False,
        joint_vel_limits: float = 0.9,
        joint_acc_limits: float = 0.9,
        debug: bool = False
    ):
        self.env = env
        self.base_env = env.unwrapped
        self.robot = self.base_env.agent.robot
        self.debug = debug
        self.visualize_grasp = visualize_grasp
        
        # 检查mplib可用性
        if not MPLIB_AVAILABLE:
            raise ImportError("mplib未安装，无法使用RRT规划功能")
        
        # 设置规划器参数
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.gripper_state = OPEN_GRIPPER
        
        # 初始化规划器
        self.planner = self._setup_planner()
        
        # 障碍物点云管理
        self.obstacle_points = None
        self.use_point_cloud = False
        
        # 可视化
        self.grasp_pose_visual = None
        if self.visualize_grasp:
            self._setup_grasp_visualization()
    
    def _setup_planner(self):
        """设置mplib规划器"""
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        
        planner = mplib.Planner(
            urdf=self.base_env.agent.urdf_path,
            srdf=self.base_env.agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        
        # 设置基座位置
        base_pose = sapien.Pose()
        planner.set_base_pose(np.hstack([base_pose.p, base_pose.q]))
        
        return planner
    
    def _setup_grasp_visualization(self):
        """设置抓取可视化"""
        if "grasp_pose_visual" not in self.base_env.scene.actors:
            self.grasp_pose_visual = self._build_gripper_visual()
        else:
            self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
    
    def _build_gripper_visual(self):
        """创建夹爪可视化"""
        builder = self.base_env.scene.create_actor_builder()
        width = 0.01
        grasp_width = 0.05
        
        # 中心球体
        builder.add_sphere_visual(
            pose=sapien.Pose(p=[0, 0, 0.0]),
            radius=width,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
        )
        
        # 夹爪基座
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.08]),
            half_size=[width, width, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        
        # 夹爪开口
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.05]),
            half_size=[width, grasp_width, width],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        
        # 左夹爪
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0.03 - width * 3, grasp_width + width, 0.03 - 0.05],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, width, width],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
        )
        
        # 右夹爪
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0.03 - width * 3, -grasp_width - width, 0.03 - 0.05],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, width, width],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
        )
        
        return builder.build_kinematic(name="grasp_pose_visual")
    
    def update_obstacle_points(self, points: np.ndarray):
        """更新障碍物点云"""
        if points is None or len(points) == 0:
            self.obstacle_points = None
            self.use_point_cloud = False
            return
        
        if self.obstacle_points is None:
            self.obstacle_points = points
        else:
            self.obstacle_points = np.vstack([self.obstacle_points, points])
        
        self.use_point_cloud = True
        self.planner.update_point_cloud(self.obstacle_points)
    
    def add_box_obstacle(self, size: np.ndarray, pose: sapien.Pose, sample_points: int = 256):
        """添加盒子形障碍物"""
        box = trimesh.creation.box(size, transform=pose.to_transformation_matrix())
        points, _ = trimesh.sample.sample_surface(box, sample_points)
        self.update_obstacle_points(points)
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        self.obstacle_points = None
        self.use_point_cloud = False
    
    def plan_to_pose(self, target_pose: sapien.Pose) -> Dict:
        """规划到目标位姿的路径"""
        target_pose = to_sapien_pose(target_pose)
        
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target_pose)
        
        # 获取当前关节角度
        current_qpos = self.robot.get_qpos().cpu().numpy()[0]
        
        # 使用RRT Connect算法规划
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([target_pose.p, target_pose.q]),
            current_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        
        return result
    
    def plan_screw_motion(self, target_pose: sapien.Pose) -> Dict:
        """规划螺旋运动路径（用于精确操作）"""
        target_pose = to_sapien_pose(target_pose)
        
        current_qpos = self.robot.get_qpos().cpu().numpy()[0]
        
        # 使用螺旋运动规划
        result = self.planner.plan_screw(
            np.concatenate([target_pose.p, target_pose.q]),
            current_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        
        return result
    
    def execute_path(self, result: Dict) -> bool:
        """执行规划的路径"""
        if result["status"] != "Success":
            if self.debug:
                print(f"路径规划失败: {result['status']}")
            return False
        
        n_steps = result["position"].shape[0]
        
        for i in range(n_steps):
            qpos = result["position"][i]
            
            # 根据控制模式构建动作
            if self.base_env.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][i]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 渲染（如果需要）
            if hasattr(self.base_env, 'render_human'):
                try:
                    self.base_env.render_human()
                except:
                    pass
        
        return True
    
    def grasp_object(self, target_pose: sapien.Pose, use_screw: bool = False) -> bool:
        """抓取物体的完整流程"""
        # 1. 打开夹爪
        self.open_gripper()
        
        # 2. 规划到预抓取位置（略高于目标）
        pre_grasp_pose = sapien.Pose(
            p=target_pose.p + np.array([0, 0, 0.1]),  # 上方10cm
            q=target_pose.q
        )
        
        if use_screw:
            result = self.plan_screw_motion(pre_grasp_pose)
        else:
            result = self.plan_to_pose(pre_grasp_pose)
        
        if not self.execute_path(result):
            return False
        
        # 3. 螺旋运动到抓取位置
        result = self.plan_screw_motion(target_pose)
        if not self.execute_path(result):
            return False
        
        # 4. 关闭夹爪
        self.close_gripper()
        
        # 5. 提升物体
        lift_pose = sapien.Pose(
            p=target_pose.p + np.array([0, 0, 0.15]),
            q=target_pose.q
        )
        
        result = self.plan_to_pose(lift_pose)
        return self.execute_path(result)
    
    def open_gripper(self, steps: int = 6):
        """打开夹爪"""
        self.gripper_state = OPEN_GRIPPER
        self._execute_gripper_action(steps)
    
    def close_gripper(self, steps: int = 6):
        """关闭夹爪"""
        self.gripper_state = CLOSE_GRIPPER
        self._execute_gripper_action(steps)
    
    def _execute_gripper_action(self, steps: int):
        """执行夹爪动作"""
        current_qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        
        for _ in range(steps):
            if self.base_env.control_mode == "pd_joint_pos":
                action = np.hstack([current_qpos, self.gripper_state])
            else:
                action = np.hstack([current_qpos, current_qpos * 0, self.gripper_state])
            
            obs, reward, terminated, truncated, info = self.env.step(action)


@register_env(
    "EnvClutter-RRT-v1",
    asset_download_ids=["ycb"],
    max_episode_steps=50,  # 减少最大步数，因为每步包含完整抓取序列
)
class EnvClutterRRTEnv(EnvClutterEnv):
    """
    集成RRT运动规划的EnvClutter环境
    结合RL决策和运动规划，实现智能抓取顺序和安全路径规划
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda"]  # 目前只支持Panda机械臂
    
    def __init__(
        self,
        use_rrt_planning: bool = True,
        enable_obstacle_detection: bool = True,
        visualize_grasp: bool = True,
        planning_debug: bool = False,
        **kwargs
    ):
        # 强制使用连续控制模式，因为RRT规划器需要关节位置控制
        kwargs["control_mode"] = "pd_joint_pos"
        kwargs["use_discrete_action"] = False  # RRT版本使用连续控制
        
        super().__init__(**kwargs)
        
        # RRT规划配置
        self.use_rrt_planning = use_rrt_planning and MPLIB_AVAILABLE
        self.enable_obstacle_detection = enable_obstacle_detection
        self.visualize_grasp = visualize_grasp
        self.planning_debug = planning_debug
        
        # 初始化RRT规划器
        if self.use_rrt_planning:
            self.motion_planner = RRTMotionPlanner(
                env=self,
                visualize_grasp=self.visualize_grasp,
                debug=self.planning_debug
            )
        else:
            print("⚠️ RRT规划器未启用，将使用基础控制")
            self.motion_planner = None
        
        # RL模型相关
        self.rl_model = None
        self.decision_mode = "manual"  # "manual", "rl_model", "greedy"
        
        # 执行状态
        self.current_target = None
        self.grasp_phase = "idle"  # "idle", "planning", "grasping", "lifting"
        
        print(f"✅ EnvClutter-RRT环境初始化完成")
        print(f"   RRT规划: {'✅启用' if self.use_rrt_planning else '❌禁用'}")
        print(f"   障碍检测: {'✅启用' if self.enable_obstacle_detection else '❌禁用'}")
    
    def load_rl_model(self, model_path: str):
        """加载训练好的RL模型用于决策"""
        try:
            if MASKABLE_AVAILABLE:
                try:
                    self.rl_model = MaskablePPO.load(model_path)
                    print(f"✅ 加载MaskablePPO模型: {model_path}")
                except:
                    self.rl_model = PPO.load(model_path)
                    print(f"✅ 加载PPO模型: {model_path}")
            else:
                self.rl_model = PPO.load(model_path)
                print(f"✅ 加载PPO模型: {model_path}")
            
            self.decision_mode = "rl_model"
            
        except Exception as e:
            print(f"❌ RL模型加载失败: {e}")
            self.rl_model = None
            self.decision_mode = "manual"
    
    def _decide_next_target(self, obs) -> Optional[int]:
        """决定下一个抓取目标"""
        if self.decision_mode == "rl_model" and self.rl_model is not None:
            # 使用RL模型决策
            action, _ = self.rl_model.predict(obs, deterministic=True)
            target_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            
            # 验证目标是否有效
            if target_idx < len(self.selectable_objects[0]) and target_idx not in self.grasped_objects[0]:
                return target_idx
        
        elif self.decision_mode == "greedy":
            # 贪心策略：选择最高的物体
            available_objects = []
            for obj_idx, obj in enumerate(self.selectable_objects[0]):
                if obj_idx not in self.grasped_objects[0]:
                    height = obj.pose.p[2].item()
                    available_objects.append((obj_idx, height))
            
            if available_objects:
                # 选择最高的物体
                target_idx = max(available_objects, key=lambda x: x[1])[0]
                return target_idx
        
        # 手动模式或失败情况，返回None
        return None
    
    def _update_obstacle_detection(self):
        """更新障碍物检测"""
        if not self.enable_obstacle_detection or not self.motion_planner:
            return
        
        # 清除之前的障碍物
        self.motion_planner.clear_obstacles()
        
        # 添加托盘作为障碍物
        tray_pose = self.traybox.pose
        tray_size = np.array([self.TRAY_SIZE_X, self.TRAY_SIZE_Y, 0.02])  # 薄盒子
        self.motion_planner.add_box_obstacle(tray_size, tray_pose, sample_points=128)
        
        # 添加其他物体作为障碍物
        for env_idx in range(self.num_envs):
            for obj_idx, obj in enumerate(self.selectable_objects[env_idx]):
                if obj_idx not in self.grasped_objects[env_idx]:
                    # 获取物体尺寸和位置
                    try:
                        # 简化：使用球形障碍物代替复杂几何
                        obj_pos = obj.pose.p.cpu().numpy()
                        # 创建小盒子障碍物
                        box_size = np.array([0.05, 0.05, 0.05])  # 5cm立方体
                        obj_pose = sapien.Pose(p=obj_pos)
                        self.motion_planner.add_box_obstacle(box_size, obj_pose, sample_points=64)
                    except:
                        pass  # 忽略错误
    
    def step(self, action):
        """重写step方法，集成RRT规划"""
        if not self.use_rrt_planning:
            # 回退到基础环境
            return super().step(action)
        
        # RRT规划模式
        if self.grasp_phase == "idle":
            # 决定下一个抓取目标
            obs = self._get_obs()
            target_idx = self._decide_next_target(obs)
            
            if target_idx is None:
                # 没有可抓取目标，episode结束
                info = self._get_info()
                return obs, 0, True, False, info
            
            self.current_target = target_idx
            self.grasp_phase = "planning"
        
        if self.grasp_phase == "planning":
            # 更新障碍物检测
            self._update_obstacle_detection()
            
            # 获取目标位置和计算抓取位姿
            target_obj = self.selectable_objects[0][self.current_target]
            target_pose = self._compute_grasp_pose(target_obj)
            
            # 执行抓取
            success = self.motion_planner.grasp_object(target_pose, use_screw=True)
            
            if success:
                # 更新抓取状态
                self.grasped_objects[0].add(self.current_target)
                self.grasp_phase = "idle"
                
                # 计算奖励
                reward = self._compute_grasp_reward()
                
                # 检查episode是否完成
                done = len(self.grasped_objects[0]) >= self.total_objects_per_env
                
            else:
                # 抓取失败
                reward = -1.0
                done = False
                self.grasp_phase = "idle"
        
        else:
            # 其他阶段，继续当前动作
            reward = 0
            done = False
        
        # 获取新观测和信息
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _compute_grasp_pose(self, target_obj: Actor) -> sapien.Pose:
        """计算目标物体的抓取位姿"""
        # 获取物体位置
        obj_pos = target_obj.pose.p
        if isinstance(obj_pos, torch.Tensor):
            obj_pos = obj_pos.cpu().numpy()
        
        # 简单策略：从上方抓取
        grasp_pos = obj_pos.copy()
        grasp_pos[2] += 0.02  # 略高于物体表面
        
        # 垂直向下的抓取姿态
        grasp_quat = np.array([0, 1, 0, 0])  # 垂直向下
        
        return sapien.Pose(p=grasp_pos, q=grasp_quat)
    
    def _compute_grasp_reward(self) -> float:
        """计算抓取奖励"""
        # 基础成功奖励
        reward = 5.0
        
        # 根据抓取的物体高度给予额外奖励
        if self.current_target is not None:
            try:
                target_obj = self.selectable_objects[0][self.current_target]
                obj_height = target_obj.pose.p[2].item()
                height_bonus = min(obj_height / 0.2, 1.0) * 3.0
                reward += height_bonus
            except:
                pass
        
        # 完成所有物体的奖励
        if len(self.grasped_objects[0]) >= self.total_objects_per_env:
            reward += 20.0
        
        return reward
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = super().reset(**kwargs)
        
        # 重置RRT相关状态
        self.current_target = None
        self.grasp_phase = "idle"
        
        if self.motion_planner:
            self.motion_planner.clear_obstacles()
        
        return obs, info