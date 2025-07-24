import os
from typing import Any, Dict, List, Union, Tuple
import numpy as np
import sapien
import torch
import cv2
import random

import mani_skill.envs.utils.randomization as randomization
from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# 新增：IK和控制器相关导入
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController

# 新增：运动规划器相关导入
from motionplanner import PandaArmMotionPlanningSolver


class MotionPlannerWrapper:
    """运动规划辅助类，封装MPLib运动规划器的抓取逻辑"""
    
    def __init__(self, env, debug=False):
        self.env = env
        self.base_env = env.unwrapped
        self.debug = debug
        self.planner = None
        self.initialized = False
        
    def initialize(self):
        """初始化运动规划器"""
        if not self.initialized:
            try:
                # 检查环境和agent是否准备就绪
                if not hasattr(self.base_env, 'agent') or self.base_env.agent is None:
                    print("环境agent未准备就绪，无法初始化运动规划器")
                    return
                
                if not hasattr(self.base_env.agent, 'robot') or self.base_env.agent.robot is None:
                    print("机器人未准备就绪，无法初始化运动规划器")
                    return
                
                # 检查机器人关节状态
                try:
                    qpos = self.base_env.agent.robot.get_qpos()
                    if qpos is None:
                        print("机器人关节状态为None，无法初始化运动规划器")
                        return
                    print(f"机器人关节状态形状: {qpos.shape}")
                except Exception as e:
                    print(f"获取机器人关节状态失败: {e}")
                    return
                
                self.planner = PandaArmMotionPlanningSolver(
                    env=self.env,
                    debug=self.debug,
                    vis=False,  # 关闭可视化以提高速度
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                    joint_vel_limits=0.9,
                    joint_acc_limits=0.9,
                    base_pose=sapien.Pose(p=[-0.615, 0, 0]),  # 提供机器人基座位姿
                )
                self.initialized = True
                print("运动规划器初始化成功")
            except Exception as e:
                print(f"运动规划器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self.initialized = False
    
    def plan_grasp_sequence(self, target_object_idx: int, goal_pos: torch.Tensor) -> Tuple[bool, float]:
        """
        执行完整的抓取序列：抓-提-放
        
        Args:
            target_object_idx: 目标物体索引
            goal_pos: 目标放置位置
            
        Returns:
            success: 抓取是否成功
            displacement: 其他物体的位移增量
        """
        if not self.initialized:
            self.initialize()
            
        if not self.initialized or self.planner is None:
            print("运动规划器未初始化，回退到原始IK方法")
            return False, 0.0
        
        # 获取目标物体
        if target_object_idx >= len(self.base_env.all_objects):
            return False, 0.0
        
        target_obj = self.base_env.all_objects[target_object_idx]
        
        # 记录抓取前其他物体的位置
        other_objects_pos_before = []
        for i, obj in enumerate(self.base_env.all_objects):
            if i != target_object_idx and i not in self.base_env.grasped_objects:
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[0]
                other_objects_pos_before.append(obj_pos.clone())
        
        success = False
        try:
            # 获取目标物体位置
            target_pos = target_obj.pose.p
            if target_pos.dim() > 1:
                target_pos = target_pos[0]
            target_pos = target_pos.clone()
            
            # === 状态机执行抓取序列 ===
            
            # 状态0：移动到物体上方（安全高度）
            above_pos = target_pos.clone()
            above_pos[2] += 0.1  # 上方10cm
            above_pos_np = above_pos.cpu().numpy()
            if above_pos_np.ndim > 1:
                above_pos_np = above_pos_np.flatten()
            above_pose = sapien.Pose(p=above_pos_np, q=[1, 0, 0, 0])
            
            result = self.planner.move_to_pose_with_RRTConnect(above_pose, dry_run=False)
            if result == -1:
                print(f"无法移动到物体{target_object_idx}上方")
                return False, 0.0
            
            # 状态1：下降到抓取位置
            grasp_pos = target_pos.clone()
            grasp_pos[2] += 0.03  # 略微高于物体表面
            grasp_pos_np = grasp_pos.cpu().numpy()
            if grasp_pos_np.ndim > 1:
                grasp_pos_np = grasp_pos_np.flatten()
            grasp_pose = sapien.Pose(p=grasp_pos_np, q=[1, 0, 0, 0])
            
            result = self.planner.move_to_pose_with_screw(grasp_pose, dry_run=False)
            if result == -1:
                print(f"无法移动到物体{target_object_idx}抓取位置")
                return False, 0.0
            
            # 状态2：闭合夹爪
            self.planner.close_gripper()
            
            # 检查是否成功抓取
            tcp_pos = self.base_env.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                tcp_pos = tcp_pos[0]
            
            obj_pos = target_obj.pose.p
            if obj_pos.dim() > 1:
                obj_pos = obj_pos[0]
            
            tcp_to_obj_dist = torch.linalg.norm(tcp_pos - obj_pos)
            
            if tcp_to_obj_dist > 0.08:  # 8cm内认为抓取成功
                print(f"抓取物体{target_object_idx}失败，距离过远: {tcp_to_obj_dist:.3f}m")
                self.planner.open_gripper()
                return False, 0.0
            
            # 状态3：提升到安全高度
            lift_pos = grasp_pos.clone()
            lift_pos[2] += 0.15  # 上方15cm
            lift_pos_np = lift_pos.cpu().numpy()
            if lift_pos_np.ndim > 1:
                lift_pos_np = lift_pos_np.flatten()
            lift_pose = sapien.Pose(p=lift_pos_np, q=[1, 0, 0, 0])
            
            result = self.planner.move_to_pose_with_screw(lift_pose, dry_run=False)
            if result == -1:
                print(f"无法提升物体{target_object_idx}")
                self.planner.open_gripper()
                return False, 0.0
            
            # 状态4：移动到目标区域上方
            goal_above_pos = goal_pos.clone()
            if goal_above_pos.dim() > 1:
                goal_above_pos = goal_above_pos[0]
            goal_above_pos[2] += 0.15  # 上方15cm
            goal_above_pos_np = goal_above_pos.cpu().numpy()
            if goal_above_pos_np.ndim > 1:
                goal_above_pos_np = goal_above_pos_np.flatten()
            goal_above_pose = sapien.Pose(p=goal_above_pos_np, q=[1, 0, 0, 0])
            
            result = self.planner.move_to_pose_with_RRTConnect(goal_above_pose, dry_run=False)
            if result == -1:
                print(f"无法移动到目标区域上方")
                self.planner.open_gripper()
                return False, 0.0
            
            # 状态5：下降到放置位置
            goal_place_pos = goal_pos.clone()
            if goal_place_pos.dim() > 1:
                goal_place_pos = goal_place_pos[0]
            goal_place_pos[2] += 0.05  # 目标位置上方5cm
            goal_place_pos_np = goal_place_pos.cpu().numpy()
            if goal_place_pos_np.ndim > 1:
                goal_place_pos_np = goal_place_pos_np.flatten()
            goal_place_pose = sapien.Pose(p=goal_place_pos_np, q=[1, 0, 0, 0])
            
            result = self.planner.move_to_pose_with_screw(goal_place_pose, dry_run=False)
            if result == -1:
                print(f"无法移动到目标放置位置")
                self.planner.open_gripper()
                return False, 0.0
            
            # 状态6：打开夹爪放置物体
            self.planner.open_gripper()
            
            # 状态7：稍微后退到安全位置
            retreat_pos = goal_place_pos.clone()
            retreat_pos[2] += 0.10  # 上方10cm
            retreat_pos_np = retreat_pos.cpu().numpy()
            if retreat_pos_np.ndim > 1:
                retreat_pos_np = retreat_pos_np.flatten()
            retreat_pose = sapien.Pose(p=retreat_pos_np, q=[1, 0, 0, 0])
            
            self.planner.move_to_pose_with_screw(retreat_pose, dry_run=False)
            
            # 标记抓取成功
            success = True
            self.base_env.grasped_objects.append(target_object_idx)
            
            print(f"成功使用运动规划器抓取并放置物体{target_object_idx}")
            
        except Exception as e:
            print(f"运动规划抓取过程中出现错误: {e}")
            success = False
            # 尝试打开夹爪
            try:
                if self.planner:
                    self.planner.open_gripper()
            except:
                pass
        
        # 计算其他物体的位移
        displacement = 0.0
        other_objects_pos_after = []
        for i, obj in enumerate(self.base_env.all_objects):
            if i != target_object_idx and i not in self.base_env.grasped_objects:
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[0]
                other_objects_pos_after.append(obj_pos.clone())
        
        # 计算位移增量
        if len(other_objects_pos_before) == len(other_objects_pos_after):
            for pos_before, pos_after in zip(other_objects_pos_before, other_objects_pos_after):
                displacement += torch.linalg.norm(pos_after - pos_before).item()
        
        return success, displacement
    
    def close(self):
        """清理资源"""
        if self.planner:
            self.planner.close()
            self.planner = None
        self.initialized = False


@register_env(
    "EnvClutter-v2",
    asset_download_ids=["ycb"],
    max_episode_steps=200,
)
class EnvClutterEnv(BaseEnv):
    """
    **任务描述:**
    复杂堆叠抓取环境，包含各种形状的YCB物体堆积在托盘中。
    机械臂需要挑选最适合抓取的物体，并将其放到指定位置。
    
    **随机化:**
    - 物体在托盘内随机生成
    - 物体初始姿态随机化
    - 目标位置固定在托盘右侧
    
    **成功条件:**
    - 目标物体被成功抓取并放置到目标位置
    - 机器人静止
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # YCB物体
    BOX_OBJECTS = [
        #"003_cracker_box",          # 饼干盒
        "004_sugar_box",            # 糖盒
        "006_mustard_bottle",       # 芥末瓶
        "008_pudding_box",      # 布丁盒
        #"009_gelatin_box",          # 明胶盒
        #"010_potted_meat_can",      # 罐装肉罐头
    ]
    
    goal_thresh = 0.03  # 成功阈值
    # 托盘参数 (基于traybox.urdf的尺寸)
    tray_size = [0.6, 0.6, 0.15]  # 托盘内部尺寸 (长x宽x高)
    tray_spawn_area = [0.23, 0.23]  # 托盘内物体生成区域 (考虑边界墙和安全边距)
    num_objects_per_type = 5  # 每种类型的物体数量
    
    # 新增：离散动作相关常量
    MAX_N = len(BOX_OBJECTS) * num_objects_per_type  # 最大物体数量
    MAX_EPISODE_STEPS = 15  # 最大episode步数
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        use_discrete_action=False,  # 新增：是否使用离散动作
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_discrete_action = use_discrete_action
        
        # 初始化离散动作相关变量
        self.remaining_indices = []  # 剩余可抓取物体的索引
        self.step_count = 0  # 当前步数
        self.grasped_objects = []  # 已抓取的物体
        
        # 新增：控制器和IK相关变量
        self.arm_controller = None  # 将在_load_agent后初始化
        self.q_init = None  # 初始关节角
        self.q_above = None  # 目标区域上方关节角
        self.q_goal = None  # 目标区域关节角
        
        # 新增：运动规划器初始化
        self.motion_planner = None  # 将在_load_agent后初始化
        self.use_motion_planner = True  # 是否使用运动规划器
        
        # 确保所有参数正确传递给父类
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", 
            pose=pose, 
            width=512, 
            height=512, 
            fov=1, 
            near=0.01, 
            far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
        
        # 新增：初始化控制器缓存
        self.arm_controller = self.agent.controller.controllers["arm"]
        
        # 新增：初始化运动规划器
        if self.use_motion_planner:
            try:
                self.motion_planner = MotionPlannerWrapper(self, debug=False)
                print("运动规划器包装器创建成功")
            except Exception as e:
                print(f"运动规划器包装器创建失败: {e}")
                self.motion_planner = None
                self.use_motion_planner = False
        
        # 预计算关节角
        self._precompute_joint_positions()
    
    def _world_to_robot_frame(self, world_pose: Pose) -> Pose:
        """
        将世界坐标系中的位姿转换到机械臂基座坐标系
        
        Args:
            world_pose: 世界坐标系中的位姿
            
        Returns:
            robot_pose: 机械臂基座坐标系中的位姿
        """
        try:
            # 获取机械臂基座在世界坐标系中的位姿
            robot_base_pose = self.agent.robot.pose
            
            # 将世界坐标转换到机械臂基座坐标系
            # robot_pose = robot_base_pose.inv() * world_pose
            robot_pose = robot_base_pose.inv() * world_pose
            
            return robot_pose
        except Exception as e:
            print(f"坐标系转换失败: {e}")
            return world_pose  # 转换失败时返回原始位姿
    
    def _precompute_joint_positions(self):
        """预计算目标区域的关节角位置"""
        # 使用Panda机器人的默认rest关节角度作为初值
        # 从Panda机器人的keyframes中获取rest姿态
        rest_qpos = self.agent.keyframes["rest"].qpos
        print(f"Panda rest qpos: {rest_qpos}")
        
        # 只取前7个关节角（arm关节）
        arm_q_init = torch.tensor(rest_qpos[:7], device=self.device, dtype=torch.float32)
        
        # 保存初始关节角
        self.q_init = arm_q_init
        
        print(f"使用rest关节角作为初值，维度: {arm_q_init.shape}, 值: {arm_q_init}")
        
        # 机械臂位于[-0.615, 0, 0]，工作空间大约是0.8m半径
        # 修改目标位置到机械臂工作空间内
        goal_above_pos = torch.tensor([0.0, 0.3, 0.15], device=self.device)  # 目标位置上方10cm
        goal_above_quat = self._rpy_to_quat(torch.tensor([0.0, np.pi/2, 0.0], device=self.device))  # 垂直向下
        goal_above_pose = Pose.create_from_pq(p=goal_above_pos, q=goal_above_quat)
        
        # # 转换到根坐标系
        # raw_root_pose = self.arm_controller.articulation.pose.raw_pose
        # if raw_root_pose.dim() > 1:
        #     raw_root_pose = raw_root_pose[0]
        # print(f"根坐标系: {raw_root_pose}")
        # root_transform = Pose.create(raw_root_pose).inv()
        # goal_above_pose = root_transform * goal_above_pose
        
        # 选择：是否使用坐标系转换
        USE_COORDINATE_TRANSFORM = False  # 设置为True可以启用坐标系转换
        
        if USE_COORDINATE_TRANSFORM:
            goal_above_pose = self._world_to_robot_frame(goal_above_pose)
            print("使用坐标系转换")
        else:
            print("直接使用世界坐标系")
        
        # 计算IK
        self.q_above = self._compute_ik(goal_above_pose, arm_q_init)
        
        # 计算目标区域的关节角
        goal_pos = torch.tensor([0.0, 0.3, 0.05], device=self.device)  # 目标位置
        goal_quat = self._rpy_to_quat(torch.tensor([0.0, np.pi/2, 0.0], device=self.device))  # 垂直向下
        goal_pose = Pose.create_from_pq(p=goal_pos, q=goal_quat)
        
        if USE_COORDINATE_TRANSFORM:
            goal_pose = self._world_to_robot_frame(goal_pose)
        
        # 计算IK
        self.q_goal = self._compute_ik(goal_pose, arm_q_init)
        
        # 检查IK解是否有效
        if self.q_above is None or self.q_goal is None:
            print("警告：无法计算目标位置的IK解，使用默认关节角")
            self.q_above = arm_q_init.clone()
            self.q_goal = arm_q_init.clone()
        else:
            print(f"IK计算成功 - 目标上方: {self.q_above.shape}, 目标位置: {self.q_goal.shape}")
    
    def _rpy_to_quat(self, euler: torch.Tensor) -> torch.Tensor:
        """将欧拉角转换为四元数"""
        r, p, y = torch.unbind(euler, dim=-1)
        cy = torch.cos(y * 0.5)
        sy = torch.sin(y * 0.5)
        cp = torch.cos(p * 0.5)
        sp = torch.sin(p * 0.5)
        cr = torch.cos(r * 0.5)
        sr = torch.sin(r * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        quaternion = torch.stack([qw, qx, qy, qz], dim=-1)
        return quaternion
    
    def _compute_ik(self, target_pose: Pose, q0: torch.Tensor) -> torch.Tensor:
        """计算逆运动学解"""
        try:
            # 确保输入维度正确 - 使用7个arm关节角
            if q0.dim() == 1:
                q0 = q0.unsqueeze(0)
            
            # 只使用前7个关节角作为初值（arm关节）
            if q0.shape[-1] > 7:
                q0 = q0[:, :7]
            elif q0.shape[-1] < 7:
                # 如果维度不足，填充到7维
                padding = torch.zeros(q0.shape[0], 7 - q0.shape[-1], device=q0.device)
                q0 = torch.cat([q0, padding], dim=-1)
                print("填充到7维")
            
            # 计算IK
            ik_result = self.arm_controller.kinematics.compute_ik(
                target_pose=target_pose,
                q0=q0,
            )
            
            if ik_result is not None:
                if ik_result.dim() > 1:
                    return ik_result[0]  # 返回第一个解
                return ik_result
            else:
                print("IK计算返回None")
                return None
        except Exception as e:
            print(f"IK计算失败: {e}")
            return None
    
    def _move_arm(self, target_pose: Pose, steps: int = 10) -> bool:
        """使用PDEEPoseController移动机械臂到目标位置"""
        try:
            # 获取当前末端执行器位置
            end_effector_link_name = "panda_hand_tcp"  # TCP 点更精确

            # 通过 links_map 获取末端连杆
            end_effector_link = self.arm_controller.articulation.links_map[end_effector_link_name]

            # 获取末端执行器的位置（pose.p 是 3 维位置坐标 [x, y, z]）
            current_pose = end_effector_link.pose
            
            print(f"开始移动机械臂，目标位置: {target_pose.p}, 当前位置: {current_pose.p}")
            
            # 计算总的位置差异
            total_pos_diff = target_pose.p - current_pose.p
            total_distance = torch.linalg.norm(total_pos_diff)
            
            print(f"总距离: {total_distance:.4f}m")
            
            # 如果距离很小，直接认为成功
            if total_distance < 0.02:
                print("已经在目标位置附近")
                return True
            
            # 使用更大的步长和更高效的控制
            for step in range(steps):
                # 重新获取当前位置
                end_effector_link = self.arm_controller.articulation.links_map[end_effector_link_name]
                current_pose = end_effector_link.pose
                
                # 计算当前位置差异
                pos_diff = target_pose.p - current_pose.p
                current_distance = torch.linalg.norm(pos_diff)
                
                # 如果已经足够接近，提前退出
                if current_distance < 0.02:
                    print(f"成功到达目标位置，误差: {current_distance:.4f}m")
                    return True
                
                # 使用自适应步长：距离越远，步长越大
                if current_distance > 0.1:
                    scale_factor = 0.5  # 大步长
                elif current_distance > 0.05:
                    scale_factor = 0.3  # 中步长
                else:
                    scale_factor = 0.1  # 小步长
                
                # 计算姿态差异（简化处理）
                quat_diff = torch.zeros(3, device=self.device)
                
                # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
                action = torch.zeros(7, device=self.device)
                action[:3] = pos_diff * scale_factor  # 自适应位置增量
                action[3:6] = quat_diff  # 姿态控制
                action[6] = 0.0  # 保持夹爪状态
                
                # 执行多步以加快收敛
                for _ in range(3):  # 每个循环执行3步
                    self._low_level_step(action)
                
                # 打印进度
                # if step % 2 == 0:
                #     print(f"步骤 {step}: 位置误差 {current_distance:.4f}m, 步长因子 {scale_factor:.2f}")
            
            # 检查最终误差
            end_effector_link = self.arm_controller.articulation.links_map[end_effector_link_name]
            final_pose = end_effector_link.pose
            final_error = torch.linalg.norm(target_pose.p - final_pose.p)
            print(f"移动完成，最终误差: {final_error:.4f}m")
            
            # 放宽成功条件到5cm
            return final_error < 0.05
            
        except Exception as e:
            print(f"机械臂移动失败: {e}")
            return False
    
    def _low_level_step(self, delta_pose: torch.Tensor):
        """单步执行delta pose，只推进仿真，不走离散逻辑"""
        # 调用父类的step方法执行连续动作
        super().step(delta_pose)
    
    def _goto_qpos(self, target_qpos: torch.Tensor, steps: int = 10) -> bool:
        """快速移动到指定关节角位置"""
        try:
            if target_qpos is None:
                print("目标关节角为None")
                return False
            
            print(f"开始移动到关节角位置: {target_qpos}")
            
            # 确保维度正确
            if target_qpos.dim() == 1:
                target_qpos = target_qpos.unsqueeze(0)
            
            # 设置关节角目标
            for step in range(steps):
                # 使用关节角控制
                self.arm_controller.articulation.set_qpos(target_qpos)
                
                # 推进仿真
                self.scene.step()
                
                # 检查是否到达目标
                current_qpos = self.agent.robot.get_qpos()
                if current_qpos.dim() > 1:
                    current_qpos = current_qpos[0]
                current_arm_qpos = current_qpos[:7]  # 前7个关节角
                
                # 确保target_qpos也是7维
                target_arm_qpos = target_qpos[0] if target_qpos.dim() > 1 else target_qpos
                if target_arm_qpos.shape[-1] > 7:
                    target_arm_qpos = target_arm_qpos[:7]
                
                error = torch.linalg.norm(target_arm_qpos - current_arm_qpos)
                if error < 0.1:  # 关节角误差小于0.1弧度
                    print(f"成功到达目标关节角，误差: {error:.4f}rad")
                    return True
                
                if step % 5 == 0:
                    print(f"步骤 {step}: 关节角误差 {error:.4f}rad")
            
            # 检查最终误差
            final_qpos = self.agent.robot.get_qpos()
            if final_qpos.dim() > 1:
                final_qpos = final_qpos[0]
            final_arm_qpos = final_qpos[:7]
            
            final_error = torch.linalg.norm(target_arm_qpos - final_arm_qpos)
            print(f"关节角移动完成，最终误差: {final_error:.4f}rad")
            
            return final_error < 0.2  # 最终误差容忍度
            
        except Exception as e:
            print(f"关节角移动失败: {e}")
            return False

    def _load_scene(self, options: dict):
        # 构建桌面场景
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        
        # 加载托盘
        self._load_tray()
        
        # 创建物体列表
        self.all_objects = []
        self.selectable_objects = []
        self.object_info = []  # 存储物体信息
        
        # 为每个环境创建物体
        for env_idx in range(self.num_envs):
            env_objects = []
            env_selectable = []
            env_info = []
            
            # 创建每种类型的物体
            for obj_type in self.BOX_OBJECTS:
                for i in range(self.num_objects_per_type):
                    # 创建物体
                    builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj_type}")
                    
                    # 在托盘内随机生成位置
                    x, y, z = self._generate_object_position_in_tray(i)
                    
                    # 随机姿态
                    quat = randomization.random_quaternions(1)[0]
                    initial_pose = sapien.Pose(p=[x, y, z], q=quat.cpu().numpy())
                    
                    builder.initial_pose = initial_pose
                    builder.set_scene_idxs([env_idx])
                    
                    obj_name = f"env_{env_idx}_{obj_type}_{i}"
                    obj = builder.build(name=obj_name)
                    
                    env_objects.append(obj)
                    env_selectable.append(obj)
                    
                    # 存储物体信息
                    obj_info = {
                        'type': obj_type,
                        'size': self._get_object_size(obj_type),
                        'initial_pose': initial_pose,
                        'center': [x, y, z],
                        'exposed_area': 1.0,  # 初始暴露面积，后续会计算
                    }
                    env_info.append(obj_info)
            
            self.all_objects.extend(env_objects)
            self.selectable_objects.append(env_selectable)
            self.object_info.append(env_info)
        
        # 合并所有物体
        if self.all_objects:
            self.merged_objects = Actor.merge(self.all_objects, name="all_objects")
        
        # 创建目标位置标记
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        
        # 初始化目标物体相关变量
        self.target_object = None
        self.target_object_indices = []

    def _load_tray(self):
        """加载托盘URDF文件"""
        # 获取托盘URDF文件路径
        tray_urdf_path = "/home2/jzh/RL_RobotArm-main/assets/tray/traybox.urdf"
        
        if not os.path.exists(tray_urdf_path):
            raise FileNotFoundError(f"托盘URDF文件未找到: {tray_urdf_path}")
        
        # 创建URDF加载器
        loader = self.scene.create_urdf_loader()
        
        # 设置托盘的物理属性
        loader.set_material(static_friction=0.8, dynamic_friction=0.6, restitution=0.1)
        loader.fix_root_link = True  # 固定托盘不动
        loader.scale = 1.0  # 保持原始尺寸
        
        # 解析URDF文件
        parsed_result = loader.parse(tray_urdf_path)
        
        # 只使用 actor_builders 方式
        actor_builders = parsed_result.get("actor_builders", [])
        
        if not actor_builders:
            raise ValueError("托盘URDF文件中没有找到actor_builders")
        
        self.trays = []
        
        # 使用 actor_builders 加载托盘
        print("使用actor builders加载托盘")
        for env_idx in range(self.num_envs):
            builder = actor_builders[0]
            # 设置托盘位置 (放在桌面上，机器人前方)
            tray_position = [0.1, 0.0, 0.02]  # 桌面高度加上托盘底部厚度
            builder.initial_pose = sapien.Pose(p=tray_position)
            builder.set_scene_idxs([env_idx])
            
            # 使用 build_static 创建静态托盘，确保不会移动
            tray = builder.build_static(name=f"tray_{env_idx}")
            self.trays.append(tray)
        
        # 合并所有托盘
        if self.trays:
            self.merged_trays = Actor.merge(self.trays, name="all_trays")
        
        print(f"成功加载托盘，共 {len(self.trays)} 个")

    def _generate_object_position_in_tray(self, stack_level=0):
        """在托盘内生成物体位置"""
        # 托盘中心位置
        tray_center_x = 0.1
        tray_center_y = 0.0
        tray_bottom_z = 0.02 + 0.03  # 托盘底部 + 小偏移
        
        # 托盘边界计算（基于URDF文件中的边界墙位置）
        # 边界墙在托盘中心的±0.25米处，考虑边界墙厚度0.02米
        # 实际可用空间：从中心向两边各0.23米（留出安全边距）
        safe_spawn_area_x = 0.23
        safe_spawn_area_y = 0.23
        
        # 在托盘内随机生成xy位置
        x = tray_center_x + random.uniform(-safe_spawn_area_x, safe_spawn_area_x)
        y = tray_center_y + random.uniform(-safe_spawn_area_y, safe_spawn_area_y)
        
        # 堆叠高度
        z = tray_bottom_z + stack_level * 0.02  # 每层高度
        
        return x, y, z

    def _get_object_size(self, obj_type):
        """获取物体的大小信息"""
        # 基于YCB数据集的实际物体尺寸（单位：米）
        sizes = {
            #"003_cracker_box": [0.16, 0.21, 0.07],         # 饼干盒: 16cm x 21cm x 7cm
            "004_sugar_box": [0.09, 0.175, 0.044],         # 糖盒: 9cm x 17.5cm x 4.4cm
            "006_mustard_bottle": [0.095, 0.095, 0.177],   # 芥末瓶: 9.5cm x 9.5cm x 17.7cm
            "008_pudding_box": [0.078, 0.109, 0.032],      # 布丁盒: 7.8cm x 10.9cm x 3.2cm
            #"009_gelatin_box": [0.028, 0.085, 0.114],      # 明胶盒: 2.8cm x 8.5cm x 11.4cm  
            #"010_potted_meat_can": [0.101, 0.051, 0.051],  # 罐装肉罐头: 10.1cm x 5.1cm x 5.1cm
           
        }
        return sizes.get(obj_type, [0.05, 0.05, 0.05])

    def _sample_target_objects(self):
        """随机选择目标物体"""
        target_objects = []
        self.target_object_indices = []
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.selectable_objects) and self.selectable_objects[env_idx]:
                # 随机选择一个可选择的物体
                target_idx = random.randint(0, len(self.selectable_objects[env_idx]) - 1)
                target_obj = self.selectable_objects[env_idx][target_idx]
                target_objects.append(target_obj)
                self.target_object_indices.append(target_idx)
        
        if target_objects:
            self.target_object = Actor.merge(target_objects, name="target_object")

    def _calculate_exposed_area(self, env_idx):
        """计算物体的暴露面积"""
        # 这里是简化的暴露面积计算
        # 实际应用中可能需要更复杂的几何计算
        if env_idx < len(self.object_info):
            for i, obj_info in enumerate(self.object_info[env_idx]):
                # 基于物体高度和周围物体数量的简单估算
                exposed_area = max(0.1, 1.0 - i * 0.1)  # 越高的物体暴露面积越大
                obj_info['exposed_area'] = exposed_area

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            # 重置托盘位置
            if hasattr(self, 'merged_trays'):
                # 在GPU仿真中，静态对象不能改变位姿，所以跳过
                if not self.scene.gpu_sim_enabled:
                    if b == self.num_envs:
                        self.merged_trays.pose = self.merged_trays.initial_pose
                    else:
                        mask = torch.isin(self.merged_trays._scene_idxs, env_idx)
                        self.merged_trays.pose = self.merged_trays.initial_pose[mask]
                else:
                    print("GPU仿真模式下跳过静态托盘位姿重置")
            
            # 重置物体到初始位置
            if hasattr(self, 'merged_objects'):
                if b == self.num_envs:
                    self.merged_objects.pose = self.merged_objects.initial_pose
                else:
                    mask = torch.isin(self.merged_objects._scene_idxs, env_idx)
                    self.merged_objects.pose = self.merged_objects.initial_pose[mask]
            
            # 设置目标位置 - 固定在托盘右侧
            goal_pos = torch.zeros((b, 3), device=self.device)
            
            # 托盘中心位置：[0.1, 0.0, 0.02]
            # 托盘尺寸：长0.6m，宽0.6m
            # 托盘右侧边界：0.1 + 0.3 = 0.4m
            # 目标位置设定在托盘右侧外10cm处，避免与托盘边界冲突
            goal_pos[:, 0] = 0.0  # 托盘右侧的固定位置
            goal_pos[:, 1] = 0.3  # y方向居中，与托盘中心对齐
            goal_pos[:, 2] = 0.05  # 桌面高度5cm，确保物体稳定放置
            
            self.goal_pos = goal_pos
            self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))
            
            # 记录初始物体位置（用于计算位移）
            self.initial_object_positions = []
            for i in range(b):
                env_positions = []
                for obj in self.all_objects:
                    if hasattr(obj, '_scene_idxs') and len(obj._scene_idxs) > 0:
                        if obj._scene_idxs[0] == env_idx[i]:
                            env_positions.append(obj.pose.p.clone())
                self.initial_object_positions.append(env_positions)
            
            # 计算暴露面积
            for i in range(b):
                self._calculate_exposed_area(env_idx[i])
            
            # 重新选择目标物体
            self._sample_target_objects()
            
            # 新增：初始化离散动作相关变量
            if self.use_discrete_action:
                self.remaining_indices = list(range(self.MAX_N))
                self.step_count = 0
                self.grasped_objects = []
            
            # 新增：重置机械臂到初始IK pose
            if self.q_init is not None:
                try:
                    # 获取机器人的实际DOF
                    robot_dof = self.agent.robot.dof[0].item()  # 获取单个环境的DOF
                    print(f"机器人实际DOF: {robot_dof}")
                    
                    # 构建完整的初始关节角
                    # Panda: 7个arm关节 + 2个gripper关节 = 9个DOF
                    if robot_dof == 9:
                        # 7个arm关节 + 2个gripper关节（打开状态：0.04, 0.04）
                        full_qpos = torch.zeros(robot_dof, device=self.device)
                        full_qpos[:7] = self.q_init  # arm关节
                        full_qpos[7:9] = torch.tensor([0.04, 0.04], device=self.device)  # gripper关节
                    else:
                        # 如果DOF不是9，使用默认方式
                        full_qpos = torch.zeros(robot_dof, device=self.device)
                        min_len = min(len(self.q_init), robot_dof)
                        full_qpos[:min_len] = self.q_init[:min_len]
                    
                    # 扩展到批次维度
                    init_qpos = full_qpos.unsqueeze(0).repeat(len(env_idx), 1)
                    
                    print(f"构建的初始qpos维度: {init_qpos.shape}, 目标形状: [{len(env_idx)}, {robot_dof}]")
                    
                    # 重置机械臂
                    self.agent.reset(init_qpos=init_qpos)
                    print(f"机械臂已重置到初始IK pose")
                except Exception as e:
                    print(f"重置机械臂到初始pose失败: {e}")
                    print(f"q_init维度: {self.q_init.shape if self.q_init is not None else 'None'}")
                    print(f"机器人DOF: {self.agent.robot.dof}")
                    # 使用默认重置
                    self.agent.reset()

    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息"""
        # 获取批次大小
        batch_size = self.num_envs
        
        # 基础观测信息
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        
        if "state" in self.obs_mode:
            if hasattr(self, 'target_object') and self.target_object is not None:
                obs.update(
                    target_obj_pose=self.target_object.pose.raw_pose,
                    tcp_to_obj_pos=self.target_object.pose.p - self.agent.tcp.pose.p,
                    obj_to_goal_pos=self.goal_site.pose.p - self.target_object.pose.p,
                )
            else:
                # 如果没有目标物体，提供零张量以保持维度一致
                zero_pose = torch.zeros((batch_size, 7), device=self.device)
                zero_pos = torch.zeros((batch_size, 3), device=self.device)
                obs.update(
                    target_obj_pose=zero_pose,
                    tcp_to_obj_pos=zero_pos,
                    obj_to_goal_pos=zero_pos,
                )
            
            # 添加标量观测信息，确保维度一致
            obs.update(
                num_objects=torch.tensor([len(self.all_objects)], device=self.device).repeat(batch_size),
            )
        
        # 新增：离散动作相关观测
        if self.use_discrete_action:
            # 创建掩码：未抓取=1，已抓取=0
            mask = torch.ones(batch_size, self.MAX_N, device=self.device)
            for i, grasped_idx in enumerate(self.grasped_objects):
                if grasped_idx < self.MAX_N:
                    mask[:, grasped_idx] = 0
            
            # 物体特征：中心坐标、尺寸、暴露面积
            object_features = torch.zeros(batch_size, self.MAX_N, 7, device=self.device)  # 3+3+1=7维特征
            
            for env_idx in range(batch_size):
                for obj_idx, obj in enumerate(self.all_objects):
                    if obj_idx < self.MAX_N and obj_idx not in self.grasped_objects:
                        # 获取物体信息
                        pos = obj.pose.p[env_idx] if len(obj.pose.p.shape) > 1 else obj.pose.p
                        
                        # 获取物体类型和尺寸
                        obj_type_idx = obj_idx // self.num_objects_per_type
                        if obj_type_idx < len(self.BOX_OBJECTS):
                            obj_type = self.BOX_OBJECTS[obj_type_idx]
                            size = self._get_object_size(obj_type)
                        else:
                            size = [0.05, 0.05, 0.05]  # 默认尺寸
                        
                        # 获取暴露面积
                        exposed_area = 1.0  # 简化处理
                        if hasattr(self, 'object_info') and env_idx < len(self.object_info):
                            if obj_idx < len(self.object_info[env_idx]):
                                exposed_area = self.object_info[env_idx][obj_idx].get('exposed_area', 1.0)
                        
                        # 组合特征：[x, y, z, w, h, d, exposed_area]
                        object_features[env_idx, obj_idx] = torch.tensor([
                            pos[0].item() if hasattr(pos[0], 'item') else pos[0],
                            pos[1].item() if hasattr(pos[1], 'item') else pos[1],
                            pos[2].item() if hasattr(pos[2], 'item') else pos[2],
                            size[0], size[1], size[2],
                            exposed_area
                        ], device=self.device)
            
            # 将离散动作相关的观测展平为1D向量，避免维度不匹配问题
            # 将action_mask展平为1D
            action_mask_flat = mask.flatten(start_dim=1)  # [batch_size, MAX_N]
            
            # 将object_features展平为1D
            object_features_flat = object_features.flatten(start_dim=1)  # [batch_size, MAX_N*7]
            
            # 将step_count扩展为与batch_size匹配的1D向量
            step_count_expanded = torch.tensor([self.step_count], device=self.device).repeat(batch_size).unsqueeze(1)  # [batch_size, 1]
            
            # 将所有离散动作相关的观测合并为一个统一的张量
            discrete_obs = torch.cat([
                action_mask_flat,
                object_features_flat,
                step_count_expanded
            ], dim=1)  # [batch_size, MAX_N + MAX_N*7 + 1]
            
            obs.update(
                discrete_action_obs=discrete_obs,
            )
            
            # 在离散动作模式下，将所有观测展平为单一张量以保持维度一致性
            # 按照固定顺序展平所有观测
            flattened_parts = []
            
            # 按照字母顺序处理各个键，确保顺序一致
            for key in sorted(obs.keys()):
                if key in ['sensor_data']:
                    continue
                value = obs[key]
                if isinstance(value, torch.Tensor):
                    flattened_parts.append(value.flatten())
                elif isinstance(value, np.ndarray):
                    flattened_parts.append(torch.from_numpy(value).flatten())
                else:
                    flattened_parts.append(torch.tensor([value]).flatten())
            
            # 合并所有部分
            flattened_obs = torch.cat(flattened_parts, dim=0)
            
            # 返回展平后的观测，确保维度一致
            return flattened_obs.unsqueeze(0)  # 添加batch维度
        
        return obs

    def _plan_grasp(self, object_idx: int) -> Tuple[bool, float]:
        """
        使用运动规划器抓取指定物体（优先），如果失败则回退到IK方法
        
        Args:
            object_idx: 物体索引
            
        Returns:
            success: 抓取是否成功
            displacement: 其他物体的位移增量
        """
        if object_idx >= len(self.all_objects):
            return False, 0.0
        
        # 优先使用运动规划器
        if self.use_motion_planner and self.motion_planner is not None:
            try:
                # 获取目标放置位置
                goal_pos = self.goal_pos[0] if hasattr(self, 'goal_pos') and self.goal_pos is not None else torch.tensor([0.0, 0.3, 0.05], device=self.device)
                
                success, displacement = self.motion_planner.plan_grasp_sequence(object_idx, goal_pos)
                
                if success:
                    print(f"运动规划器成功抓取物体{object_idx}")
                    return success, displacement
                else:
                    print(f"运动规划器抓取物体{object_idx}失败，回退到IK方法")
            except Exception as e:
                print(f"运动规划器抓取过程中出现错误: {e}，回退到IK方法")
        
        # 回退到原始IK方法
        return self._ik_grasp_fallback(object_idx)
    
    def _ik_grasp_fallback(self, object_idx: int) -> Tuple[bool, float]:
        """
        使用真实的IK+PD_EE_DELTA_POSE控制器抓取指定物体
        
        Args:
            object_idx: 物体索引
            
        Returns:
            success: 抓取是否成功
            displacement: 其他物体的位移增量
        """
        if object_idx >= len(self.all_objects):
            return False, 0.0
        
        # 获取目标物体
        target_obj = self.all_objects[object_idx]
        
        # 记录抓取前其他物体的位置
        other_objects_pos_before = []
        for i, obj in enumerate(self.all_objects):
            if i != object_idx and i not in self.grasped_objects:
                # 安全地获取物体位置
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    # 如果是批次张量，取第一个环境的位置
                    obj_pos = obj_pos[0]
                other_objects_pos_before.append(obj_pos.clone())
        
        # 真实的IK+控制器抓取流程
        success = False
        try:
            # 获取目标物体位置
            target_pos = target_obj.pose.p
            if target_pos.dim() > 1:
                target_pos = target_pos[0]
            target_pos = target_pos.clone()
            
            # === 阶段1：移动到物体上方 ===
            # 1.1 计算物体上方10cm的位置和姿态
            above_pos = target_pos.clone()
            above_pos[2] += 0.1  # 上方10cm
            above_quat = self._rpy_to_quat(torch.tensor([0.0, np.pi/2, 0.0], device=self.device))  # 垂直向下
            above_pose = Pose.create_from_pq(p=above_pos, q=above_quat)
            
            # # 转换到根坐标系
            # raw_root_pose = self.arm_controller.articulation.pose.raw_pose
            # if raw_root_pose.dim() > 1:
            #     raw_root_pose = raw_root_pose[0]
            # root_transform = Pose.create(raw_root_pose).inv()
            # above_pose = root_transform * above_pose
            
            # 选择：是否使用坐标系转换
            USE_COORDINATE_TRANSFORM = False  # 设置为True可以启用坐标系转换
            
            if USE_COORDINATE_TRANSFORM:
                above_pose = self._world_to_robot_frame(above_pose)
                print("使用坐标系转换")
            else:
                print("直接使用世界坐标系")
            
            # 1.2 使用_move_arm移动到物体上方
            if not self._move_arm(above_pose, steps=200):
                print(f"无法移动到物体{object_idx}上方")
                return False, 0.0
            
            # === 阶段2：下降到抓取位置 ===
            # 2.1 计算抓取位置（贴合物体）
            grasp_pos = target_pos.clone()
            grasp_pos[2] += 0.02  # 略微高于物体表面
            grasp_quat = self._rpy_to_quat(torch.tensor([0.0, np.pi/2, 0.0], device=self.device))  # 垂直向下
            grasp_pose = Pose.create_from_pq(p=grasp_pos, q=grasp_quat)
            
            # # 转换到根坐标系
            # grasp_pose = root_transform * grasp_pose
            
            if USE_COORDINATE_TRANSFORM:
                grasp_pose = self._world_to_robot_frame(grasp_pose)
            
            # 2.2 移动到抓取位置
            if not self._move_arm(grasp_pose, steps=200):
                print(f"无法移动到物体{object_idx}抓取位置")
                return False, 0.0
            
            # 2.3 闭合夹爪
            self._close_gripper()
            
            # 检查是否成功抓取（简化判断）
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                tcp_pos = tcp_pos[0]
            
            obj_pos = target_obj.pose.p
            if obj_pos.dim() > 1:
                obj_pos = obj_pos[0]
            
            tcp_to_obj_dist = torch.linalg.norm(tcp_pos - obj_pos)
            
            if tcp_to_obj_dist > 0.05:  # 5cm内认为抓取成功
                print(f"抓取物体{object_idx}失败，距离过远: {tcp_to_obj_dist:.3f}m")
                self._open_gripper()
                return False, 0.0
            
            # === 阶段3：提升到安全高度 ===
            # 3.1 移动到当前位置上方10cm
            current_pos = self.agent.tcp.pose.p
            if current_pos.dim() > 1:
                current_pos = current_pos[0]
            
            lift_pos = current_pos.clone()
            lift_pos[2] += 0.1  # 上方10cm
            lift_quat = self._rpy_to_quat(torch.tensor([0.0, np.pi/2, 0.0], device=self.device))  # 垂直向下
            lift_pose = Pose.create_from_pq(p=lift_pos, q=lift_quat)
            
            # # 转换到根坐标系
            # lift_pose = root_transform * lift_pose
            
            if USE_COORDINATE_TRANSFORM:
                lift_pose = self._world_to_robot_frame(lift_pose)
            
            # 3.2 提升
            if not self._move_arm(lift_pose, steps=200):
                print(f"无法提升物体{object_idx}")
                self._open_gripper()
                return False, 0.0
            
            # === 阶段4：使用预计算的关节角快速移动到目标区域 ===
            # 4.1 移动到目标区域上方
            if not self._move_arm(self.q_above, steps=200):
                print(f"无法移动到目标区域上方")
                self._open_gripper()
                return False, 0.0
            
            # 4.2 下降到目标位置
            if not self._move_arm(self.q_goal, steps=200):
                print(f"无法移动到目标位置")
                self._open_gripper()
                return False, 0.0
            
            # 4.3 打开夹爪放置物体
            self._open_gripper()
            
            # 4.4 稍微后退
            retreat_pos = self.agent.tcp.pose.p
            if retreat_pos.dim() > 1:
                retreat_pos = retreat_pos[0]
            retreat_pos = retreat_pos.clone()
            retreat_pos[2] += 0.05  # 上方5cm
            retreat_quat = self._rpy_to_quat(torch.tensor([0.0, np.pi/2, 0.0], device=self.device))  # 垂直向下
            retreat_pose = Pose.create_from_pq(p=retreat_pos, q=retreat_quat)
            
            # # 转换到根坐标系
            # retreat_pose = root_transform * retreat_pose
            
            if USE_COORDINATE_TRANSFORM:
                retreat_pose = self._world_to_robot_frame(retreat_pose)
            
            self._move_arm(retreat_pose, steps=200)
            
            # 标记抓取成功
            success = True
            self.grasped_objects.append(object_idx)
            
            print(f"成功使用IK方法抓取并放置物体{object_idx}")
            
        except Exception as e:
            print(f"IK抓取过程中出现错误: {e}")
            success = False
            # 尝试打开夹爪
            try:
                self._open_gripper()
            except:
                pass
        
        # 计算其他物体的位移
        displacement = 0.0
        other_objects_pos_after = []
        for i, obj in enumerate(self.all_objects):
            if i != object_idx and i not in self.grasped_objects:
                # 安全地获取物体位置
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos = obj_pos[0]
                other_objects_pos_after.append(obj_pos.clone())
        
        # 计算位移增量
        if len(other_objects_pos_before) == len(other_objects_pos_after):
            for pos_before, pos_after in zip(other_objects_pos_before, other_objects_pos_after):
                displacement += torch.linalg.norm(pos_after - pos_before).item()
        
        return success, displacement
    
    def _close_gripper(self):
        """闭合夹爪"""
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        action = torch.zeros(7, device=self.device)
        action[6] = 0.00  # 闭合夹爪
        
        # 执行多步以确保夹爪闭合
        for _ in range(5):
            self._low_level_step(action)
    
    def _open_gripper(self):
        """打开夹爪"""
        # 构建7维动作向量 [dx, dy, dz, drx, dry, drz, gripper]
        action = torch.zeros(7, device=self.device)
        action[6] = 0.04  # 打开夹爪
        
        # 执行多步以确保夹爪打开
        for _ in range(5):
            self._low_level_step(action)

    def step(self, action):
        """
        覆盖step方法以支持离散动作选择
        
        Args:
            action: 如果use_discrete_action=True，则为物体索引；否则为连续动作
        """
        if self.use_discrete_action:
            return self._discrete_step(action)
        else:
            # 调用父类的连续动作step
            return super().step(action)
    
    def _discrete_step(self, action):
        """
        处理离散动作的step方法
        
        Args:
            action: 要抓取的物体在remaining_indices中的索引
        """
        # 检查动作合法性
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        
        if action < 0 or action >= len(self.remaining_indices):
            # 非法动作，返回失败结果
            return self._get_failed_step_result()
        
        # 获取实际的物体索引
        target_idx = self.remaining_indices[action]
        
        # 执行抓取
        success, displacement = self._plan_grasp(target_idx)
        
        # 从剩余列表中移除已抓取的物体
        self.remaining_indices.pop(action)
        
        # 更新步数
        self.step_count += 1
        
        # 计算奖励
        reward = self.compute_select_reward(success, displacement)
        
        # 检查终止条件
        terminated = self.step_count >= self.MAX_EPISODE_STEPS or len(self.remaining_indices) == 0
        truncated = False
        
        # 获取新的观测
        info = self.evaluate()
        info.update({
            'success': success,
            'displacement': displacement,
            'remaining_objects': len(self.remaining_indices),
            'grasped_objects': len(self.grasped_objects),
        })
        
        obs = self._get_obs_extra(info)
        
        return obs, reward, terminated, truncated, info
    
    def _get_failed_step_result(self):
        """获取失败步骤的结果"""
        # 惩罚性奖励
        reward = -1.0
        
        # 不终止，让智能体学习
        terminated = False
        truncated = False
        
        # 获取当前观测
        info = self.evaluate()
        info.update({
            'success': False,
            'displacement': 0.0,
            'remaining_objects': len(self.remaining_indices),
            'grasped_objects': len(self.grasped_objects),
        })
        
        obs = self._get_obs_extra(info)
        
        return obs, reward, terminated, truncated, info

    def compute_select_reward(self, grasp_success: bool, displacement: float) -> float:
        """
        计算离散动作选择的奖励
        
        Args:
            grasp_success: 抓取是否成功
            displacement: 其他物体的位移
            
        Returns:
            reward: 奖励值
        """
        # 基础奖励：成功抓取
        reward = 1.0 if grasp_success else 0.0
        
        # 位移惩罚
        reward -= displacement * 1.5  # disp_coeff
        
        # 时间惩罚
        reward -= 0.01  # time_coeff
        
        return reward

    @property
    def discrete_action_space(self):
        """获取离散动作空间"""
        if self.use_discrete_action:
            import gymnasium as gym
            return gym.spaces.Discrete(self.MAX_N)
        else:
            return None

    def evaluate(self):
        """评估任务完成情况"""
        success = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_grasped = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_robot_static = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_obj_placed = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        
        if hasattr(self, 'target_object') and self.target_object is not None:
            # 检查物体是否放置到目标位置
            obj_to_goal_dist = torch.linalg.norm(
                self.goal_site.pose.p - self.target_object.pose.p, axis=1
            )
            is_obj_placed = obj_to_goal_dist <= self.goal_thresh
            
            # 检查是否抓取
            is_grasped = self.agent.is_grasping(self.target_object)
            
            # 检查机器人是否静止
            is_robot_static = self.agent.is_static(0.2)
            
            # 成功条件：物体放置到位且机器人静止
            success = is_obj_placed & is_robot_static
        
        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def _calculate_other_objects_displacement(self):
        """计算其他物体的位移距离"""
        total_displacement = torch.zeros(self.num_envs, device=self.device)
        
        for env_idx in range(self.num_envs):
            displacement = 0.0
            obj_count = 0
            
            for i, obj in enumerate(self.all_objects):
                if hasattr(obj, '_scene_idxs') and len(obj._scene_idxs) > 0:
                    if obj._scene_idxs[0] == env_idx:
                        # 跳过目标物体
                        if hasattr(self, 'target_object_indices') and env_idx < len(self.target_object_indices):
                            if i == self.target_object_indices[env_idx]:
                                continue
                        
                        # 计算位移
                        if hasattr(self, 'initial_object_positions') and env_idx < len(self.initial_object_positions):
                            if obj_count < len(self.initial_object_positions[env_idx]):
                                initial_pos = self.initial_object_positions[env_idx][obj_count]
                                current_pos = obj.pose.p
                                displacement += torch.linalg.norm(current_pos - initial_pos)
                                obj_count += 1
            
            total_displacement[env_idx] = displacement
        
        return total_displacement

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """计算密集奖励"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        if not hasattr(self, 'target_object') or self.target_object is None:
            return reward
        
        # 1. 接近奖励（优先级最高）
        tcp_to_obj_dist = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward * 2.0  # 权重2.0
        
        # 2. 抓取奖励
        is_grasped = info["is_grasped"]
        reward += is_grasped * 3.0  # 权重3.0
        
        # 3. 放置奖励
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.target_object.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped * 2.0  # 只有抓取时才给放置奖励
        
        # 4. 其他物体位移惩罚（优先级第二）
        other_displacement = self._calculate_other_objects_displacement()
        displacement_penalty = torch.tanh(other_displacement)
        reward -= displacement_penalty * 1.5  # 权重1.5
        
        # 5. 时间惩罚（优先级第三）
        time_penalty = 0.01  # 每步小惩罚
        reward -= time_penalty
        
        # 6. 静止奖励
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * 1.0
        
        # 7. 成功奖励
        reward[info["success"]] = 10.0
        
        return reward

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """计算稀疏奖励"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 只有成功时才给奖励
        reward[info["success"]] = 1.0
        
        # 其他物体位移惩罚
        other_displacement = self._calculate_other_objects_displacement()
        displacement_penalty = other_displacement * 0.1
        reward -= displacement_penalty
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """计算归一化密集奖励"""
        # 检查是否有 reward_mode 属性，如果没有则默认使用 dense 模式
        if hasattr(self, 'reward_mode') and self.reward_mode == "sparse":
            return self.compute_sparse_reward(obs=obs, action=action, info=info)
        else:
            return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0 

    def reset(self, seed=None, options=None):
        """重置环境，确保返回一致的观测结构"""
        # 调用父类的reset方法
        obs, info = super().reset(seed=seed, options=options)
        
        # 在离散动作模式下，确保返回一致的观测结构
        if self.use_discrete_action:
            # 获取评估信息
            eval_info = self.evaluate()
            
            # 使用我们的观测处理方法
            obs = self._get_obs_extra(eval_info)
        
        return obs, info
    
    def close(self):
        """清理环境资源"""
        # 清理运动规划器资源
        if hasattr(self, 'motion_planner') and self.motion_planner is not None:
            try:
                self.motion_planner.close()
                print("运动规划器资源已清理")
            except Exception as e:
                print(f"清理运动规划器资源时出错: {e}")
        
        # 调用父类的close方法
        super().close() 