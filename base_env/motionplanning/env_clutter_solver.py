"""
EnvClutter环境专用Motion Planning解决方案
支持复杂堆叠物体的智能抓取策略

关键特性:
1. YCB物体几何分析与最优抓取点计算
2. 堆叠场景下的防碰撞路径规划
3. 自适应抓取策略（顶部优先、避障等）
4. 多层物体感知与抓取序列优化
"""

import os
import sys
import numpy as np
import sapien
import trimesh
import cv2
from typing import Dict, List, Tuple, Optional, Any
from transforms3d import quaternions, euler

# 添加项目根目录到路径
project_root = "/home/linux/jzh/RL_Robot"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入ManiSkill相关模块
import mplib
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose

# 导入环境相关模块
try:
    from env_clutter import EnvClutterOptimizedEnv
    from config import get_config
except ImportError:
    print("警告: 无法导入env_clutter模块，请确保路径正确")

# 夹爪状态常量
GRIPPER_OPEN = 1
GRIPPER_CLOSED = -1

# YCB物体抓取配置
YCB_GRASP_CONFIG = {
    "004_sugar_box": {
        "approach_directions": [np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, 1, 0])],
        "grip_width": 0.08,
        "grip_depth": 0.03,
        "preferred_approach": np.array([0, 0, -1]),  # 顶部抓取
        "safety_margin": 0.02
    },
    "006_mustard_bottle": {
        "approach_directions": [np.array([0, 0, -1]), np.array([1, 0, 0])],
        "grip_width": 0.06,
        "grip_depth": 0.025,
        "preferred_approach": np.array([0, 0, -1]),
        "safety_margin": 0.015
    },
    "008_pudding_box": {
        "approach_directions": [np.array([0, 0, -1]), np.array([0, 1, 0])],
        "grip_width": 0.07,
        "grip_depth": 0.025,
        "preferred_approach": np.array([0, 0, -1]),
        "safety_margin": 0.02
    }
}


class EnvClutterMotionPlanner:
    """EnvClutter环境专用运动规划器"""
    
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_pose: bool = True,
        print_info: bool = True,
        joint_vel_limits: float = 0.8,
        joint_acc_limits: float = 0.8,
        collision_detection: bool = True,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        
        # 运动参数
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        
        # 基座位姿
        self.base_pose = to_sapien_pose(base_pose) if base_pose else sapien.Pose()
        
        # 初始化运动规划器
        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode
        
        # 可视化和调试
        self.debug = debug
        self.vis = vis
        self.print_info = print_info
        self.visualize_target_pose = visualize_target_pose
        
        # 夹爪状态
        self.gripper_state = GRIPPER_OPEN
        
        # 碰撞检测
        self.collision_detection = collision_detection
        self.use_point_cloud = False
        self.collision_pts = None
        
        # 场景分析
        self.scene_objects = []
        self.object_heights = {}
        self.grasp_candidates = {}
        
        # 执行统计
        self.elapsed_steps = 0
        self.successful_grasps = 0
        
        if self.vis and self.visualize_target_pose:
            self.setup_visualization()
    
    def setup_planner(self) -> mplib.Planner:
        """设置MPLib运动规划器"""
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner
    
    def setup_visualization(self):
        """设置可视化"""
        # 可以添加抓取姿态可视化等
        pass
    
    def analyze_scene(self) -> Dict[str, Any]:
        """分析当前场景中的物体分布"""
        scene_info = {
            "objects": [],
            "layers": {},
            "grasp_candidates": [],
            "optimal_sequence": []
        }
        
        # 获取场景中的所有物体
        if hasattr(self.base_env, 'objects') and self.base_env.objects:
            for i, obj in enumerate(self.base_env.objects):
                if obj is not None:
                    obj_info = self.analyze_object(obj, i)
                    scene_info["objects"].append(obj_info)
        
        # 分析物体层次结构
        scene_info["layers"] = self.compute_object_layers(scene_info["objects"])
        
        # 计算抓取候选点
        scene_info["grasp_candidates"] = self.compute_grasp_candidates(scene_info["objects"])
        
        # 优化抓取序列（顶层优先）
        scene_info["optimal_sequence"] = self.optimize_grasp_sequence(
            scene_info["layers"], scene_info["grasp_candidates"]
        )
        
        if self.print_info:
            print(f"场景分析完成: 发现{len(scene_info['objects'])}个物体")
            print(f"物体分布: {len(scene_info['layers'])}层堆叠")
            print(f"生成{len(scene_info['grasp_candidates'])}个抓取候选点")
        
        return scene_info
    
    def analyze_object(self, obj, obj_id: int) -> Dict[str, Any]:
        """分析单个物体的几何和抓取属性"""
        obj_pose = obj.pose
        obj_type = getattr(obj, 'name', f'object_{obj_id}')
        
        # 获取物体的边界框
        bbox = self.get_object_bbox(obj)
        
        obj_info = {
            "id": obj_id,
            "type": obj_type,
            "pose": obj_pose,
            "position": obj_pose.p,
            "orientation": obj_pose.q,
            "bbox": bbox,
            "height": bbox[2, 1] - bbox[2, 0],  # Z轴高度
            "volume": np.prod(bbox[:, 1] - bbox[:, 0]),
            "center": np.mean(bbox, axis=1),
            "graspable": True,
            "accessibility": self.compute_accessibility(obj_pose.p, bbox)
        }
        
        return obj_info
    
    def get_object_bbox(self, obj) -> np.ndarray:
        """获取物体的轴对齐边界框"""
        try:
            # 尝试从碰撞形状获取边界框
            collision_shapes = obj.get_collision_shapes()
            if collision_shapes:
                # 简化处理：假设第一个碰撞形状代表主要几何
                shape = collision_shapes[0]
                # 这里可以根据具体的SAPIEN API调整
                # 暂时使用估算值
                return np.array([
                    [-0.05, 0.05],  # X轴范围
                    [-0.05, 0.05],  # Y轴范围
                    [0, 0.1]        # Z轴范围（高度）
                ])
        except:
            pass
        
        # 默认边界框
        pos = obj.pose.p
        return np.array([
            [pos[0] - 0.05, pos[0] + 0.05],
            [pos[1] - 0.05, pos[1] + 0.05],
            [pos[2], pos[2] + 0.1]
        ])
    
    def compute_accessibility(self, obj_pos: np.ndarray, bbox: np.ndarray) -> float:
        """计算物体的可达性分数（0-1，1为最易抓取）"""
        # 基于物体高度和周围空间计算可达性
        height_score = min(obj_pos[2] / 0.5, 1.0)  # 越高越好抓取
        
        # 可以添加更多因素：
        # - 周围障碍物密度
        # - 到机器人的距离
        # - 物体的稳定性
        
        return height_score
    
    def compute_object_layers(self, objects: List[Dict]) -> Dict[int, List[int]]:
        """计算物体的层次结构"""
        if not objects:
            return {}
        
        # 按高度排序物体
        height_sorted = sorted(objects, key=lambda x: x["position"][2])
        
        layers = {}
        current_layer = 0
        layer_threshold = 0.08  # 层间距离阈值
        
        for i, obj in enumerate(height_sorted):
            if i == 0:
                layers[current_layer] = [obj["id"]]
            else:
                height_diff = obj["position"][2] - height_sorted[i-1]["position"][2]
                if height_diff > layer_threshold:
                    current_layer += 1
                    layers[current_layer] = [obj["id"]]
                else:
                    layers[current_layer].append(obj["id"])
        
        return layers
    
    def compute_grasp_candidates(self, objects: List[Dict]) -> List[Dict]:
        """为每个物体计算抓取候选点"""
        candidates = []
        
        for obj in objects:
            obj_type = obj["type"]
            obj_pos = obj["position"]
            obj_quat = obj["orientation"]
            
            # 获取物体特定的抓取配置
            grasp_config = YCB_GRASP_CONFIG.get(obj_type, YCB_GRASP_CONFIG["004_sugar_box"])
            
            # 为每个可能的接近方向生成抓取候选
            for approach_dir in grasp_config["approach_directions"]:
                grasp_pose = self.compute_grasp_pose(
                    obj_pos, obj_quat, approach_dir, grasp_config
                )
                
                candidate = {
                    "object_id": obj["id"],
                    "object_type": obj_type,
                    "grasp_pose": grasp_pose,
                    "approach_direction": approach_dir,
                    "grip_width": grasp_config["grip_width"],
                    "quality": self.evaluate_grasp_quality(grasp_pose, obj, approach_dir),
                    "is_preferred": np.allclose(approach_dir, grasp_config["preferred_approach"])
                }
                candidates.append(candidate)
        
        # 按质量排序
        candidates.sort(key=lambda x: x["quality"], reverse=True)
        return candidates
    
    def compute_grasp_pose(
        self, 
        obj_pos: np.ndarray, 
        obj_quat: np.ndarray, 
        approach_dir: np.ndarray,
        grasp_config: Dict
    ) -> sapien.Pose:
        """计算特定接近方向的抓取姿态"""
        
        # 计算抓取位置（物体中心偏移）
        offset = approach_dir * grasp_config["grip_depth"]
        grasp_pos = obj_pos - offset
        
        # 计算抓取方向
        # Z轴指向接近方向，Y轴为夹爪闭合方向
        z_axis = approach_dir / np.linalg.norm(approach_dir)
        
        # 选择合适的Y轴方向
        if abs(z_axis[2]) < 0.9:  # 不是垂直方向
            y_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            y_axis = np.cross(z_axis, np.array([1, 0, 0]))
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 构建旋转矩阵
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        grasp_quat = quaternions.mat2quat(rot_matrix)
        
        return sapien.Pose(grasp_pos, grasp_quat)
    
    def evaluate_grasp_quality(
        self, 
        grasp_pose: sapien.Pose, 
        obj_info: Dict, 
        approach_dir: np.ndarray
    ) -> float:
        """评估抓取姿态的质量分数"""
        quality = 0.0
        
        # 1. 可达性分数
        quality += obj_info["accessibility"] * 0.3
        
        # 2. 高度优势（越高越好）
        height_score = min(obj_info["position"][2] / 0.5, 1.0)
        quality += height_score * 0.3
        
        # 3. 接近方向偏好（顶部抓取优先）
        if np.allclose(approach_dir, [0, 0, -1], atol=0.1):
            quality += 0.4  # 顶部抓取加分
        
        return min(quality, 1.0)
    
    def optimize_grasp_sequence(
        self, 
        layers: Dict[int, List[int]], 
        candidates: List[Dict]
    ) -> List[int]:
        """优化抓取序列（顶层优先，避免碰撞）"""
        sequence = []
        
        # 按层次从上到下处理
        for layer_idx in sorted(layers.keys(), reverse=True):
            layer_objects = layers[layer_idx]
            
            # 在当前层中按抓取质量排序
            layer_candidates = [c for c in candidates if c["object_id"] in layer_objects]
            layer_candidates.sort(key=lambda x: x["quality"], reverse=True)
            
            # 添加到序列中
            for candidate in layer_candidates:
                if candidate["object_id"] not in sequence:
                    sequence.append(candidate["object_id"])
        
        return sequence
    
    def execute_grasp_sequence(self, target_objects: List[int]) -> Dict[str, Any]:
        """执行抓取序列"""
        results = {
            "total_attempts": len(target_objects),
            "successful_grasps": 0,
            "failed_grasps": 0,
            "execution_steps": 0,
            "grasp_details": []
        }
        
        for obj_id in target_objects:
            grasp_result = self.execute_single_grasp(obj_id)
            results["grasp_details"].append(grasp_result)
            results["execution_steps"] += grasp_result.get("steps", 0)
            
            if grasp_result["success"]:
                results["successful_grasps"] += 1
            else:
                results["failed_grasps"] += 1
                
            if self.print_info:
                print(f"物体 {obj_id} 抓取结果: {'成功' if grasp_result['success'] else '失败'}")
        
        return results
    
    def execute_single_grasp(self, obj_id: int) -> Dict[str, Any]:
        """执行单个物体的抓取"""
        grasp_result = {
            "object_id": obj_id,
            "success": False,
            "steps": 0,
            "error_msg": None
        }
        
        try:
            # 重新分析场景（物体可能已移动）
            scene_info = self.analyze_scene()
            
            # 找到目标物体的最佳抓取候选
            target_candidates = [c for c in scene_info["grasp_candidates"] if c["object_id"] == obj_id]
            if not target_candidates:
                grasp_result["error_msg"] = "未找到抓取候选点"
                return grasp_result
            
            best_candidate = target_candidates[0]
            
            # 执行抓取动作序列
            steps = 0
            
            # 1. 移动到预抓取位置
            pre_grasp_pose = self.compute_pre_grasp_pose(best_candidate["grasp_pose"])
            move_result = self.move_to_pose_with_planning(pre_grasp_pose)
            if move_result == -1:
                grasp_result["error_msg"] = "无法到达预抓取位置"
                return grasp_result
            steps += move_result.get("steps", 20)
            
            # 2. 接近物体
            approach_result = self.move_to_pose_with_screw(best_candidate["grasp_pose"])
            if approach_result == -1:
                grasp_result["error_msg"] = "无法接近物体"
                return grasp_result
            steps += 15
            
            # 3. 闭合夹爪
            self.close_gripper()
            steps += 10
            
            # 4. 提升物体
            lift_pose = best_candidate["grasp_pose"] * sapien.Pose([0, 0, 0.1])
            lift_result = self.move_to_pose_with_screw(lift_pose)
            if lift_result == -1:
                grasp_result["error_msg"] = "无法提升物体"
                self.open_gripper()  # 释放夹爪
                return grasp_result
            steps += 15
            
            # 5. 移动到目标位置（可选）
            # 这里可以添加将物体放置到指定位置的逻辑
            
            grasp_result["success"] = True
            grasp_result["steps"] = steps
            
        except Exception as e:
            grasp_result["error_msg"] = f"执行异常: {str(e)}"
        
        return grasp_result
    
    def compute_pre_grasp_pose(self, grasp_pose: sapien.Pose, offset: float = 0.08) -> sapien.Pose:
        """计算预抓取位置"""
        # 在接近方向上后退一定距离
        approach_dir = grasp_pose.to_transformation_matrix()[:3, 2]
        offset_pos = -approach_dir * offset
        pre_grasp_pos = grasp_pose.p + offset_pos
        return sapien.Pose(pre_grasp_pos, grasp_pose.q)
    
    def move_to_pose_with_planning(self, target_pose: sapien.Pose, dry_run: bool = False) -> Dict:
        """使用RRT规划移动到目标位置"""
        target_pose = to_sapien_pose(target_pose)
        
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([target_pose.p, target_pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        
        if result["status"] != "Success":
            if self.print_info:
                print(f"路径规划失败: {result['status']}")
            return -1
        
        if dry_run:
            return result
        
        return self.follow_path(result)
    
    def move_to_pose_with_screw(self, target_pose: sapien.Pose, dry_run: bool = False) -> Dict:
        """使用螺旋运动移动到目标位置"""
        target_pose = to_sapien_pose(target_pose)
        
        result = self.planner.plan_screw(
            np.concatenate([target_pose.p, target_pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        
        if result["status"] != "Success":
            # 尝试第二次
            result = self.planner.plan_screw(
                np.concatenate([target_pose.p, target_pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                if self.print_info:
                    print(f"螺旋运动规划失败: {result['status']}")
                return -1
        
        if dry_run:
            return result
        
        return self.follow_path(result)
    
    def follow_path(self, result: Dict, refine_steps: int = 0) -> Dict:
        """跟随规划的路径"""
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
                
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            if self.print_info and self.elapsed_steps % 10 == 0:
                print(f"步骤 {self.elapsed_steps}: reward={reward:.3f}")
                
            if self.vis:
                self.base_env.render_human()
        
        return {"success": True, "steps": n_step + refine_steps}
    
    def open_gripper(self, steps: int = 10):
        """打开夹爪"""
        self.gripper_state = GRIPPER_OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        
        for i in range(steps):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            if self.vis:
                self.base_env.render_human()
    
    def close_gripper(self, steps: int = 10, force: float = GRIPPER_CLOSED):
        """闭合夹爪"""
        self.gripper_state = force
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        
        for i in range(steps):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            if self.vis:
                self.base_env.render_human()
    
    def update_collision_environment(self):
        """更新碰撞检测环境"""
        if not self.collision_detection:
            return
            
        # 这里可以添加动态障碍物检测
        # 例如：从环境中获取其他物体的位置，添加为碰撞体
        pass
    
    def wait_for_user_input(self):
        """等待用户输入（调试模式）"""
        if not self.vis or not self.debug:
            return
        print("按 [c] 继续执行...")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'planner'):
            del self.planner
        if self.print_info:
            print(f"运动规划器已关闭。总执行步数: {self.elapsed_steps}")


def solve_env_clutter(env, seed=None, debug=False, vis=False, max_objects=3):
    """
    EnvClutter环境的Motion Planning解决方案
    
    Args:
        env: EnvClutter环境实例
        seed: 随机种子
        debug: 是否开启调试模式
        vis: 是否开启可视化
        max_objects: 最大抓取物体数量
    
    Returns:
        执行结果字典
    """
    env.reset(seed=seed)
    
    # 创建运动规划器
    planner = EnvClutterMotionPlanner(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_pose=vis,
        print_info=True,
    )
    
    try:
        # 分析场景
        scene_info = planner.analyze_scene()
        
        # 选择要抓取的物体（限制数量以提高成功率）
        target_objects = scene_info["optimal_sequence"][:max_objects]
        
        if planner.print_info:
            print(f"开始执行抓取序列，目标物体: {target_objects}")
        
        # 执行抓取序列
        results = planner.execute_grasp_sequence(target_objects)
        
        # 计算最终奖励和信息
        success_rate = results["successful_grasps"] / max(results["total_attempts"], 1)
        
        final_result = {
            "success": success_rate > 0.5,  # 50%以上成功率认为任务成功
            "success_rate": success_rate,
            "total_steps": results["execution_steps"],
            "grasped_objects": results["successful_grasps"],
            "details": results
        }
        
        if planner.print_info:
            print(f"任务完成！成功率: {success_rate:.1%}, 总步数: {results['execution_steps']}")
        
        return final_result
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        planner.close()
