"""
复杂堆叠环境Motion Planning求解器
基于PandaArmMotionPlanningSolver的高级封装，专门处理复杂堆叠场景

核心设计理念：
1. 分层抽象：将复杂堆叠分解为原子操作序列
2. 动态规划：实时更新碰撞约束和空间状态
3. 智能重试：多策略并行尝试，提升成功率
"""

import numpy as np
import sapien
import trimesh
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.structs import Actor

@dataclass
class StackingTarget:
    """堆叠目标定义"""
    source_obj: Actor
    target_obj: Actor  
    stack_height: float = 0.0  # 相对高度偏移
    approach_angles: List[float] = None  # 候选抓取角度

@dataclass  
class EnvironmentState:
    """环境状态快照"""
    objects: Dict[str, Actor]
    collision_meshes: List[trimesh.Trimesh]
    forbidden_regions: List[np.ndarray]  # 禁止区域点云

class ComplexStackingMotionPlanner:
    """复杂堆叠Motion Planning求解器"""
    
    def __init__(self, base_planner: PandaArmMotionPlanningSolver):
        self.planner = base_planner
        self.env_state = None
        self.execution_history = []
        self.safety_margin = 0.02  # 安全边界
        
    def solve_complex_stacking(self, stacking_sequence: List[StackingTarget]) -> bool:
        """
        求解复杂堆叠序列
        
        Args:
            stacking_sequence: 堆叠操作序列
            
        Returns:
            是否成功完成所有堆叠
        """
        self._initialize_environment()
        
        for i, target in enumerate(stacking_sequence):
            print(f"执行堆叠步骤 {i+1}/{len(stacking_sequence)}")
            
            # 更新环境状态
            self._update_collision_environment()
            
            # 多策略尝试
            success = self._execute_stacking_with_retry(target)
            
            if not success:
                print(f"堆叠步骤 {i+1} 失败，开始回退...")
                self._rollback_to_safe_state()
                return False
                
            self.execution_history.append(target)
            
        return True
    
    def _initialize_environment(self):
        """初始化环境状态"""
        # 清除之前的碰撞约束
        self.planner.clear_collisions()
        
        # 记录初始状态
        self.env_state = EnvironmentState(
            objects={},
            collision_meshes=[],
            forbidden_regions=[]
        )
    
    def _update_collision_environment(self):
        """动态更新碰撞环境"""
        # 为所有已堆叠的物体添加碰撞约束
        for history_item in self.execution_history:
            obj_obb = get_actor_obb(history_item.target_obj)
            collision_points = self._sample_object_surface(obj_obb)
            self.planner.add_collision_pts(collision_points)
    
    def _execute_stacking_with_retry(self, target: StackingTarget) -> bool:
        """
        带重试机制的堆叠执行
        
        核心策略：
        1. 计算最优抓取姿态
        2. 多角度尝试
        3. 路径规划优化
        4. 碰撞避免
        """
        # 计算基础抓取信息
        obb = get_actor_obb(target.source_obj)
        approaching = np.array([0, 0, -1])
        target_closing = self.planner.env_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, 
            target_closing=target_closing, 
            depth=0.025
        )
        
        base_grasp_pose = self.planner.env_agent.build_grasp_pose(
            grasp_info["approaching"], 
            grasp_info["closing"], 
            grasp_info["center"]
        )
        
        # 生成候选抓取角度
        candidate_angles = target.approach_angles or np.arange(0, 2*np.pi, np.pi/3)
        
        for angle in candidate_angles:
            # 尝试当前角度的抓取姿态
            rotation_delta = sapien.Pose(q=euler2quat(0, 0, angle))
            grasp_pose = base_grasp_pose * rotation_delta
            
            # 测试路径可行性
            if self._test_grasp_feasibility(grasp_pose):
                # 执行完整的抓取-堆叠序列
                if self._execute_pick_and_stack(grasp_pose, target):
                    return True
        
        return False
    
    def _test_grasp_feasibility(self, grasp_pose: sapien.Pose) -> bool:
        """测试抓取姿态的可行性"""
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        
        # 测试到达路径
        reach_result = self.planner.move_to_pose_with_screw(reach_pose, dry_run=True)
        if reach_result == -1:
            return False
            
        # 测试抓取路径  
        grasp_result = self.planner.move_to_pose_with_screw(grasp_pose, dry_run=True)
        return grasp_result != -1
    
    def _execute_pick_and_stack(self, grasp_pose: sapien.Pose, target: StackingTarget) -> bool:
        """执行完整的抓取和堆叠动作"""
        try:
            # 1. 到达预抓取位置
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
            if self.planner.move_to_pose_with_screw(reach_pose) == -1:
                return False
            
            # 2. 执行抓取
            if self.planner.move_to_pose_with_screw(grasp_pose) == -1:
                return False
            self.planner.close_gripper()
            
            # 3. 提升物体
            lift_pose = grasp_pose * sapien.Pose([0, 0, 0.1])  
            if self.planner.move_to_pose_with_screw(lift_pose) == -1:
                return False
            
            # 4. 移动到目标位置
            target_pose = self._compute_stacking_target_pose(target, grasp_pose)
            if self.planner.move_to_pose_with_screw(target_pose) == -1:
                return False
                
            # 5. 放置物体
            self.planner.open_gripper()
            
            return True
            
        except Exception as e:
            print(f"执行抓取堆叠时发生异常: {e}")
            return False
    
    def _compute_stacking_target_pose(self, target: StackingTarget, grasp_pose: sapien.Pose) -> sapien.Pose:
        """计算堆叠目标姿态"""
        # 获取目标物体的上表面位置
        target_pos = target.target_obj.pose.sp.p
        
        # 计算堆叠高度（物体尺寸 + 安全边界）
        source_obb = get_actor_obb(target.source_obj) 
        target_obb = get_actor_obb(target.target_obj)
        
        stack_height = target_obb.primitive.extents[2] + source_obb.primitive.extents[2]/2 + target.stack_height
        
        # 构造目标位置
        target_position = target_pos + np.array([0, 0, stack_height])
        
        return sapien.Pose(target_position, grasp_pose.q)
    
    def _sample_object_surface(self, obb: trimesh.primitives.Box, n_points: int = 256) -> np.ndarray:
        """采样物体表面点云用于碰撞检测"""
        mesh = obb.as_outline()
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        return points
    
    def _rollback_to_safe_state(self):
        """回退到安全状态"""
        # 实现回退逻辑：张开抓手，移动到安全位置
        self.planner.open_gripper()
        
        # 移动到预设的安全位置
        safe_pose = sapien.Pose([0, 0, 0.3])  # 高空安全位置
        self.planner.move_to_pose_with_RRTConnect(safe_pose)
        
        print("已回退到安全状态")


def create_complex_stacking_plan(objects: List[Actor], stacking_pattern: str = "pyramid") -> List[StackingTarget]:
    """
    创建复杂堆叠计划
    
    Args:
        objects: 待堆叠物体列表
        stacking_pattern: 堆叠模式 ("pyramid", "tower", "bridge"等)
    
    Returns:
        堆叠目标序列
    """
    if stacking_pattern == "pyramid":
        return _create_pyramid_pattern(objects)
    elif stacking_pattern == "tower":
        return _create_tower_pattern(objects)  
    elif stacking_pattern == "bridge":
        return _create_bridge_pattern(objects)
    else:
        raise ValueError(f"不支持的堆叠模式: {stacking_pattern}")

def _create_pyramid_pattern(objects: List[Actor]) -> List[StackingTarget]:
    """创建金字塔堆叠模式"""
    targets = []
    
    if len(objects) >= 3:
        # 底层：将物体A和B靠近
        targets.append(StackingTarget(
            source_obj=objects[0],
            target_obj=objects[1],
            stack_height=0.0  # 同一层
        ))
        
        # 顶层：将物体C堆叠在A和B的中间上方
        targets.append(StackingTarget(
            source_obj=objects[2], 
            target_obj=objects[0],  # 以A为参考
            stack_height=0.05,  # 额外高度
            approach_angles=np.arange(0, np.pi*2/3, np.pi/6)  # 更多角度尝试
        ))
    
    return targets

def _create_tower_pattern(objects: List[Actor]) -> List[StackingTarget]:
    """创建塔式堆叠模式"""
    targets = []
    
    for i in range(1, len(objects)):
        targets.append(StackingTarget(
            source_obj=objects[i],
            target_obj=objects[i-1],
            stack_height=0.02 * i  # 逐渐增加的高度补偿
        ))
    
    return targets

def _create_bridge_pattern(objects: List[Actor]) -> List[StackingTarget]:
    """创建桥式堆叠模式"""
    targets = []
    
    if len(objects) >= 3:
        # 将中间物体架在两端物体之间
        targets.append(StackingTarget(
            source_obj=objects[1],
            target_obj=objects[0],
            stack_height=0.1,
            approach_angles=[0, np.pi/2]  # 特定角度抓取
        ))
    
    return targets


