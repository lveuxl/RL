"""
运动规划器模块
实现基于IK的轨迹规划和执行功能
"""

import numpy as np
import torch
import sapien
from typing import List, Tuple, Optional, Dict, Any
from mani_skill.utils.structs import Pose
from mani_skill.utils import common
import time

class SimpleMotionPlanner:
    """简单的运动规划器，基于IK和线性插值"""
    
    def __init__(self, env, resolution=0.05, max_iter=1000):
        self.env = env
        self.resolution = resolution
        self.max_iter = max_iter
        self.robot = env.agent.robot
        self.scene = env.scene
        
    def plan_straight_line(self, q_start: np.ndarray, q_goal: np.ndarray, 
                          n_steps: int = 50) -> Tuple[bool, List[np.ndarray]]:
        """
        规划直线路径（关节空间）
        
        Args:
            q_start: 起始关节角度
            q_goal: 目标关节角度  
            n_steps: 插值步数
            
        Returns:
            (success, path): 成功标志和路径点列表
        """
        # 检查起始和目标配置是否有效
        if not self._is_valid_config(q_start) or not self._is_valid_config(q_goal):
            return False, []
        
        # 线性插值生成路径
        path = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q_interp = q_start + alpha * (q_goal - q_start)
            
            # 检查中间点是否有效
            if not self._is_valid_config(q_interp):
                return False, []
            
            path.append(q_interp)
        
        return True, path
    
    def plan_cartesian_path(self, tcp_start: Pose, tcp_goal: Pose, 
                           n_steps: int = 50) -> Tuple[bool, List[np.ndarray]]:
        """
        规划笛卡尔空间路径
        
        Args:
            tcp_start: 起始TCP位姿
            tcp_goal: 目标TCP位姿
            n_steps: 插值步数
            
        Returns:
            (success, path): 成功标志和关节角度路径
        """
        # 获取当前关节角度
        q_current = self.robot.get_qpos()[:7].cpu().numpy()  # 只取前7个关节
        
        path = []
        
        # 在笛卡尔空间插值
        for i in range(n_steps + 1):
            alpha = i / n_steps
            
            # 位置插值
            pos_interp = tcp_start.p + alpha * (tcp_goal.p - tcp_start.p)
            
            # 四元数插值（球面线性插值）
            quat_interp = self._slerp_quaternion(tcp_start.q, tcp_goal.q, alpha)
            
            # 构造中间位姿
            pose_interp = Pose.create_from_pq(
                torch.tensor(pos_interp, device=self.env.device),
                torch.tensor(quat_interp, device=self.env.device)
            )
            
            # 计算IK
            q_target = self.env.agent.controller.controllers['arm'].kinematics.compute_ik(
                pose_interp, 
                torch.tensor(q_current, device=self.env.device)[None]
            )
            
            if q_target is None:
                print(f"IK求解失败，步骤 {i}")
                return False, []
            
            q_target_np = q_target[0].cpu().numpy()
            
            # 检查配置是否有效
            if not self._is_valid_config(q_target_np):
                print(f"无效配置，步骤 {i}")
                return False, []
            
            path.append(q_target_np)
            q_current = q_target_np  # 更新当前配置
        
        return True, path
    
    def _is_valid_config(self, q: np.ndarray) -> bool:
        """检查关节配置是否有效（碰撞检测）"""
        # 简化的碰撞检测 - 只检查关节限制
        joint_limits = self.robot.get_qlimits()[0, :7].cpu().numpy()  # 转换为numpy数组
        
        # 确保q是一维数组
        if q.ndim > 1:
            q = q.flatten()
        
        for i, q_val in enumerate(q):
            if i >= len(joint_limits):
                break
            q_min, q_max = joint_limits[i]  # 每个关节的限制是[min, max]
            
            # 确保q_val是标量
            if hasattr(q_val, 'item'):
                q_val = q_val.item()
            
            if q_val < q_min or q_val > q_max:
                return False
        
        # TODO: 添加更复杂的碰撞检测
        return True
    
    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """球面线性插值四元数"""
        # 确保四元数是单位四元数
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算点积
        dot = np.dot(q1, q2)
        
        # 如果点积为负，反转一个四元数以选择较短路径
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # 如果四元数非常接近，使用线性插值
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # 计算角度
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2


class TrajectoryPlayer:
    """轨迹播放器"""
    
    def __init__(self, env, control_freq=20):
        self.env = env
        self.control_dt = 1.0 / control_freq
        
    def play_trajectory(self, q_trajectory: List[np.ndarray], 
                       suction_commands: Optional[List[bool]] = None) -> bool:
        """
        播放关节轨迹
        
        Args:
            q_trajectory: 关节角度轨迹
            suction_commands: 吸盘命令序列（可选）
            
        Returns:
            成功标志
        """
        try:
            for i, q in enumerate(q_trajectory):
                # 构造动作
                action = torch.tensor(q, device=self.env.device, dtype=torch.float32)
                
                # 添加吸盘控制
                if suction_commands and i < len(suction_commands):
                    suction_val = 1.0 if suction_commands[i] else 0.0
                    action = torch.cat([action, torch.tensor([suction_val], device=self.env.device)])
                
                # 执行动作
                self.env.step(action)
                
                # 控制执行频率
                time.sleep(self.control_dt)
            
            return True
            
        except Exception as e:
            print(f"轨迹播放失败: {e}")
            return False


class GraspPlanner:
    """抓取规划器"""
    
    def __init__(self, env):
        self.env = env
        self.planner = SimpleMotionPlanner(env)
        self.player = TrajectoryPlayer(env)
        
    def plan_and_execute_grasp(self, obj_actor, place_pos: np.ndarray) -> bool:
        """
        规划并执行抓取动作
        
        Args:
            obj_actor: 目标物体
            place_pos: 放置位置
            
        Returns:
            成功标志
        """
        try:
            # 获取当前机械臂状态
            current_q = self.env.agent.robot.get_qpos()[:7].cpu().numpy()
            
            # 1. 规划到预抓取位置
            obj_pose = obj_actor.pose
            pre_grasp_pos = obj_pose.p.cpu().numpy() + np.array([0, 0, 0.10])  # 上方10cm
            pre_grasp_quat = np.array([0, 1, 0, 0])  # 垂直向下
            
            pre_grasp_pose = Pose.create_from_pq(
                torch.tensor(pre_grasp_pos, device=self.env.device),
                torch.tensor(pre_grasp_quat, device=self.env.device)
            )
            
            # 计算预抓取关节角度
            q_pre_grasp = self.env.agent.controller.controllers['arm'].kinematics.compute_ik(
                pre_grasp_pose,
                torch.tensor(current_q, device=self.env.device)[None]
            )
            
            if q_pre_grasp is None:
                print("预抓取IK求解失败")
                return False
            
            q_pre_grasp_np = q_pre_grasp[0].cpu().numpy()
            
            # 规划到预抓取位置的路径
            success, path1 = self.planner.plan_straight_line(current_q, q_pre_grasp_np)
            if not success:
                print("到预抓取位置的路径规划失败")
                return False
            
            # 2. 执行到预抓取位置
            self.player.play_trajectory(path1)
            
            # 3. 下降到抓取位置
            grasp_pos = obj_pose.p.cpu().numpy() + np.array([0, 0, 0.02])  # 下降到2cm
            grasp_pose = Pose.create_from_pq(
                torch.tensor(grasp_pos, device=self.env.device),
                torch.tensor(pre_grasp_quat, device=self.env.device)
            )
            
            q_grasp = self.env.agent.controller.controllers['arm'].kinematics.compute_ik(
                grasp_pose,
                torch.tensor(q_pre_grasp_np, device=self.env.device)[None]
            )
            
            if q_grasp is None:
                print("抓取IK求解失败")
                return False
            
            q_grasp_np = q_grasp[0].cpu().numpy()
            
            # 规划下降路径
            success, path2 = self.planner.plan_straight_line(q_pre_grasp_np, q_grasp_np, n_steps=20)
            if not success:
                print("下降路径规划失败")
                return False
            
            # 执行下降，最后一步激活吸盘
            suction_commands = [False] * (len(path2) - 1) + [True]
            self.player.play_trajectory(path2, suction_commands)
            
            # 激活吸盘
            if hasattr(self.env.agent, 'activate_suction'):
                self.env.agent.activate_suction(obj_actor)
            
            # 4. 提升物体
            lift_pos = grasp_pos + np.array([0, 0, 0.15])  # 提升15cm
            lift_pose = Pose.create_from_pq(
                torch.tensor(lift_pos, device=self.env.device),
                torch.tensor(pre_grasp_quat, device=self.env.device)
            )
            
            q_lift = self.env.agent.controller.controllers['arm'].kinematics.compute_ik(
                lift_pose,
                torch.tensor(q_grasp_np, device=self.env.device)[None]
            )
            
            if q_lift is None:
                print("提升IK求解失败")
                return False
            
            q_lift_np = q_lift[0].cpu().numpy()
            
            # 规划提升路径
            success, path3 = self.planner.plan_straight_line(q_grasp_np, q_lift_np, n_steps=25)
            if not success:
                print("提升路径规划失败")
                return False
            
            # 执行提升，保持吸盘激活
            suction_commands = [True] * len(path3)
            self.player.play_trajectory(path3, suction_commands)
            
            # 5. 移动到放置位置
            place_pos_with_height = place_pos + np.array([0, 0, 0.15])  # 放置位置上方
            place_pose = Pose.create_from_pq(
                torch.tensor(place_pos_with_height, device=self.env.device),
                torch.tensor(pre_grasp_quat, device=self.env.device)
            )
            
            q_place = self.env.agent.controller.controllers['arm'].kinematics.compute_ik(
                place_pose,
                torch.tensor(q_lift_np, device=self.env.device)[None]
            )
            
            if q_place is None:
                print("放置IK求解失败")
                return False
            
            q_place_np = q_place[0].cpu().numpy()
            
            # 规划到放置位置的路径
            success, path4 = self.planner.plan_straight_line(q_lift_np, q_place_np)
            if not success:
                print("到放置位置的路径规划失败")
                return False
            
            # 执行到放置位置
            suction_commands = [True] * len(path4)
            self.player.play_trajectory(path4, suction_commands)
            
            # 6. 下降并放置
            final_place_pos = place_pos + np.array([0, 0, 0.05])  # 最终放置位置
            final_place_pose = Pose.create_from_pq(
                torch.tensor(final_place_pos, device=self.env.device),
                torch.tensor(pre_grasp_quat, device=self.env.device)
            )
            
            q_final = self.env.agent.controller.controllers['arm'].kinematics.compute_ik(
                final_place_pose,
                torch.tensor(q_place_np, device=self.env.device)[None]
            )
            
            if q_final is None:
                print("最终放置IK求解失败")
                return False
            
            q_final_np = q_final[0].cpu().numpy()
            
            # 规划下降路径
            success, path5 = self.planner.plan_straight_line(q_place_np, q_final_np, n_steps=20)
            if not success:
                print("下降到放置位置的路径规划失败")
                return False
            
            # 执行下降，最后一步关闭吸盘
            suction_commands = [True] * (len(path5) - 1) + [False]
            self.player.play_trajectory(path5, suction_commands)
            
            # 关闭吸盘
            if hasattr(self.env.agent, 'deactivate_suction'):
                self.env.agent.deactivate_suction()
            
            print("抓取和放置成功完成")
            return True
            
        except Exception as e:
            print(f"抓取规划执行失败: {e}")
            return False


def interpolate_joint_trajectory(q_start: np.ndarray, q_end: np.ndarray, 
                               n_steps: int = 50) -> List[np.ndarray]:
    """
    在关节空间插值生成轨迹
    
    Args:
        q_start: 起始关节角度
        q_end: 结束关节角度
        n_steps: 插值步数
        
    Returns:
        关节角度轨迹列表
    """
    trajectory = []
    for i in range(n_steps + 1):
        alpha = i / n_steps
        q_interp = q_start + alpha * (q_end - q_start)
        trajectory.append(q_interp)
    
    return trajectory 