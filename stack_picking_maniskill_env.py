import gymnasium as gym
import numpy as np
import random
import torch
from typing import Any, Dict, Union

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.sensors.camera import CameraConfig
from sapien import Pose as SapienPose

@register_env("StackPickingManiSkill-v1", max_episode_steps=200)
class StackPickingManiSkillEnv(BaseEnv):
    """
    简化版本的堆叠抓取环境，基于PickCube结构
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    def __init__(self, *args, robot_uids="panda", max_objects=3, **kwargs):
        self.max_objects = max_objects
        self.robot_uids = robot_uids
        
        # 环境状态
        self.step_counter = 0
        self.success_count = 0
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        """传感器配置 - 添加基础相机配置"""
        return []

    @property
    def _default_human_render_camera_configs(self):
        """人类渲染相机配置 - 桌子高度正面视角"""
        return [
            CameraConfig(
                uid="render_camera",
                # 相机位置：在机器人正前方，桌子高度，水平看向机器人
                # 修正四元数，确保相机是正立的
                pose=Pose.create_from_pq([1.5, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]),  # 正立的正面视角
                width=1024,
                height=768,
                fov=1.0,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_sim_config(self):
        """仿真配置"""
        return SimConfig(
            sim_freq=500,
            control_freq=20,
        )

    def _load_agent(self, options: dict):
        # 设置机器人的初始位置，参考PickCube示例
        # 这样可以避免"No initial pose set"的警告
        import sapien
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # 构建桌面场景
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.02)
        self.table_scene.build()
        
        # 创建简单的立方体物体
        self.cubes = []
        for i in range(self.max_objects):
            # 简单地放在原点附近，让物理引擎处理掉落
            initial_pos = [0.1 + i * 0.05, 0.0, 1.0]  # 从高处掉落到桌面
            initial_quat = [1.0, 0.0, 0.0, 0.0]
            
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=[1.0, 0.0, 0.0, 1.0],  # 红色立方体
                name=f"cube_{i}",
                body_type="dynamic",
                add_collision=True,
                initial_pose=SapienPose(initial_pos, initial_quat)
            )
            self.cubes.append(cube)
        
        # 创建目标位置标记
        goal_initial_pos = [0.3, 0.3, 1.0]  # 目标位置
        goal_initial_quat = [1.0, 0.0, 0.0, 0.0]
        
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[0.0, 1.0, 0.0, 0.8],  # 绿色半透明球体
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=SapienPose(goal_initial_pos, goal_initial_quat)
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每个episode"""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # 重置环境状态
            self.step_counter = 0
            self.success_count = 0
            
            # 简单放置立方体，让它们掉落到桌面
            for i, cube in enumerate(self.cubes):
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = 0.1 + i * 0.05  # 沿x轴排列
                xyz[:, 1] = torch.rand((b,)) * 0.1 - 0.05  # y轴小幅随机
                xyz[:, 2] = 1.0  # 从高处掉落
                
                qs = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)
                cube.set_pose(Pose.create_from_pq(xyz, qs))
            
            # 设置目标位置
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = 0.3
            goal_xyz[:, 1] = 0.3
            goal_xyz[:, 2] = 1.0
            goal_qs = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz, goal_qs))

    def _get_obs_extra(self, info: Dict):
        """获取额外观测信息 - 参考PickCube的格式"""
        # 选择第一个立方体作为目标
        target_cube = self.cubes[0] if self.cubes else None
        
        # 获取批量大小
        batch_size = self.num_envs
        
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            # 修复：success_count应该匹配批量大小
            success_count=torch.full((batch_size,), self.success_count, dtype=torch.float32, device=self.device),
        )
        
        if "state" in self.obs_mode and target_cube is not None:
            obs.update(
                obj_pose=target_cube.pose.raw_pose,
                tcp_to_obj_pos=target_cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - target_cube.pose.p,
            )
        
        return obs

    def evaluate(self):
        """评估函数"""
        # 简单的成功条件：第一个立方体接近目标位置
        if self.cubes:
            target_cube = self.cubes[0]
            is_obj_placed = (
                torch.linalg.norm(self.goal_site.pose.p - target_cube.pose.p, axis=1)
                <= 0.05
            )
            is_robot_static = self.agent.is_static(0.2)
            success = is_obj_placed & is_robot_static
        else:
            success = torch.tensor([True], device=self.device)
            is_obj_placed = torch.tensor([True], device=self.device)
            is_robot_static = torch.tensor([True], device=self.device)
        
        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "success_count": self.success_count,
            "remaining_objects": len(self.cubes),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """稠密奖励函数 - 参考PickCube的奖励结构"""
        if not self.cubes:
            return torch.tensor(0.0, device=self.device)
        
        target_cube = self.cubes[0]
        
        # 接近奖励
        tcp_to_obj_dist = torch.linalg.norm(
            target_cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward
        
        # 放置奖励
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - target_cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward
        
        # 成功奖励
        if info.get("success", torch.tensor([False], device=self.device)).any():
            reward += 5
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 7 