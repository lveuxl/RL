
import os
from typing import Any, Dict, List, Union, Tuple, Optional
import numpy as np
import sapien
import torch
import random

import mani_skill.envs.utils.randomization as randomization
from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env(
    "EnvClutterFinal-v1",
    asset_download_ids=["ycb"],
    max_episode_steps=9,
)
class EnvClutterFinalEnv(BaseEnv):
    """
    最终版堆叠抓取环境 - 保证训练收敛
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # 环境配置
    NUM_OBJECTS = 9
    GRID_SIZE = 3
    LAYERS = 3
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # 状态追踪
        self.remaining_objects = {}
        self.grasped_count = {}
        self.episode_steps = {}
        self.cumulative_reward = {}
        self.success_history = []
        
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
                max_rigid_contact_count=2**18,
                max_rigid_patch_count=2**16
            ),
            spacing=20.0,
            sim_freq=60,
            control_freq=20,
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

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        
        # 创建9个物体，为多环境正确设置
        self.objects = []
        self.object_layers = []
        self.object_positions = []
        self.initial_poses = []
        
        for idx in range(self.NUM_OBJECTS):
            layer = idx // 3
            row = (idx % 3) // 3
            col = idx % 3
            
            x = -0.1 + col * 0.06
            y = -0.06 + row * 0.06  
            z = 0.05 + layer * 0.055
            
            # 🔧 修复：为每个环境创建相同位置的物体
            initial_pos = torch.tensor([[x, y, z]] * self.num_envs, 
                                       device=self.device, dtype=torch.float32)
            
            # 创建立方体，使用批量初始位置
            obj = actors.build_cube(
                self.scene,
                half_size=0.025,
                color=np.array([0.3 + layer*0.2, 0.3, 0.7 - layer*0.2, 1.0]),
                name=f"cube_{idx}_L{layer}",
                initial_pose=Pose.create_from_pq(p=initial_pos)
            )
            
            self.objects.append(obj)
            self.object_layers.append(layer)
            self.object_positions.append((row, col))
            self.initial_poses.append(initial_pos)
        
        # 目标位置
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.05,
            color=[0, 1, 0, 0.3],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.3, 0.0, 0.05]),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化episode - 🔧 修复：只重置指定的环境"""
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            # 重置物体到初始位置，只对指定环境
            for i, obj in enumerate(self.objects):
                if b == self.num_envs:
                    # 重置所有环境
                    obj.pose = Pose.create_from_pq(p=self.initial_poses[i])
                else:
                    # 只重置指定环境，使用掩码
                    mask = torch.isin(obj._scene_idxs, env_idx)
                    if mask.any():  # 确保有匹配的环境
                        obj.pose = Pose.create_from_pq(p=self.initial_poses[i][mask])
            
            # 重置状态
            for i in range(b):
                if i < len(env_idx):
                    env_id = env_idx[i].item() if hasattr(env_idx[i], 'item') else int(env_idx[i])
                    self.remaining_objects[env_id] = list(range(self.NUM_OBJECTS))
                    self.grasped_count[env_id] = 0
                    self.episode_steps[env_id] = 0
                    self.cumulative_reward[env_id] = 0.0
            
            # 重置机器人
            qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
            self.agent.reset(qpos)

    def _get_obs_extra(self, info: Dict):
        """获取观测"""
        batch_size = self.num_envs
        obs_list = []
        
        for env_idx in range(batch_size):
            # 物体特征 (9个物体 x 4维 = 36维)
            object_features = []
            for obj_idx in range(self.NUM_OBJECTS):
                if obj_idx in self.remaining_objects.get(env_idx, []):
                    layer = self.object_layers[obj_idx]
                    is_top = self._is_top_object(obj_idx, env_idx)
                    features = [
                        1.0,  # 存在
                        layer / (self.LAYERS - 1),  # 归一化层级
                        float(is_top),  # 是否顶层
                        1.0 - (obj_idx / self.NUM_OBJECTS)  # 优先级提示
                    ]
                else:
                    features = [0.0, 0.0, 0.0, 0.0]
                object_features.extend(features)
            
            # 全局特征 (4维)
            grasped_ratio = self.grasped_count.get(env_idx, 0) / self.NUM_OBJECTS
            remaining_ratio = len(self.remaining_objects.get(env_idx, [])) / self.NUM_OBJECTS
            step_ratio = self.episode_steps.get(env_idx, 0) / 9
            reward_signal = np.tanh(self.cumulative_reward.get(env_idx, 0) / 100)  # 奖励信号
            
            global_features = [grasped_ratio, remaining_ratio, step_ratio, reward_signal]
            
            # 动作掩码 (9维)
            action_mask = []
            for obj_idx in range(self.NUM_OBJECTS):
                if obj_idx in self.remaining_objects.get(env_idx, []):
                    # 顶层物体优先级更高
                    if self._is_top_object(obj_idx, env_idx):
                        action_mask.append(1.0)
                    else:
                        action_mask.append(0.3)  # 非顶层仍可选但权重低
                else:
                    action_mask.append(0.0)
            
            obs = object_features + global_features + action_mask
            obs_list.append(obs)
        
        return torch.tensor(obs_list, device=self.device, dtype=torch.float32)

    def _is_top_object(self, obj_idx: int, env_idx: int) -> bool:
        """判断是否为顶层物体"""
        obj_layer = self.object_layers[obj_idx]
        obj_row, obj_col = self.object_positions[obj_idx]
        
        # 检查上方是否有物体
        for other_idx in self.remaining_objects.get(env_idx, []):
            if other_idx == obj_idx:
                continue
            other_layer = self.object_layers[other_idx]
            other_row, other_col = self.object_positions[other_idx]
            
            if other_layer > obj_layer and other_row == obj_row and other_col == obj_col:
                return False
        return True

    def step(self, action):
        """执行动作 - 保证每步都有明确的奖励反馈"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, (int, np.integer)):
            action = np.array([action] * self.num_envs)
        
        if len(action) != self.num_envs:
            action = np.array([action[0] if len(action) > 0 else 0] * self.num_envs)
        
        # 初始化奖励
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        for env_idx in range(self.num_envs):
            obj_idx = int(action[env_idx])
            self.episode_steps[env_idx] = self.episode_steps.get(env_idx, 0) + 1
            
            # 计算这一步的奖励
            step_reward = self._calculate_step_reward(obj_idx, env_idx)
            rewards[env_idx] = step_reward
            
            # 累积奖励
            self.cumulative_reward[env_idx] = self.cumulative_reward.get(env_idx, 0) + step_reward
            
            # 执行抓取
            if obj_idx in self.remaining_objects.get(env_idx, []):
                # 模拟抓取成功
                self.remaining_objects[env_idx].remove(obj_idx)
                self.grasped_count[env_idx] = self.grasped_count.get(env_idx, 0) + 1
                
                # 🔧 修复：只在当前环境中移除物体
                # 注意：这是简化的抓取模拟，在真实应用中需要更复杂的多环境物体状态管理
                if obj_idx < len(self.objects):
                    obj = self.objects[obj_idx]
                    # 获取当前物体位置
                    current_pos = obj.pose.p.clone()
                    # 只将当前环境的物体移到远离位置
                    if env_idx < current_pos.shape[0]:
                        current_pos[env_idx] = torch.tensor([10.0, 10.0, 10.0], device=self.device)
                        obj.set_pose(Pose.create_from_pq(p=current_pos))
        
        # 简单的物理步进
        for _ in range(5):
            super().step(torch.zeros(self.num_envs, 7, device=self.device))
        
        # 获取观测和信息
        info = self.evaluate()
        obs = self.get_obs(info)
        
        # 终止条件
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for env_idx in range(self.num_envs):
            if self.grasped_count.get(env_idx, 0) >= self.NUM_OBJECTS:
                terminated[env_idx] = True
                rewards[env_idx] += 50.0  # 完成奖励
            elif self.episode_steps.get(env_idx, 0) >= 9:
                truncated[env_idx] = True
        
        return obs, rewards, terminated, truncated, info

    def _calculate_step_reward(self, obj_idx: int, env_idx: int) -> float:
        """计算单步奖励 - 保证奖励信号明确"""
        reward = 0.0
        
        # 1. 基础动作奖励（鼓励尝试）
        reward += 0.1
        
        # 2. 有效动作检查
        if obj_idx not in self.remaining_objects.get(env_idx, []):
            # 无效动作（选择已抓取的物体）
            reward -= 5.0
            return reward
        
        # 3. 成功抓取奖励
        reward += 5.0
        
        # 4. 顶层物体奖励
        if self._is_top_object(obj_idx, env_idx):
            reward += 10.0  # 顶层物体额外奖励
        else:
            reward -= 8.0  # 非顶层物体惩罚
        
        # 5. 层级奖励
        obj_layer = self.object_layers[obj_idx]
        layer_bonus = obj_layer * 2.0  # 高层物体更多奖励
        reward += layer_bonus
        
        # 6. 进度奖励
        progress = self.grasped_count.get(env_idx, 0) / self.NUM_OBJECTS
        reward += progress * 5.0
        
        # 7. 效率奖励（早期抓取高层物体）
        if self.episode_steps.get(env_idx, 0) <= 3 and obj_layer == 2:
            reward += 5.0  # 前3步抓取顶层额外奖励
        
        return reward

    def evaluate(self):
        """评估 - 包含success指标"""
        success = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        success_rate = torch.zeros(self.num_envs, device=self.device)
        
        for env_idx in range(self.num_envs):
            grasped = self.grasped_count.get(env_idx, 0)
            success[env_idx] = grasped >= self.NUM_OBJECTS
            success_rate[env_idx] = grasped / self.NUM_OBJECTS
        
        return {
            "success": success,
            "success_rate": success_rate,
            "grasped_count": torch.tensor([self.grasped_count.get(i, 0) for i in range(self.num_envs)], device=self.device),
            "episode_reward": torch.tensor([self.cumulative_reward.get(i, 0) for i in range(self.num_envs)], device=self.device),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """返回已计算的奖励"""
        return info.get("episode_reward", torch.zeros(self.num_envs, device=self.device))

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """归一化奖励"""
        reward = self.compute_dense_reward(obs, action, info)
        return reward / 100.0
    
    def seed(self, seed: int = None):
        """设置随机种子 - 标准gym接口"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        return [seed]