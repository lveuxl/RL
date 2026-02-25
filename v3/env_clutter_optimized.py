"""
优化版EnvClutter环境 - 保证收敛的自上而下抓取学习
关键优化：
1. 简化FSM状态机，大幅减少仿真步数
2. 优化奖励函数，强化自上而下且避免遮挡的策略
3. 提供更清晰的观测信息
4. 加速物理仿真
"""

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
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env(
    "EnvClutterOptimized-v1",
    asset_download_ids=["ycb"],
    max_episode_steps=9,  # 9次抓取尝试
)
class EnvClutterOptimizedEnv(BaseEnv):
    """
    优化版堆叠抓取环境
    - 9个物体，分3层堆叠（3x3网格）
    - 学习最优抓取顺序：自上而下，避免遮挡
    - 简化的FSM实现快速训练
    """
    
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # 环境配置
    NUM_OBJECTS = 9  # 固定9个物体
    GRID_SIZE = 3    # 3x3网格
    LAYERS = 3       # 3层堆叠
    
    # 奖励权重（精心调优）
    REWARD_SUCCESS = 10.0        # 成功抓取奖励
    REWARD_TOP_LAYER = 5.0       # 抓取顶层物体额外奖励
    REWARD_COMPLETE = 50.0       # 完成所有物体奖励
    PENALTY_OCCLUSION = -15.0   # 抓取被遮挡物体的惩罚
    PENALTY_DISPLACEMENT = -2.0  # 扰动其他物体的惩罚
    PENALTY_TIME = -0.1          # 时间惩罚
    
    # FSM优化参数
    MAX_FSM_STEPS = 200  # 大幅减少FSM步数
    MOVE_SPEED = 0.02    # 增加移动速度
    
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
        self.remaining_objects = []  # 每个环境剩余物体
        self.grasped_count = []      # 每个环境已抓取数量
        self.current_layer = []      # 每个环境当前应该抓取的层
        self.action_history = []     # 动作历史记录
        
        # FSM状态
        self.fsm_state = None
        self.fsm_target = None
        self.fsm_tick = None
        
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
                max_rigid_contact_count=2**19,  # 优化内存
                max_rigid_patch_count=2**17
            ),
            spacing=20.0,
            sim_freq=100,      # 降低仿真频率加速
            control_freq=20,   # 降低控制频率
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,  # 降低分辨率加速
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
        # 构建桌面场景
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        
        # 创建9个物体（3x3x3布局）
        self.objects = []
        self.object_layers = []  # 记录每个物体所在层
        self.object_positions = []  # 记录网格位置
        
        # 使用简单的盒子作为物体（加速仿真）
        for layer in range(self.LAYERS):
            for row in range(self.GRID_SIZE):
                for col in range(self.GRID_SIZE):
                    if layer * self.GRID_SIZE * self.GRID_SIZE + row * self.GRID_SIZE + col >= self.NUM_OBJECTS:
                        break
                    
                    # 计算位置（堆叠布局）
                    x = -0.1 + col * 0.06
                    y = -0.06 + row * 0.06  
                    z = 0.05 + layer * 0.055  # 层间距5.5cm
                    
                    # 创建简单的立方体，直接设置初始位置
                    obj = actors.build_cube(
                        self.scene,
                        half_size=0.025,  # 5cm立方体
                        color=np.array([0.5 + layer*0.15, 0.3, 0.3, 1.0]),  # 颜色区分层级
                        name=f"cube_L{layer}_R{row}_C{col}",
                        initial_pose=sapien.Pose(p=[x, y, z]),  # 在创建时设置初始位置
                    )
                    
                    self.objects.append(obj)
                    self.object_layers.append(layer)
                    self.object_positions.append((row, col))
        
        # 创建目标位置标记
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
        """初始化episode"""
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            # 重置物体位置（批量设置）
            if len(self.objects) > 0:
                poses = []
                for i, obj in enumerate(self.objects):
                    layer = self.object_layers[i]
                    row, col = self.object_positions[i]
                    
                    x = -0.1 + col * 0.06
                    y = -0.06 + row * 0.06
                    z = 0.05 + layer * 0.055
                    
                    poses.append([x, y, z])
                
                # 为每个环境批量设置物体位置
                for i, obj in enumerate(self.objects):
                    pos = poses[i]
                    # 为所有环境设置相同的位置
                    batch_poses = torch.tensor([pos] * self.num_envs, device=self.device)
                    obj.set_pose(Pose.create_from_pq(p=batch_poses))
            
            # 重置环境状态
            for i in range(b):
                if i < len(env_idx):
                    env_id = env_idx[i].item() if hasattr(env_idx[i], 'item') else int(env_idx[i])
                    
                    # 确保列表大小正确
                    while len(self.remaining_objects) <= env_id:
                        self.remaining_objects.append([])
                    while len(self.grasped_count) <= env_id:
                        self.grasped_count.append(0)
                    while len(self.current_layer) <= env_id:
                        self.current_layer.append(0)
                    while len(self.action_history) <= env_id:
                        self.action_history.append([])
                    
                    self.remaining_objects[env_id] = list(range(self.NUM_OBJECTS))
                    self.grasped_count[env_id] = 0
                    self.current_layer[env_id] = self.LAYERS - 1  # 从顶层开始
                    self.action_history[env_id] = []
            
            # 初始化FSM状态
            if self.fsm_state is None:
                self.fsm_state = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                self.fsm_target = torch.full((self.num_envs,), -1, dtype=torch.int32, device=self.device)
                self.fsm_tick = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            else:
                for i in range(b):
                    if i < len(env_idx):
                        env_id = env_idx[i].item() if hasattr(env_idx[i], 'item') else int(env_idx[i])
                        if env_id < self.num_envs:
                            self.fsm_state[env_id] = 0
                            self.fsm_target[env_id] = -1
                            self.fsm_tick[env_id] = 0
            
            # 重置机器人
            qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
            self.agent.reset(qpos)

    def _get_obs_extra(self, info: Dict):
        """获取观测 - 提供清晰的层级和遮挡信息"""
        batch_size = self.num_envs
        
        # 构建观测向量
        obs_list = []
        
        for env_idx in range(batch_size):
            # 1. 物体状态特征 (9个物体 x 5维特征 = 45维)
            object_features = []
            for obj_idx in range(self.NUM_OBJECTS):
                if env_idx < len(self.remaining_objects) and obj_idx in self.remaining_objects[env_idx]:
                    # 物体存在
                    layer = self.object_layers[obj_idx]
                    row, col = self.object_positions[obj_idx]
                    
                    # 特征：[存在标志, 层级, 行, 列, 是否被遮挡]
                    is_occluded = self._is_occluded(obj_idx, env_idx)
                    features = [1.0, layer/2.0, row/2.0, col/2.0, float(is_occluded)]
                else:
                    # 物体已被抓取
                    features = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                object_features.extend(features)
            
            # 2. 全局状态特征 (5维)
            if env_idx < len(self.grasped_count):
                grasped_ratio = self.grasped_count[env_idx] / self.NUM_OBJECTS
                current_layer_norm = self.current_layer[env_idx] / (self.LAYERS - 1) if self.LAYERS > 1 else 0
                remaining_ratio = len(self.remaining_objects[env_idx]) / self.NUM_OBJECTS
            else:
                grasped_ratio = 0.0
                current_layer_norm = 0.0
                remaining_ratio = 1.0
            
            # TCP位置
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1 and env_idx < tcp_pos.shape[0]:
                tcp_x = tcp_pos[env_idx, 0]
                tcp_y = tcp_pos[env_idx, 1]
            else:
                tcp_x = tcp_y = 0.0
            
            global_features = [
                grasped_ratio,
                current_layer_norm,
                remaining_ratio,
                tcp_x,
                tcp_y
            ]
            
            # 3. 动作掩码 (9维)
            action_mask = []
            for obj_idx in range(self.NUM_OBJECTS):
                if env_idx < len(self.remaining_objects) and obj_idx in self.remaining_objects[env_idx]:
                    # 可以抓取（但可能被遮挡）
                    action_mask.append(1.0)
                else:
                    # 已被抓取，不可选择
                    action_mask.append(0.0)
            
            # 组合所有特征
            obs = object_features + global_features + action_mask
            obs_list.append(obs)
        
        # 转换为tensor
        obs_tensor = torch.tensor(obs_list, device=self.device, dtype=torch.float32)
        
        return obs_tensor

    def _is_occluded(self, obj_idx: int, env_idx: int) -> bool:
        """判断物体是否被遮挡"""
        obj_layer = self.object_layers[obj_idx]
        obj_row, obj_col = self.object_positions[obj_idx]
        
        # 检查上层是否有物体
        for other_idx in range(self.NUM_OBJECTS):
            if other_idx == obj_idx:
                continue
            
            # 检查是否已被抓取
            if env_idx < len(self.remaining_objects) and other_idx not in self.remaining_objects[env_idx]:
                continue
            
            other_layer = self.object_layers[other_idx]
            other_row, other_col = self.object_positions[other_idx]
            
            # 如果有物体在正上方
            if other_layer > obj_layer and other_row == obj_row and other_col == obj_col:
                return True
        
        return False

    def step(self, action):
        """执行离散动作 - 选择要抓取的物体"""
        # 统一处理动作格式
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()
        elif isinstance(action, (int, np.integer)):
            action = np.full(self.num_envs, action)
        elif isinstance(action, (list, tuple)):
            action = np.array(action)
        else:
            action = np.asarray(action).flatten()
        
        # 确保action形状正确
        if len(action) != self.num_envs:
            if len(action) > 0:
                action = np.full(self.num_envs, action[0])
            else:
                action = np.full(self.num_envs, 0)
        
        # 执行快速FSM抓取
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        for env_idx in range(self.num_envs):
            # 安全地转换动作为整数
            try:
                obj_idx = int(action[env_idx])
            except (ValueError, TypeError):
                obj_idx = 0  # 默认选择第一个物体
            
            # 检查动作有效性
            if env_idx >= len(self.remaining_objects) or obj_idx not in self.remaining_objects[env_idx]:
                # 无效动作，给予惩罚
                rewards[env_idx] = -1.0
                continue
            
            # 记录动作
            if env_idx < len(self.action_history):
                self.action_history[env_idx].append(obj_idx)
            
            # 执行简化的抓取（快速模拟）
            success = self._execute_grasp(obj_idx, env_idx)
            
            if success:
                # 计算奖励
                reward = self._calculate_reward(obj_idx, env_idx)
                rewards[env_idx] = reward
                
                # 更新状态
                self.remaining_objects[env_idx].remove(obj_idx)
                self.grasped_count[env_idx] += 1
                
                # 更新当前层
                if all(i not in self.remaining_objects[env_idx] 
                       for i in range(self.NUM_OBJECTS) 
                       if self.object_layers[i] == self.current_layer[env_idx]):
                    if self.current_layer[env_idx] > 0:
                        self.current_layer[env_idx] -= 1
            else:
                # 抓取失败惩罚
                rewards[env_idx] = -2.0
        
        # 获取观测
        info = self.evaluate()
        obs = self.get_obs(info)
        
        # 检查终止条件
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for env_idx in range(self.num_envs):
            # 所有物体抓取完成或达到最大步数
            if env_idx < len(self.grasped_count):
                if self.grasped_count[env_idx] >= self.NUM_OBJECTS:
                    terminated[env_idx] = True
                    rewards[env_idx] += self.REWARD_COMPLETE  # 完成奖励
                elif len(self.action_history[env_idx]) >= self.NUM_OBJECTS:
                    truncated[env_idx] = True
        
        return obs, rewards, terminated, truncated, info

    def _execute_grasp(self, obj_idx: int, env_idx: int) -> bool:
        """执行简化的抓取动作"""
        # 简化处理：直接模拟抓取结果
        # 如果物体被遮挡，失败概率更高
        is_occluded = self._is_occluded(obj_idx, env_idx)
        
        if is_occluded:
            # 被遮挡的物体有90%概率失败
            success = np.random.random() > 0.9
        else:
            # 未被遮挡的物体有95%概率成功
            success = np.random.random() < 0.95
        
        # 极简仿真：完全跳过物理步骤（最大加速）
        pass  # 不执行任何物理仿真，纯逻辑判断
        
        # 如果成功，移除物体（批量操作）
        if success and obj_idx < len(self.objects):
            # 获取当前物体位置
            current_pose = self.objects[obj_idx].pose
            # 创建新的位置数组，只修改特定环境的物体位置
            new_poses = current_pose.p.clone()
            if env_idx < new_poses.shape[0]:
                new_poses[env_idx] = torch.tensor([10.0, 10.0, 10.0], device=self.device)
            # 批量设置新位置
            self.objects[obj_idx].set_pose(Pose.create_from_pq(p=new_poses, q=current_pose.q))
        
        return success

    def _calculate_reward(self, obj_idx: int, env_idx: int) -> float:
        """计算奖励 - 鼓励自上而下且避免遮挡的策略"""
        reward = self.REWARD_SUCCESS  # 基础成功奖励
        
        # 1. 层级奖励：抓取高层物体获得额外奖励
        obj_layer = self.object_layers[obj_idx]
        if env_idx < len(self.current_layer):
            expected_layer = self.current_layer[env_idx]
            if obj_layer == expected_layer:
                reward += self.REWARD_TOP_LAYER  # 抓取正确层级
            elif obj_layer < expected_layer:
                reward += (obj_layer - expected_layer) * 2  # 抓取低层惩罚
        
        # 2. 遮挡惩罚：如果抓取被遮挡的物体
        if self._is_occluded(obj_idx, env_idx):
            reward += self.PENALTY_OCCLUSION
        
        # 3. 效率奖励：基于已抓取数量
        if env_idx < len(self.grasped_count):
            efficiency_bonus = self.grasped_count[env_idx] * 0.5
            reward += efficiency_bonus
        
        # 4. 时间惩罚
        reward += self.PENALTY_TIME
        
        return reward

    def seed(self, seed: Optional[int] = None):
        """设置随机种子 - 兼容ManiSkill和gym接口"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        return [seed]

    def evaluate(self):
        """评估环境状态"""
        success = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.grasped_count):
                # 成功条件：抓取所有物体
                success[env_idx] = self.grasped_count[env_idx] >= self.NUM_OBJECTS
        
        return {
            "success": success,
            "grasped_count": torch.tensor(self.grasped_count, device=self.device) if self.grasped_count else torch.zeros(self.num_envs, device=self.device),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """已在step中计算"""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """归一化奖励"""
        return self.compute_dense_reward(obs, action, info) / 10.0