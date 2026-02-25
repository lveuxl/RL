"""
掩码包装器 - 将EnvClutter的离散观测拆分成state和action_mask
用于与stable-baselines3的MaskablePPO配合使用
"""

import numpy as np
import gymnasium as gym
import torch
from typing import Dict, Any


class ActionConversionWrapper(gym.ActionWrapper):
    """
    动作转换包装器 - 将连续动作转换为离散动作
    用于让SB3的连续动作PPO与离散动作环境兼容
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # 获取原始离散动作空间 - 使用推荐的方式
        if hasattr(env.unwrapped, 'discrete_action_space') and env.unwrapped.discrete_action_space is not None:
            self.discrete_action_space = env.unwrapped.discrete_action_space
            n_discrete_actions = env.unwrapped.discrete_action_space.n
        else:
            # 默认值
            n_discrete_actions = 15
            self.discrete_action_space = gym.spaces.Discrete(n_discrete_actions)
        
        # 将动作空间设为连续空间，让SB3的PPO可以处理
        # 输出一个n维向量，通过argmax转换为离散动作
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(n_discrete_actions,), 
            dtype=np.float32
        )
    
    def action(self, action):
        """将连续动作转换为离散动作"""
        if isinstance(action, np.ndarray):
            # 使用argmax选择最大值对应的动作
            discrete_action = int(np.argmax(action))
        else:
            # 如果是标量，直接转换
            discrete_action = int(action)
        
        # 确保动作在有效范围内
        discrete_action = max(0, min(discrete_action, self.discrete_action_space.n - 1))
        
        return discrete_action


class ExtractMaskWrapper(gym.ObservationWrapper):
    """
    把 EnvClutter 的离散观测拆成包含掩码信息的tensor格式
    
    与ManiSkillSB3VectorEnv兼容，返回torch.Tensor格式的观测
    观测结构：[action_mask(15), state_features(...)]
    """
    
    def __init__(self, env, max_n=15):
        super().__init__(env)
        self.max_n = max_n
        
        # 获取原始观测空间
        original_obs_space = env.observation_space
        
        # 保持原始的Box观测空间格式，但调整维度以包含掩码
        if isinstance(original_obs_space, gym.spaces.Box):
            original_shape = original_obs_space.shape
            if len(original_shape) == 2:
                # 形状为(batch_size, feature_dim)
                batch_size, feature_dim = original_shape
                # 新的特征维度 = 掩码维度 + 原始特征维度
                new_feature_dim = max_n + feature_dim
                new_shape = (batch_size, new_feature_dim)
            else:
                # 1D情况，直接添加掩码维度
                new_shape = (original_shape[0] + max_n,)
            
            # 创建新的Box观测空间
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=new_shape,
                dtype=np.float32
            )
        else:
            # 默认情况
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(max_n + 121,), 
                dtype=np.float32
            )
    
    def observation(self, obs):
        """转换观测，保持torch.Tensor格式"""
        import torch
        
        # 确保观测是torch.Tensor格式
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        # 处理观测维度
        if obs.dim() == 2:
            # 形状为(batch_size, feature_dim)
            batch_size, feature_dim = obs.shape
            
            # 提取掩码（前MAX_N个元素）
            mask = obs[:, :self.max_n]  # [batch_size, max_n]
            
            # 提取状态特征（剩余元素）
            state_features = obs[:, self.max_n:]  # [batch_size, remaining_features]
            
            # 重新组合：[mask, state_features]
            combined_obs = torch.cat([mask, state_features], dim=1)
            
            return combined_obs
        
        elif obs.dim() == 1:
            # 1D情况
            mask = obs[:self.max_n]
            state_features = obs[self.max_n:]
            combined_obs = torch.cat([mask, state_features])
            return combined_obs
        
        else:
            # 其他情况，直接返回
            return obs


class SB3CompatWrapper(gym.Wrapper):
    """
    确保环境与SB3完全兼容的包装器
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # 确保动作空间是gymnasium格式 - 使用推荐的方式获取属性
        # 优先使用discrete_action_space（用于MaskablePPO）
        if hasattr(env.unwrapped, 'discrete_action_space') and env.unwrapped.discrete_action_space is not None:
            self.action_space = env.unwrapped.discrete_action_space
        else:
            self.action_space = env.action_space
    
    def step(self, action):
        """确保step返回的都是numpy格式"""
        # 处理动作转换 - 将连续动作转换为离散动作索引
        if hasattr(self.env.unwrapped, 'use_discrete_action') and self.env.unwrapped.use_discrete_action:
            # 如果是离散动作环境，需要转换动作
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    # 单个动作
                    action = int(action.item())
                else:
                    # 多个动作，取第一个或使用argmax
                    if len(action) == self.action_space.n:
                        # 如果动作维度等于动作空间大小，使用argmax
                        action = int(np.argmax(action))
                    else:
                        # 否则取第一个元素
                        action = int(action[0])
            elif isinstance(action, (list, tuple)):
                action = int(action[0])
            elif not isinstance(action, (int, np.integer)):
                action = int(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 保持原始tensor格式，让ManiSkillSB3VectorEnv处理
        # 确保info中的success是tensor格式，且维度匹配环境数量
        if isinstance(info, dict) and 'success' in info:
            if isinstance(info['success'], bool):
                # 将bool转换为tensor格式，维度匹配环境数量
                import torch
                # 获取环境数量 - 使用推荐的方式
                num_envs = getattr(self.env.unwrapped, 'num_envs', 1)
                
                # 创建匹配环境数量的tensor - 使用推荐的方式获取device
                device = getattr(self.env.unwrapped, 'device', 'cpu')
                
                info['success'] = torch.tensor([info['success']] * num_envs, device=device)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """确保reset返回的观测是numpy格式"""
        # 确保options参数不为None，避免RecordEpisode包装器报错
        if 'options' not in kwargs or kwargs['options'] is None:
            kwargs['options'] = {}
        
        return self.env.reset(**kwargs) 