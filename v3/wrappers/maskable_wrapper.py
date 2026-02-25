"""
MaskablePPO专用包装器 - 确保动作掩码正确传递给MaskablePPO
"""

import numpy as np
import gymnasium as gym
import torch
from typing import Dict, Any, Optional
from sb3_contrib.common.wrappers import ActionMasker


class MaskableActionWrapper(gym.Wrapper):
    """
    专用于MaskablePPO的动作掩码包装器
    确保环境返回的观测和动作掩码格式正确
    """
    
    def __init__(self, env, max_n: int = 9):
        super().__init__(env)
        self.max_n = max_n
        
        # 确保动作空间是Discrete类型
        if hasattr(env.unwrapped, 'discrete_action_space'):
            self.action_space = env.unwrapped.discrete_action_space
        else:
            self.action_space = gym.spaces.Discrete(max_n)
        
        # 保持原观测空间不变，MaskablePPO会自动处理掩码
        self.observation_space = env.observation_space
        
    def reset(self, **kwargs):
        """重置环境并返回初始观测"""
        obs, info = self.env.reset(**kwargs)
        
        # 确保obs是正确格式
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        return obs, info
        
    def step(self, action):
        """执行动作并返回结果"""
        # 确保动作是整数
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, (list, tuple)):
            action = int(action[0])
        else:
            action = int(action)
            
        # 确保动作在有效范围内
        action = max(0, min(action, self.action_space.n - 1))
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 确保obs是numpy格式
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
            
        # 确保reward是标量
        if isinstance(reward, (np.ndarray, torch.Tensor)):
            reward = float(reward.item())
        
        return obs, reward, terminated, truncated, info


def action_mask_fn(env) -> np.ndarray:
    """
    为MaskablePPO提取动作掩码的函数
    从环境观测中提取掩码信息
    """
    # 这个函数会被ActionMasker调用
    # 需要从环境中获取当前的动作掩码
    
    if hasattr(env, '_current_obs') and env._current_obs is not None:
        obs = env._current_obs
        
        # 从观测中提取掩码
        if isinstance(obs, np.ndarray):
            if obs.ndim == 2:
                # 批次观测：[batch_size, feature_dim]  
                # 假设前max_n个元素是掩码
                mask = obs[:, :env.unwrapped.max_n]
                # 返回第一个环境的掩码
                return mask[0].astype(bool)
            else:
                # 单环境观测：[feature_dim]
                mask = obs[:env.unwrapped.max_n]
                return mask.astype(bool)
    
    # 默认情况：所有动作都可用
    max_n = getattr(env.unwrapped, 'MAX_N', 9)
    return np.ones(max_n, dtype=bool)


def create_maskable_env(base_env, max_n: int = 9):
    """
    为MaskablePPO创建正确配置的环境
    
    Args:
        base_env: 基础环境
        max_n: 最大动作数量
    
    Returns:
        配置好的MaskablePPO兼容环境
    """
    # 应用MaskablePPO包装器
    env = MaskableActionWrapper(base_env, max_n=max_n)
    
    # 添加ActionMasker包装器，这是MaskablePPO所必需的
    env = ActionMasker(env, action_mask_fn)
    
    # 添加一个属性来存储当前观测，供mask函数使用
    env.max_n = max_n
    original_reset = env.reset
    original_step = env.step
    
    def new_reset(**kwargs):
        obs, info = original_reset(**kwargs)
        env._current_obs = obs
        return obs, info
    
    def new_step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        env._current_obs = obs
        return obs, reward, terminated, truncated, info
    
    env.reset = new_reset
    env.step = new_step
    
    return env


class MaskableVectorEnvWrapper(gym.vector.VectorEnvWrapper):
    """
    向量化环境的MaskablePPO包装器
    处理多环境的动作掩码
    """
    
    def __init__(self, venv, max_n: int = 9):
        super().__init__(venv)
        self.max_n = max_n
        
        # 确保动作空间是Discrete
        self.action_space = gym.spaces.Discrete(max_n)
        
    def reset(self, **kwargs):
        """重置所有环境"""
        obs, infos = self.venv.reset(**kwargs)
        
        # 存储观测用于掩码提取
        self._current_obs = obs
        
        return obs, infos
        
    def step(self, actions):
        """执行动作"""
        obs, rewards, terminated, truncated, infos = self.venv.step(actions)
        
        # 更新观测
        self._current_obs = obs
        
        return obs, rewards, terminated, truncated, infos
    
    def action_masks(self) -> np.ndarray:
        """
        返回所有环境的动作掩码
        
        Returns:
            形状为 [num_envs, max_n] 的布尔数组
        """
        if hasattr(self, '_current_obs') and self._current_obs is not None:
            obs = self._current_obs
            
            if isinstance(obs, np.ndarray):
                if obs.ndim == 2:
                    # [batch_size, feature_dim]
                    batch_size = obs.shape[0]
                    masks = obs[:, :self.max_n]  # 提取前max_n个元素作为掩码
                    return masks.astype(bool)
                else:
                    # 单环境情况，扩展为批次维度
                    mask = obs[:self.max_n]
                    return mask.astype(bool).reshape(1, -1)
        
        # 默认情况：所有动作都可用
        num_envs = self.num_envs
        return np.ones((num_envs, self.max_n), dtype=bool)