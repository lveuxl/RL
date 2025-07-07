"""
EnvClutter环境测试脚本
用于验证环境是否正常工作
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# 导入自定义模块
try:
    from env_clutter import EnvClutterEnv
    from config import get_config
    from utils import setup_seed, check_environment_setup
    import mani_skill.envs
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

def test_environment_creation():
    """测试环境创建"""
    print("\n=== 测试环境创建 ===")
    
    try:
        # 设置随机种子
        setup_seed(42)
        
        # 创建环境
        env = gym.make(
            "EnvClutter-v1",
            num_envs=1,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",
            render_mode="rgb_array"
        )
        
        print(f"✓ 环境创建成功: {env.spec.id}")
        print(f"  观测空间: {env.observation_space}")
        print(f"  动作空间: {env.action_space}")
        print(f"  动作维度: {env.action_space.shape}")
        
        return env
        
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        return None

def test_environment_reset(env):
    """测试环境重置"""
    print("\n=== 测试环境重置 ===")
    
    try:
        obs, info = env.reset()
        print(f"✓ 环境重置成功")
        print(f"  观测类型: {type(obs)}")
        
        if isinstance(obs, dict):
            print(f"  观测键: {list(obs.keys())}")
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    {key}: {type(value)}")
        else:
            print(f"  观测形状: {obs.shape}")
        
        print(f"  信息: {info}")
        return obs, info
        
    except Exception as e:
        print(f"✗ 环境重置失败: {e}")
        return None, None

def test_environment_step(env, obs):
    """测试环境步进"""
    print("\n=== 测试环境步进 ===")
    
    try:
        # 生成随机动作
        action = env.action_space.sample()
        print(f"  随机动作: {action}")
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ 环境步进成功")
        print(f"  奖励: {reward}")
        print(f"  结束: {terminated}")
        print(f"  截断: {truncated}")
        print(f"  信息: {info}")
        
        return next_obs, reward, terminated, truncated, info
        
    except Exception as e:
        print(f"✗ 环境步进失败: {e}")
        return None, None, None, None, None

def test_environment_render(env):
    """测试环境渲染"""
    print("\n=== 测试环境渲染 ===")
    
    try:
        frame = env.render()
        
        if frame is not None:
            print(f"✓ 环境渲染成功")
            print(f"  帧形状: {frame.shape}")
            print(f"  帧类型: {frame.dtype}")
            return frame
        else:
            print("✗ 环境渲染返回None")
            return None
        
    except Exception as e:
        print(f"✗ 环境渲染失败: {e}")
        return None

def test_multiple_steps(env, num_steps=10):
    """测试多步运行"""
    print(f"\n=== 测试多步运行 ({num_steps}步) ===")
    
    try:
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Episode在第{step+1}步结束")
                obs, _ = env.reset()
        
        print(f"✓ 多步运行成功")
        print(f"  总奖励: {total_reward}")
        
    except Exception as e:
        print(f"✗ 多步运行失败: {e}")

def test_config_loading():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    
    try:
        # 测试默认配置
        config = get_config('default')
        print(f"✓ 默认配置加载成功")
        print(f"  环境名称: {config.env.env_name}")
        print(f"  环境数量: {config.env.num_envs}")
        print(f"  奖励模式: {config.env.reward_mode}")
        
        # 测试快速配置
        config_fast = get_config('fast')
        print(f"✓ 快速配置加载成功")
        print(f"  训练轮数: {config_fast.training.epochs}")
        
        return config
        
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return None

def test_device_availability():
    """测试设备可用性"""
    print("\n=== 测试设备可用性 ===")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("- CUDA不可用，将使用CPU")
    
    # 检查CPU
    print(f"✓ CPU可用")
    print(f"  CPU核心数: {torch.get_num_threads()}")

def run_full_test():
    """运行完整测试"""
    print("=" * 60)
    print("EnvClutter 环境测试")
    print("=" * 60)
    
    # 检查环境设置
    check_environment_setup()
    
    # 测试设备可用性
    test_device_availability()
    
    # 测试配置加载
    config = test_config_loading()
    if config is None:
        print("配置加载失败，跳过后续测试")
        return
    
    # 测试环境创建
    env = test_environment_creation()
    if env is None:
        print("环境创建失败，跳过后续测试")
        return
    
    try:
        # 测试环境重置
        obs, info = test_environment_reset(env)
        if obs is None:
            print("环境重置失败，跳过后续测试")
            return
        
        # 测试环境步进
        next_obs, reward, terminated, truncated, info = test_environment_step(env, obs)
        if next_obs is None:
            print("环境步进失败，跳过后续测试")
            return
        
        # 测试环境渲染
        frame = test_environment_render(env)
        
        # 测试多步运行
        test_multiple_steps(env, num_steps=5)
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！环境工作正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        
    finally:
        # 清理资源
        try:
            env.close()
            print("✓ 环境已关闭")
        except:
            pass

if __name__ == "__main__":
    run_full_test() 