#!/usr/bin/env python3
"""
测试EnvClutter环境的托盘功能
"""

import numpy as np
import torch
import gymnasium as gym
import mani_skill
from env_clutter import EnvClutterEnv
import time

def test_tray_environment():
    """测试托盘环境的基本功能"""
    print("开始测试托盘环境...")
    
    try:
        # 创建环境
        env = gym.make(
            "EnvClutter-v1",
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode="human",
            num_envs=1,
        )
        
        print("✅ 环境创建成功")
        
        # 重置环境
        obs, info = env.reset()
        print("✅ 环境重置成功")
        
        # 检查观测空间
        print(f"观测数据类型: {type(obs)}")
        print(f"观测数据键: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
        
        # 安全地访问观测数据
        if isinstance(obs, dict):
            if 'agent' in obs:
                agent_obs = obs['agent']
                print(f"Agent观测类型: {type(agent_obs)}")
                if isinstance(agent_obs, dict):
                    print(f"Agent观测键: {agent_obs.keys()}")
                    if 'qpos' in agent_obs:
                        print(f"qpos形状: {agent_obs['qpos'].shape}")
                elif hasattr(agent_obs, 'shape'):
                    print(f"Agent观测形状: {agent_obs.shape}")
        
        print(f"动作空间维度: {env.action_space.shape}")
        
        # 运行几个步骤
        print("\n开始运行环境...")
        for step in range(10):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"步骤 {step + 1}:")
            print(f"  奖励: {reward}")
            print(f"  是否抓取: {info.get('is_grasped', 'N/A')}")
            print(f"  是否成功: {info.get('success', 'N/A')}")
            
            # 短暂暂停以便观察
            time.sleep(0.1)
            
            if terminated or truncated:
                print("Episode结束，重置环境...")
                obs, info = env.reset()
        
        print("\n✅ 托盘环境测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'env' in locals():
            env.close()

def test_tray_loading():
    """测试托盘加载功能"""
    print("\n开始测试托盘加载...")
    
    try:
        # 创建环境但不渲染
        env = gym.make(
            "EnvClutter-v1",
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            num_envs=1,
        )
        
        # 检查托盘是否正确加载
        if hasattr(env.unwrapped, 'merged_trays'):
            print("✅ 托盘成功加载")
            print(f"托盘数量: {len(env.unwrapped.trays)}")
        else:
            print("❌ 托盘未正确加载")
        
        # 检查物体是否在托盘内生成
        obs, info = env.reset()
        
        if hasattr(env.unwrapped, 'merged_objects'):
            object_positions = env.unwrapped.merged_objects.pose.p
            print(f"物体位置: {object_positions}")
            
            # 检查物体是否在托盘范围内
            tray_center = [0.1, 0.0]
            tray_size = env.unwrapped.tray_spawn_area
            
            in_tray = True
            for pos in object_positions:
                x, y = pos[0].item(), pos[1].item()
                if (abs(x - tray_center[0]) > tray_size[0] or 
                    abs(y - tray_center[1]) > tray_size[1]):
                    in_tray = False
                    break
            
            if in_tray:
                print("✅ 物体正确生成在托盘内")
            else:
                print("❌ 物体生成位置超出托盘范围")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 托盘加载测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_object_positions():
    """测试物体位置生成"""
    print("\n开始测试物体位置生成...")
    
    try:
        env = gym.make(
            "EnvClutter-v1",
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            num_envs=1,
        )
        
        # 测试多次重置，检查物体位置变化
        positions_history = []
        
        for i in range(5):
            obs, info = env.reset()
            if hasattr(env.unwrapped, 'merged_objects'):
                positions = env.unwrapped.merged_objects.pose.p.cpu().numpy()
                positions_history.append(positions)
                print(f"重置 {i+1} - 物体位置: {positions[0]}")  # 只打印第一个物体
        
        # 检查位置是否有变化（随机化）
        if len(positions_history) > 1:
            pos_diff = np.linalg.norm(positions_history[0] - positions_history[1])
            if pos_diff > 0.01:  # 如果位置差异大于1cm
                print("✅ 物体位置正确随机化")
            else:
                print("⚠️ 物体位置可能没有充分随机化")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 物体位置测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_simple_creation():
    """简单的环境创建测试"""
    print("\n开始简单环境创建测试...")
    
    try:
        # 创建最简单的环境
        env = gym.make(
            "EnvClutter-v1",
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            num_envs=1,
        )
        
        print("✅ 环境创建成功")
        
        # 重置环境
        obs, info = env.reset()
        print("✅ 环境重置成功")
        
        # 检查基本信息
        print(f"观测数据类型: {type(obs)}")
        print(f"信息数据类型: {type(info)}")
        print(f"动作空间: {env.action_space}")
        
        # 执行一个随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✅ 步骤执行成功")
        print(f"奖励: {reward}")
        print(f"终止: {terminated}")
        print(f"截断: {truncated}")
        
        env.close()
        print("✅ 环境关闭成功")
        
    except Exception as e:
        print(f"❌ 简单创建测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== EnvClutter 托盘功能测试 ===")
    
    # 先进行简单的创建测试
    test_simple_creation()
    
    # 测试托盘加载
    test_tray_loading()
    
    # 测试物体位置生成
    test_object_positions()
    
    # 测试完整环境运行
    test_tray_environment()
    
    print("\n=== 测试完成 ===") 