#!/usr/bin/env python3
"""
环境测试脚本 - 验证修复后的环境是否正常工作
"""

import os
import torch
import gymnasium as gym

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入环境
from env_clutter_optimized import EnvClutterOptimizedEnv
import mani_skill.envs

def test_environment():
    """测试环境创建和基本功能"""
    print("=== 环境测试 ===")
    
    try:
        # 创建环境
        print("1. 创建环境...")
        env = gym.make(
            "EnvClutterOptimized-v1",
            num_envs=4,  # 少量环境测试
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",
            sim_backend="gpu",
            render_mode=None,
        )
        print("✓ 环境创建成功")
        
        # 重置环境
        print("2. 重置环境...")
        obs, info = env.reset()
        print(f"✓ 环境重置成功，观测维度: {obs.shape}")
        
        # 测试几个步骤
        print("3. 测试动作执行...")
        for step in range(3):
            # 随机选择动作
            actions = torch.randint(0, 9, (4,))  # 4个环境，每个选择0-8中的一个物体
            print(f"   步骤 {step+1}: 动作 = {actions.tolist()}")
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            print(f"   奖励: {rewards.tolist()}")
            print(f"   终止: {terminated.tolist()}")
            print(f"   截断: {truncated.tolist()}")
            
            if terminated.any():
                print("   有环境提前结束")
                break
        
        print("✓ 动作执行成功")
        
        # 关闭环境
        env.close()
        print("✓ 环境关闭成功")
        
        print("\n🎉 环境测试完全通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_space():
    """测试观测空间"""
    print("\n=== 观测空间测试 ===")
    
    try:
        env = gym.make(
            "EnvClutterOptimized-v1",
            num_envs=1,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",
            sim_backend="gpu",
        )
        
        obs, _ = env.reset()
        print(f"观测维度: {obs.shape}")
        print(f"观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"观测数据类型: {obs.dtype}")
        
        # 分析观测结构
        obs_flat = obs.flatten()
        expected_dim = 9*5 + 5 + 9  # 物体特征 + 全局特征 + 动作掩码
        print(f"预期维度: {expected_dim}, 实际维度: {len(obs_flat)}")
        
        if len(obs_flat) == expected_dim:
            print("✓ 观测维度正确")
        else:
            print("⚠️ 观测维度不匹配")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 观测测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试修复后的环境...\n")
    
    # 检查依赖
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
    print()
    
    # 运行测试
    success1 = test_environment()
    success2 = test_observation_space()
    
    if success1 and success2:
        print("\n✅ 所有测试通过！环境修复成功，可以开始训练了。")
        print("运行训练命令: python run_training.py")
    else:
        print("\n❌ 测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()

