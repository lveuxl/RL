#!/usr/bin/env python3
"""
测试多环境可视化功能
基于官方ManiSkill API的最小示例
"""

import gymnasium as gym
import numpy as np
import time

# 导入ManiSkill环境
import mani_skill.envs
from env_clutter import EnvClutterEnv


def test_multi_env_basic():
    """基础多环境测试"""
    print("=== 基础多环境测试 ===")
    
    # 使用gym.make创建多环境，参考官方demo_random_action.py的实现
    env = gym.make(
        "EnvClutter-v1",
        num_envs=4,                      # 4个并行环境
        parallel_in_single_scene=True,   # 所有环境显示在同一场景中
        obs_mode="state",                # 状态观察模式，兼容parallel_in_single_scene
        control_mode="pd_ee_delta_pose", # 控制模式
        reward_mode="dense",             # 奖励模式
        sim_backend="gpu",               # GPU仿真
        render_mode="human",             # 人机交互渲染
        robot_uids="panda"               # 使用Panda机械臂
    )
    
    print(f"✅ 环境创建成功:")
    print(f"  环境数量: {env.num_envs}")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print()
    
    return env


def run_simple_demo(env, steps=50):
    """运行简单演示"""
    print("🎬 开始运行演示...")
    print("💡 你应该看到4个环境同时显示在同一个窗口中")
    
    # 重置环境
    obs, info = env.reset()
    
    for step in range(steps):
        # 生成随机动作
        actions = env.action_space.sample()
        
        # 执行动作
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # 每10步打印一次信息
        if step % 10 == 0:
            print(f"Step {step}: rewards = {[f'{r:.2f}' for r in rewards]}")
        
        # 如果有环境结束，重置
        if terminations.any() or truncations.any():
            obs, info = env.reset()
            print("🔄 环境重置")
        
        time.sleep(0.1)  # 短暂延迟以便观察
    
    print("✅ 演示完成")


def main():
    """主函数"""
    try:
        print("多环境可视化测试")
        print("==================")
        print("参数配置:")
        print("- num_envs = 4")
        print("- parallel_in_single_scene = True")
        print("- render_mode = 'human'")
        print()
        
        # 创建环境
        env = test_multi_env_basic()
        
        # 运行演示
        run_simple_demo(env, steps=100)
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
            print("🔧 环境已关闭")


if __name__ == "__main__":
    main()

