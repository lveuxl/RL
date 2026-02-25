#!/usr/bin/env python3
"""
多环境可视化演示脚本
基于官方ManiSkill实现，支持gym.make时设置num_envs=4和parallel_in_single_scene=True

主要特性：
1. 基于官方BaseEnv的parallel_in_single_scene实现
2. 支持num_envs=4的多环境并行
3. 所有环境显示在同一个视图中
4. 支持人工交互和可视化
"""

import argparse
import time
import numpy as np
import gymnasium as gym
import torch

# 导入ManiSkill和环境
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from env_clutter import EnvClutterEnv


def create_multi_env_visualization_demo():
    """创建多环境可视化演示环境 - 基于官方实现"""
    
    print("=== 创建多环境可视化演示 ===")
    print("🎯 参数配置：")
    print("  - num_envs = 4")
    print("  - parallel_in_single_scene = True") 
    print("  - render_mode = 'human'")
    print("  - obs_mode = 'state' (兼容parallel_in_single_scene)")
    print()
    
    # 基于官方demo_random_action.py的实现方式
    env_kwargs = {
        # 核心多环境参数
        "num_envs": 4,
        "parallel_in_single_scene": True,  # 官方参数：所有环境显示在同一视图
        
        # 观察和控制模式
        "obs_mode": "state",  # 根据官方文档，parallel_in_single_scene需要使用state模式
        "control_mode": "pd_ee_delta_pose",
        "reward_mode": "dense",
        
        # 渲染配置
        "render_mode": "human",  # 人机交互模式
        
        # 仿真后端配置
        "sim_backend": "gpu",  # GPU仿真支持多环境
        "render_backend": "gpu",  # GPU渲染
        
        # 其他配置
        "enable_shadow": True,  # 增强视觉效果
        "robot_uids": "panda",  # 使用Panda机械臂
    }
    
    # 使用gym.make创建环境 - 完全基于官方API
    env = gym.make("EnvClutter-v1", **env_kwargs)
    
    print("✅ 多环境创建成功！")
    print(f"   环境数量: {env.num_envs}")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print()
    
    return env


def run_multi_env_demo(env, max_steps=100):
    """运行多环境演示"""
    
    print("=== 开始多环境演示 ===")
    print("💡 提示：")
    print("  - 你将看到4个环境同时显示在一个场景中")
    print("  - 每个环境都有独立的机械臂和物体")
    print("  - 环境会自动执行随机动作")
    print("  - 按ESC或关闭窗口退出")
    print()
    
    # 重置环境
    obs, info = env.reset()
    print("🔄 环境重置完成")
    
    # 运行演示循环
    for step in range(max_steps):
        # 为每个环境生成随机动作
        actions = env.action_space.sample()
        
        # 执行动作
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # 打印状态信息
        if step % 10 == 0:
            print(f"Step {step:3d}: ", end="")
            for i in range(env.num_envs):
                print(f"Env{i} R={rewards[i]:.2f} ", end="")
            print()
        
        # 检查是否有环境结束
        if terminations.any() or truncations.any():
            print(f"\n有环境在第{step}步结束，重置中...")
            obs, info = env.reset()
            time.sleep(1)  # 暂停观察重置效果
        
        # 适当延迟以便观察
        time.sleep(0.1)
    
    print("\n=== 演示结束 ===")


def run_manual_control_demo(env):
    """手动控制演示"""
    
    print("=== 手动控制模式 ===")
    print("💡 控制说明：")
    print("  - 输入数字选择动作: 0~6")
    print("  - 输入'r'重置环境")
    print("  - 输入'q'退出")
    print("  - 输入'help'显示帮助")
    print()
    
    obs, info = env.reset()
    
    while True:
        try:
            cmd = input("请输入命令 (0-6/r/q/help): ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'r':
                obs, info = env.reset()
                print("🔄 环境已重置")
                continue
            elif cmd == 'help':
                print("动作空间说明:")
                if hasattr(env.unwrapped, 'ACTION_NAMES'):
                    for i, name in enumerate(env.unwrapped.ACTION_NAMES):
                        print(f"  {i}: {name}")
                else:
                    print("  动作空间大小:", env.action_space.shape)
                continue
            
            # 尝试解析为动作
            try:
                action_idx = int(cmd)
                if 0 <= action_idx < env.action_space.n:
                    # 为所有环境执行相同动作
                    actions = np.full(env.num_envs, action_idx, dtype=np.int64)
                    obs, rewards, terminations, truncations, infos = env.step(actions)
                    
                    print("执行结果:")
                    for i in range(env.num_envs):
                        print(f"  Env{i}: R={rewards[i]:.3f}, Done={terminations[i] or truncations[i]}")
                    
                    if (terminations | truncations).any():
                        print("有环境结束，自动重置...")
                        obs, info = env.reset()
                else:
                    print(f"❌ 动作索引超出范围 [0, {env.action_space.n-1}]")
            except ValueError:
                print("❌ 请输入有效的数字或命令")
                
        except KeyboardInterrupt:
            break
    
    print("手动控制结束")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多环境可视化演示")
    parser.add_argument("--mode", type=str, default="auto", 
                       choices=["auto", "manual"],
                       help="运行模式: auto(自动演示) 或 manual(手动控制)")
    parser.add_argument("--steps", type=int, default=200,
                       help="自动模式的最大步数")
    
    args = parser.parse_args()
    
    try:
        # 创建多环境
        env = create_multi_env_visualization_demo()
        
        # 根据模式运行演示
        if args.mode == "auto":
            run_multi_env_demo(env, max_steps=args.steps)
        elif args.mode == "manual":
            run_manual_control_demo(env)
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'env' in locals():
            env.close()
            print("✅ 环境已关闭")


if __name__ == "__main__":
    main()

