#!/usr/bin/env python3
"""
ManiSkill可视化训练演示脚本
展示如何在训练过程中启用实时可视化
"""

import os
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import mani_skill.envs

class VisualizationCallback(BaseCallback):
    """可视化训练回调"""
    
    def __init__(self, render_freq=10, verbose=1):
        super(VisualizationCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.step_count = 0
        self.episode_count = 0
    
    def _on_step(self):
        self.step_count += 1
        
        # 定期渲染环境
        if self.step_count % self.render_freq == 0:
            try:
                # 获取环境并渲染
                if hasattr(self.training_env, 'envs'):
                    env = self.training_env.envs[0]
                    if hasattr(env, 'render'):
                        img = env.render()
                        if img is not None and self.verbose > 0:
                            print(f"步数 {self.step_count}: 渲染图像形状 {img.shape}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"渲染错误: {e}")
        
        # 检查episode结束
        dones = self.locals.get('dones')
        if dones is not None and np.any(dones):
            self.episode_count += 1
            
            if self.verbose > 0 and self.episode_count % 5 == 0:
                print(f"已完成 {self.episode_count} 个episodes")
        
        return True

def make_visual_env(env_id, rank=0, seed=0, render_mode="rgb_array"):
    """创建可视化环境"""
    def _init():
        env = gym.make(
            env_id,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode=render_mode,
            robot_uids="panda",
        )
        
        # 设置种子 - 使用新的gymnasium标准方式
        try:
            # 新的gymnasium方式：通过reset传递种子
            env.reset(seed=seed + rank)
        except Exception:
            # 如果环境不支持新方式，尝试旧方式但避免警告
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'seed'):
                env.unwrapped.seed(seed + rank)
            elif hasattr(env, 'seed'):
                # 这里会产生弃用警告，但作为后备方案
                env.seed(seed + rank)
        
        return env
    
    return _init

def demo_visual_training():
    """演示可视化训练"""
    print("=== ManiSkill可视化训练演示 ===")
    
    # 创建输出目录
    output_dir = "visual_training_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建环境
    env_id = "PickCube-v1"
    render_mode = "rgb_array"  # 使用rgb_array模式，适合服务器环境
    
    print(f"创建环境: {env_id}")
    print(f"渲染模式: {render_mode}")
    
    # 创建向量化环境
    env_fns = [make_visual_env(env_id, i, seed=42, render_mode=render_mode) 
               for i in range(1)]  # 单环境便于观察
    vec_env = DummyVecEnv(env_fns)
    
    # 创建PPO模型 - 使用较小的参数便于快速演示
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard",
        n_steps=128,  # 较小的步数
        batch_size=32,
        learning_rate=3e-4,
        gamma=0.99,
        n_epochs=4,
        policy_kwargs={
            "net_arch": [64, 64],  # 较小的网络
            "activation_fn": torch.nn.ReLU,
        },
    )
    
    # 创建可视化回调
    vis_callback = VisualizationCallback(render_freq=50, verbose=1)
    
    print(f"\n{'='*50}")
    print("开始可视化训练演示")
    print("注意：这是一个快速演示，仅训练1000步")
    print(f"{'='*50}\n")
    
    try:
        # 开始训练
        model.learn(
            total_timesteps=1000,  # 较短的训练时间用于演示
            callback=vis_callback,
            progress_bar=True
        )
        
        print(f"\n{'='*50}")
        print("可视化训练演示完成!")
        print(f"{'='*50}\n")
        
        # 保存模型
        model_path = f"{output_dir}/demo_model"
        model.save(model_path)
        print(f"演示模型已保存到: {model_path}")
        
        # 演示训练好的模型
        print("\n开始演示训练好的模型...")
        demo_trained_model(model, vec_env)
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vec_env.close()

def demo_trained_model(model, env, num_episodes=3):
    """演示训练好的模型"""
    print(f"运行 {num_episodes} 个episodes来演示训练结果...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < 100:
            # 使用训练好的模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # 处理张量奖励
            reward_value = float(reward[0]) if hasattr(reward[0], 'item') else reward[0]
            total_reward += reward_value
            step_count += 1
            
            # 渲染环境
            try:
                img = env.envs[0].render()
                if img is not None and step_count % 10 == 0:
                    print(f"  步数 {step_count}: 渲染成功, 累计奖励: {total_reward:.3f}")
            except Exception as e:
                if step_count % 10 == 0:
                    print(f"  步数 {step_count}: 渲染失败, 累计奖励: {total_reward:.3f}")
            
            time.sleep(0.05)  # 控制播放速度
        
        print(f"Episode {episode + 1} 完成:")
        print(f"  总步数: {step_count}")
        print(f"  总奖励: {total_reward:.3f}")
        print(f"  平均奖励: {total_reward/step_count:.3f}")

def demo_environment_info():
    """演示环境信息"""
    print("=== 环境信息演示 ===")
    
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        robot_uids="panda",
    )
    
    print(f"环境ID: PickCube-v1")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"观察空间形状: {env.observation_space.shape}")
    print(f"动作空间形状: {env.action_space.shape}")
    print(f"动作空间范围: [{env.action_space.low[0]:.2f}, {env.action_space.high[0]:.2f}]")
    
    # 重置环境并获取初始观察
    obs, info = env.reset()
    print(f"初始观察形状: {obs.shape}")
    print(f"初始观察前10个元素: {obs[:10]}")
    
    # 执行几步随机动作
    print("\n执行5步随机动作:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 将PyTorch张量转换为Python数值
        reward_value = float(reward) if hasattr(reward, 'item') else reward
        
        print(f"  步数 {step + 1}:")
        print(f"    动作: {action}")
        print(f"    奖励: {reward_value:.4f}")
        print(f"    结束: {terminated or truncated}")
        
        # 尝试渲染
        try:
            img = env.render()
            if img is not None:
                print(f"    渲染图像形状: {img.shape}")
        except Exception as e:
            print(f"    渲染失败: {e}")
        
        if terminated or truncated:
            print("    Episode结束")
            break
    
    env.close()
    print("环境信息演示完成")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ManiSkill可视化训练演示')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'info', 'all'], 
                       default='all',
                       help='选择演示模式')
    
    args = parser.parse_args()
    
    print("ManiSkill可视化训练演示")
    print("适用于服务器环境（无图形界面）")
    print("-" * 50)
    
    try:
        if args.mode == 'info' or args.mode == 'all':
            demo_environment_info()
            print()
        
        if args.mode == 'train' or args.mode == 'all':
            demo_visual_training()
    
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n演示完成！")

if __name__ == "__main__":
    main() 