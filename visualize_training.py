import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import argparse
from ppo_maniskill_training import make_env

def visualize_random_policy(num_episodes=5):
    """可视化随机策略，用于测试环境"""
    print("开始可视化随机策略...")
    
    # 创建单个环境用于可视化
    env = gym.make(
        "StackPickingManiSkill-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",  # 使用human模式进行可视化
        max_objects=3,
        robot_uids="panda",
    )
    
    for episode in range(num_episodes):
        print(f"\n开始第 {episode + 1} 个episode...")
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染环境
            env.render()
            
            total_reward += reward if hasattr(reward, 'item') else float(reward)
            steps += 1
            
            # 添加延迟以便观察
            time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} 完成:")
        print(f"  步数: {steps}")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  成功: {info.get('success', False)}")
        
        # 暂停一下再开始下一个episode
        time.sleep(1)
    
    env.close()

def visualize_trained_model(model_path, num_episodes=5):
    """可视化训练好的模型"""
    print(f"加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建环境
    env = gym.make(
        "StackPickingManiSkill-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        max_objects=3,
        robot_uids="panda",
    )
    
    # 加载模型
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        print(f"\n开始第 {episode + 1} 个episode...")
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 使用训练好的模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染环境
            env.render()
            
            total_reward += reward if hasattr(reward, 'item') else float(reward)
            steps += 1
            
            # 添加延迟以便观察
            time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} 完成:")
        print(f"  步数: {steps}")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  成功: {info.get('success', False)}")
        
        # 暂停一下再开始下一个episode
        time.sleep(1)
    
    env.close()

def visualize_multiple_envs(num_envs=4, num_episodes=3):
    """可视化多个环境（保存为视频文件）"""
    print(f"创建 {num_envs} 个环境进行可视化...")
    
    # 创建多个环境
    env_fns = [make_env("StackPickingManiSkill-v1", i, seed=42, enable_render=False) 
               for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    for episode in range(num_episodes):
        print(f"\n开始第 {episode + 1} 个episode...")
        obs = vec_env.reset()
        total_rewards = np.zeros(num_envs)
        steps = 0
        
        while True:
            # 随机动作
            actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
            obs, rewards, dones, infos = vec_env.step(actions)
            
            total_rewards += rewards
            steps += 1
            
            # 获取渲染图像并保存（可选）
            if steps % 50 == 0:  # 每50步输出一次状态
                print(f"  步数: {steps}, 平均奖励: {np.mean(total_rewards):.4f}")
            
            if np.any(dones):
                break
        
        print(f"Episode {episode + 1} 完成:")
        print(f"  步数: {steps}")
        print(f"  各环境总奖励: {total_rewards}")
        print(f"  平均奖励: {np.mean(total_rewards):.4f}")
    
    vec_env.close()

def main():
    parser = argparse.ArgumentParser(description='ManiSkill训练可视化工具')
    parser.add_argument('--mode', type=str, choices=['random', 'model', 'multi'], 
                       default='random', help='可视化模式')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='训练好的模型路径（仅用于model模式）')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='要运行的episode数量')
    parser.add_argument('--num_envs', type=int, default=4, 
                       help='多环境模式下的环境数量')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        print("=== 随机策略可视化 ===")
        visualize_random_policy(args.episodes)
    
    elif args.mode == 'model':
        if args.model_path is None:
            # 自动寻找最新的模型
            model_dir = "maniskill_ppo_model/trained_model"
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
                if models:
                    # 找到最新的模型
                    model_nums = [int(f.split('_')[-1].split('.')[0]) for f in models]
                    latest_model = f"maniskill_model_{max(model_nums)}.zip"
                    args.model_path = os.path.join(model_dir, latest_model)
                    print(f"自动选择最新模型: {args.model_path}")
                else:
                    print("未找到训练好的模型！")
                    return
            else:
                print("模型目录不存在！")
                return
        
        print("=== 训练模型可视化 ===")
        visualize_trained_model(args.model_path, args.episodes)
    
    elif args.mode == 'multi':
        print("=== 多环境可视化 ===")
        visualize_multiple_envs(args.num_envs, args.episodes)

if __name__ == "__main__":
    main() 