#!/usr/bin/env python3
"""
测试EnvClutter环境的可视化界面
"""

import gymnasium as gym
import numpy as np
import time
import mani_skill.envs
from env_clutter import EnvClutterEnv

def test_visualization():
    """测试可视化界面"""
    print("=== EnvClutter环境可视化测试 ===")
    
    # 创建环境
    env = gym.make(
        "EnvClutter-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode="human",  # 开启可视化
    )
    
    print("环境创建成功！")
    print("控制说明：")
    print("- 环境将自动执行随机动作")
    print("- 按 Ctrl+C 退出")
    print("- 观察机械臂和物体的交互")
    
    try:
        # 重置环境
        obs, info = env.reset()
        print(f"环境重置成功，观测维度: {len(obs) if isinstance(obs, dict) else obs.shape}")
        
        episode_count = 0
        step_count = 0
        
        while True:
            # 随机动作
            action = env.action_space.sample()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # 渲染环境
            env.render()
            
            # 打印信息
            if step_count % 20 == 0:
                success = info.get('success', False) if isinstance(info, dict) else False
                print(f"步骤 {step_count}: 奖励={reward:.3f}, 成功={success}")
            
            # 如果episode结束
            if terminated or truncated:
                episode_count += 1
                success = info.get('success', False) if isinstance(info, dict) else False
                print(f"\nEpisode {episode_count} 结束!")
                print(f"总步数: {step_count}")
                print(f"最终奖励: {reward:.3f}")
                print(f"任务成功: {success}")
                
                # 重置环境
                obs, info = env.reset()
                step_count = 0
                
                # 询问是否继续
                print("\n按 Enter 继续下一个episode，或按 Ctrl+C 退出...")
                input()
            
            # 控制帧率
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n用户中断，退出测试")
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("环境已关闭")

def test_random_policy():
    """测试随机策略的性能"""
    print("\n=== 随机策略性能测试 ===")
    
    env = gym.make(
        "EnvClutter-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode="human",
    )
    
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                env.render()
                
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    success = info.get('success', False) if isinstance(info, dict) else False
                    if success:
                        success_count += 1
                    
                    print(f"Episode {episode + 1} 结束:")
                    print(f"  奖励: {episode_reward:.3f}")
                    print(f"  长度: {episode_length}")
                    print(f"  成功: {success}")
                    break
                
                time.sleep(0.02)
        
        # 统计结果
        print(f"\n=== 随机策略统计结果 ===")
        print(f"平均奖励: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"成功率: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EnvClutter环境可视化测试')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['interactive', 'random_policy'],
                       help='测试模式')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        test_visualization()
    elif args.mode == 'random_policy':
        test_random_policy() 