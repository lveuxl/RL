#!/usr/bin/env python3
"""
ManiSkill环境可视化演示脚本
展示如何在ManiSkill环境中启用实时可视化
"""

import time
import numpy as np
import gymnasium as gym
import mani_skill.envs

def demo_basic_visualization():
    """基础可视化演示"""
    print("=== ManiSkill环境基础可视化演示 ===")
    
    # 创建环境 - 使用ManiSkill内置的PickCube环境
    env = gym.make(
        "PickCube-v1",  # 使用内置环境
        obs_mode="state",
        control_mode="pd_joint_delta_pos", 
        render_mode="human",  # 启用窗口显示
        robot_uids="panda",
    )
    
    print("环境已创建，开始演示...")
    print("提示：关闭渲染窗口可结束演示")
    
    try:
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")
            obs, info = env.reset()
            print(f"初始观察空间形状: {obs.shape}")
            
            done = False
            step_count = 0
            total_reward = 0
            
            while not done and step_count < 50:
                # 生成随机动作
                action = env.action_space.sample()
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                step_count += 1
                
                # 渲染环境
                env.render()
                
                # 控制播放速度
                time.sleep(0.1)
                
                # 每10步打印一次信息
                if step_count % 10 == 0:
                    print(f"  步数: {step_count}, 累计奖励: {total_reward:.3f}")
            
            print(f"Episode {episode + 1} 完成")
            print(f"  总步数: {step_count}")
            print(f"  总奖励: {total_reward:.3f}")
            print(f"  平均奖励: {total_reward/step_count:.3f}")
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("环境已关闭")

def demo_controlled_visualization():
    """受控可视化演示 - 展示特定动作"""
    print("\n=== 受控动作可视化演示 ===")
    
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        robot_uids="panda",
    )
    
    try:
        obs, info = env.reset()
        print("开始受控动作演示...")
        
        # 演示不同类型的动作
        actions = [
            ("向前移动", np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            ("向上移动", np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])),
            ("夹爪开合", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])),
            ("复合动作", np.array([0.05, 0.05, -0.05, 0.1, 0.0, 0.0, -0.2])),
        ]
        
        for action_name, action in actions:
            print(f"\n执行动作: {action_name}")
            
            # 执行动作多次以观察效果
            for i in range(10):
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(0.2)
                
                if terminated or truncated:
                    print("Episode结束，重置环境...")
                    obs, info = env.reset()
                    break
            
            # 停顿一下让用户观察
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    finally:
        env.close()
        print("受控演示结束")

def demo_observation_visualization():
    """观察数据可视化演示"""
    print("\n=== 观察数据可视化演示 ===")
    
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        robot_uids="panda",
    )
    
    try:
        obs, info = env.reset()
        
        print(f"观察空间维度: {env.observation_space.shape}")
        print(f"动作空间维度: {env.action_space.shape}")
        print(f"动作空间范围: {env.action_space.low} 到 {env.action_space.high}")
        
        print("\n开始观察数据演示...")
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            
            # 每5步显示一次详细信息
            if step % 5 == 0:
                print(f"\n步数 {step}:")
                print(f"  观察向量前10个元素: {obs[:10]}")
                print(f"  奖励: {reward:.4f}")
                print(f"  动作: {action}")
                if info:
                    print(f"  额外信息: {info}")
            
            time.sleep(0.15)
            
            if terminated or truncated:
                print("Episode结束，重置环境...")
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    finally:
        env.close()
        print("观察数据演示结束")

def demo_different_render_modes():
    """演示不同的渲染模式"""
    print("\n=== 不同渲染模式演示 ===")
    
    render_modes = ["human", "rgb_array"]
    
    for mode in render_modes:
        print(f"\n--- 渲染模式: {mode} ---")
        
        env = gym.make(
            "PickCube-v1",
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode=mode,
            robot_uids="panda",
        )
        
        try:
            obs, info = env.reset()
            
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 渲染环境
                if mode == "human":
                    env.render()
                    time.sleep(0.1)
                elif mode == "rgb_array":
                    img = env.render()
                    if img is not None:
                        print(f"  步数 {step}: 图像形状 {img.shape}")
                
                if terminated or truncated:
                    print(f"  Episode在步数 {step} 结束")
                    break
        
        except Exception as e:
            print(f"渲染模式 {mode} 发生错误: {e}")
        finally:
            env.close()
            print(f"渲染模式 {mode} 演示结束")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ManiSkill可视化演示')
    parser.add_argument('--demo', type=str, 
                       choices=['basic', 'controlled', 'observation', 'render_modes', 'all'],
                       default='basic',
                       help='选择演示类型')
    
    args = parser.parse_args()
    
    print("ManiSkill环境可视化演示")
    print("请确保您有图形界面支持")
    print("-" * 50)
    
    try:
        if args.demo == 'basic' or args.demo == 'all':
            demo_basic_visualization()
        
        if args.demo == 'controlled' or args.demo == 'all':
            demo_controlled_visualization()
        
        if args.demo == 'observation' or args.demo == 'all':
            demo_observation_visualization()
        
        if args.demo == 'render_modes' or args.demo == 'all':
            demo_different_render_modes()
    
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n演示完成！")

if __name__ == "__main__":
    main() 