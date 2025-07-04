#!/usr/bin/env python3
"""
测试ManiSkill环境可视化的脚本
用于验证渲染是否正常工作
"""

import gymnasium as gym
import time
import numpy as np

# 导入ManiSkill相关模块以注册环境
import mani_skill.envs
from stack_picking_maniskill_env import StackPickingManiSkillEnv

def test_visualization():
    """测试可视化功能"""
    print("正在创建ManiSkill环境...")
    
    try:
        # 创建环境，启用人类渲染模式
        env = gym.make(
            "StackPickingManiSkill-v1",
            obs_mode="state",
            control_mode="pd_joint_delta_pos", 
            render_mode="human",  # 人类渲染模式
            max_objects=3,
            robot_uids="panda",
        )
        
        print("环境创建成功！")
        print("观测空间:", env.observation_space)
        print("动作空间:", env.action_space)
        
        # 运行几个episode来测试可视化
        for episode in range(3):
            print(f"\n开始第 {episode + 1} 个episode...")
            
            obs, info = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            
            while not done and step_count < 100:
                # 随机动作
                action = env.action_space.sample()
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 将张量转换为标量
                if hasattr(reward, 'item'):
                    reward_scalar = reward.item()
                else:
                    reward_scalar = float(reward)
                
                total_reward += reward_scalar
                step_count += 1
                
                # 渲染环境
                try:
                    env.render()
                    time.sleep(0.02)  # 稍微延迟以便观察
                except Exception as e:
                    print(f"渲染错误: {e}")
                
                # 打印一些信息
                if step_count % 20 == 0:
                    print(f"步数: {step_count}, 奖励: {reward_scalar:.3f}, 总奖励: {total_reward:.3f}")
            
            print(f"Episode {episode + 1} 完成:")
            print(f"  总步数: {step_count}")
            print(f"  总奖励: {total_reward:.3f}")
            print(f"  成功: {info.get('success', False)}")
            
            time.sleep(1)  # episode间隔
        
        env.close()
        print("\n可视化测试完成！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization() 