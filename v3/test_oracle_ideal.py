#!/usr/bin/env python3
"""
理想化神谕抓取测试脚本
主要特性：
1. 理想化神谕抓取 - 100%可预测的成功/失败逻辑
2. 遮挡与支撑检查 - 基于物理查询的逻辑判断
3. 完美夹爪控制 - 替代吸盘约束的精确抓取
4. 视频录制功能 - 验证抓取流程的视觉效果
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import time

# 注意：导入修改后的环境文件（文件名包含空格，需要特殊处理）
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("env_clutter", "env_clutter copy.py")
env_clutter = importlib.util.module_from_spec(spec)
sys.modules["env_clutter"] = env_clutter
spec.loader.exec_module(env_clutter)
from env_clutter import EnvClutterEnv

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def test_oracle_ideal():
    """测试理想化神谕抓取"""
    print("=== 🔮 理想化神谕抓取测试 ===")
    print("🎯 核心特性：")
    print("1. 神谕逻辑检查 - 遮挡与支撑关系判断")
    print("2. 100%可预测性 - 逻辑正确→必成功，逻辑错误→必失败")
    print("3. 理想化FSM - 预抓取→抓取→提升→瞬移→回位")
    print("4. 夹爪精确控制 - 替代吸盘约束的真实抓取")
    print()
    
    # 配置测试参数
    num_envs = 1  # 单环境测试，便于观察
    capture_video = True
    save_trajectory = False
    test_name = f"oracle_ideal_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    
    print(f"📹 视频录制目录: {video_output_dir}")
    print()

    # 创建理想化神谕环境
    env = EnvClutterEnv(
        render_mode="rgb_array",
        obs_mode="state", 
        control_mode="pd_ee_delta_pose",
        use_discrete_action=True,     # 启用离散动作选择
        use_ideal_oracle=True,        # 🔮 启用理想化神谕模式
        num_envs=num_envs
    )
    
    print(f"✅ 理想化神谕环境创建成功")
    print(f"🎮 模式设置: 离散动作={env.use_discrete_action}, 神谕模式={env.use_ideal_oracle}")
    print()
    
    # 添加视频录制包装器
    if capture_video or save_trajectory:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"视频将保存到: {video_output_dir}")
        
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=save_trajectory,
            save_video=capture_video,
            trajectory_name="oracle_ideal_trajectory",
            max_steps_per_video=2000,  # 足够长以观察完整的理想化流程
            video_fps=60,
            render_substeps=True,
            info_on_video=True,
        )
        print("✅ 视频录制包装器添加成功")
    
    # 添加向量化包装器
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
    print("✅ 向量化包装器添加成功")
    print()

    try:
        # 测试理想化神谕抓取
        total_episodes = 2
        success_count = 0
        failure_count = 0
        
        print(f"🚀 开始测试 {total_episodes} 个episode")
        print("-" * 60)
        
        for episode in range(total_episodes):
            print(f"\n🎬 Episode {episode + 1}/{total_episodes}")
            
            obs, info = env.reset()
            episode_reward = 0
            episode_success = False
            step_count = 0
            max_steps = 500  # 防止无限循环
            
            print(f"📊 环境重置完成，开始执行")
            
            while step_count < max_steps:
                # 在理想化模式中，我们可以测试不同的策略
                if episode == 0:
                    # 第一个episode：尝试选择逻辑上"正确"的物体（通常是表层物体）
                    action = 0  # 选择第一个物体
                    strategy = "选择表层物体（预期成功）"
                else:
                    # 第二个episode：随机选择，可能遇到被遮挡或支撑的物体
                    available_actions = min(8, len(obs))  # 假设最多8个物体
                    action = np.random.randint(0, available_actions)
                    strategy = f"随机选择物体{action}（测试神谕判断）"
                
                if step_count == 0:
                    print(f"📋 执行策略: {strategy}")
                
                obs, reward, terminated, truncated, info = env.step([action])
                episode_reward += reward[0]
                step_count += 1
                
                # 打印重要状态信息（每50步输出一次，减少噪音）
                if step_count % 50 == 0:
                    print(f"   Step {step_count}: 奖励累计={episode_reward:.3f}")
                
                # 检查是否完成
                if terminated[0] or truncated[0]:
                    if reward[0] > 0:  # 假设正奖励表示成功
                        episode_success = True
                        success_count += 1
                        print(f"✅ Episode完成: 抓取成功! 总步数={step_count}, 奖励={episode_reward:.3f}")
                    else:
                        failure_count += 1
                        print(f"❌ Episode完成: 抓取失败或神谕拒绝. 总步数={step_count}, 奖励={episode_reward:.3f}")
                    break
            
            if step_count >= max_steps:
                failure_count += 1
                print(f"⏰ Episode超时: 达到最大步数{max_steps}")
        
        # 测试总结
        print("\n" + "="*60)
        print("🏆 理想化神谕抓取测试总结")
        print("="*60)
        print(f"📊 总episode数: {total_episodes}")
        print(f"✅ 成功次数: {success_count}")
        print(f"❌ 失败次数: {failure_count}")
        print(f"📈 成功率: {success_count/total_episodes*100:.1f}%")
        print()
        
        # 理想化神谕的预期行为说明
        print("🔮 理想化神谕预期行为:")
        print("- 逻辑正确的选择 → 100%成功（流畅的抓取→提升→瞬移流程）")
        print("- 逻辑错误的选择 → 100%失败（神谕在Stage 0直接拒绝）")
        print("- 所有成功的抓取都应展示完整的6阶段FSM流程")
        print()
        
        if capture_video:
            print(f"📹 测试视频已保存到: {video_output_dir}")
            print("   可通过视频验证:")
            print("   1. 神谕检查阶段的决策过程")
            print("   2. 理想化FSM的各个阶段")
            print("   3. 夹爪控制的精确性")
            print("   4. 物体瞬移的效果")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        try:
            env.close()
            print("🧹 环境资源已清理")
        except:
            pass


if __name__ == "__main__":
    print("🔮 启动理想化神谕抓取测试")
    print("=" * 60)
    test_oracle_ideal()
    print("🎉 测试完成!")
