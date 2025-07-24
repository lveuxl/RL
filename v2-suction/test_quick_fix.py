#!/usr/bin/env python3
"""
快速测试最新修复的8状态抓取功能
主要修复：
1. 修复运行中成功条件从2cm->8cm的bug
2. 减少z轴高度变化，使目标更容易到达
3. 改进卡住检测和救援机制
4. 进一步放宽成功阈值到12cm
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import time

from env_clutter import EnvClutterEnv

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def test_quick_fix():
    """快速测试修复效果"""
    print("=== 快速测试最新修复效果 ===")
    print("🔧 最新修复：")
    print("1. 修复bug：运行中成功条件从2cm->8cm")
    print("2. 减少z轴变化：状态0(15cm)，状态1(3cm)，状态2(1cm)")
    print("3. 改进卡住检测：15步+救援机制")
    print("4. 放宽最终阈值：12cm")
    print()
    
    # 配置视频录制参数 - 优化视频质量和长度
    capture_video = True
    save_trajectory = False
    test_name = f"test_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    

    # 创建环境
    env = EnvClutterEnv(
        render_mode="rgb_array",  # 不显示界面，专注测试
        obs_mode="state", 
        control_mode="pd_ee_delta_pose",
        use_discrete_action=True,
        num_envs=1
    )
    
    print(f"✅ 环境创建成功")
    print(f"🎯 新的成功阈值: 8cm (运行中), 12cm (最终)")
    print()
    # 添加视频录制包装器 - 优化参数
    if capture_video or save_trajectory:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"视频将保存到: {video_output_dir}")
        
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=save_trajectory,
            save_video=capture_video,
            trajectory_name="test_trajectory",
            max_steps_per_video=5000,  # 增加到1200步以容纳等待时间
            video_fps=120,  # 提高帧率到60fps
            render_substeps=True,  # 启用子步渲染以获得更流畅的视频
            info_on_video=True,  # 在视频上显示信息
        )
        print("✓ 视频录制包装器添加成功")
    
    # 添加向量化包装器
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
    print("✓ 向量化包装器添加成功")

    try:
        # 快速测试2次
        total_episodes = 2
        success_count = 0
        
        for episode in range(total_episodes):
            print(f"\n🎮 === 快速测试 {episode + 1}/{total_episodes} ===")
            
            obs, info = env.reset()
            episode_start_time = time.time()
            
            unwrapped_env = env.unwrapped
            # 测试一次抓取
            if hasattr(unwrapped_env, 'remaining_indices') and unwrapped_env.remaining_indices:

                action_idx = 3  # 选择第一个可用物体
                target_obj_idx = unwrapped_env.remaining_indices[action_idx]
                
                print(f"🎯 开始抓取物体索引 {target_obj_idx}")
                
                # 执行抓取动作
                obs, reward, terminated, truncated, info = env.step(action_idx)
                
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                
                # 统计结果
                success = info.get('success', False)
                displacement = info.get('displacement', 0)
                
                print(f"\n📊 测试 {episode + 1} 结果:")
                print(f"  ✅ 成功: {'✅ 是' if success else '❌ 否'}")
                print(f"  ⏱️  耗时: {episode_duration:.2f}秒")
                print(f"  🏆 奖励: {reward.item():.3f}")
                print(f"  📏 位移: {displacement:.3f}m")
                
                # 累计统计
                if success:
                    success_count += 1
                    print(f"  🎉 成功! 当前成功率: {success_count}/{episode+1} = {success_count/(episode+1)*100:.1f}%")
                else:
                    print(f"  😞 失败! 当前成功率: {success_count}/{episode+1} = {success_count/(episode+1)*100:.1f}%")
            else:
                print("❌ 没有可抓取的物体")
        
        # 快速结果
        success_rate = success_count / total_episodes * 100
        print(f"\n📈 === 快速测试结果 ===")
        print(f"🔢 测试次数: {total_episodes}")
        print(f"✅ 成功次数: {success_count}")
        print(f"📊 成功率: {success_rate:.1f}%")
        
        if success_rate > 0:
            print(f"🎉 修复有效！成功率从0%提升到{success_rate:.1f}%")
            print(f"💡 建议：可以进行更大规模的测试")
        else:
            print(f"⚠️ 仍需进一步调试")
            print(f"💡 建议：检查机械臂工作空间限制")
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔚 快速测试完成")

if __name__ == "__main__":
    test_quick_fix() 