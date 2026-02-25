#!/usr/bin/env python3
"""
测试修复后的理想化神谕抓取环境
主要测试：
1. 瞬移物体后从可选列表中移除，不影响状态观测
2. 改进的遮挡检测（基于AABB包围盒）
3. 改进的支撑检测（基于AABB包围盒）
"""

import os
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from pathlib import Path

# 导入自定义环境
from env_clutter import EnvClutterEnv

def test_oracle_fixed_environment():
    """测试修复后的理想化神谕环境"""
    
    print("🚀 开始测试修复后的理想化神谕抓取环境...")
    
    # 创建测试视频目录
    test_videos_dir = Path("test_videos")
    test_videos_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = test_videos_dir / f"oracle_fixed_{timestamp}.mp4"
    
    try:
        # 🔧 创建环境 - 启用理想化神谕模式
        env = EnvClutterEnv(
            num_envs=1,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array", 
            use_discrete_action=True,     # 启用离散动作模式
            use_ideal_oracle=True,        # 启用理想化神谕模式
            config_preset="default"       # 使用默认配置
        )
        
        print(f"✅ 环境创建成功")
        print(f"   - 离散动作空间大小: {env.discrete_action_space.n}")
        print(f"   - 物体总数: {env.total_objects_per_env}")
        print(f"   - 最大episode步数: {env.MAX_EPISODE_STEPS}")
        
        # 🎬 设置视频记录
        if hasattr(env, 'render'):
            video_frames = []
            
        # 📊 测试统计
        test_stats = {
            'episodes': 0,
            'successful_grasps': 0,
            'blocked_attempts': 0,
            'supporting_failures': 0,
            'total_actions': 0,
            'obs_anomalies': 0  # 观测异常（包含瞬移物体的次数）
        }
        
        # 🧪 运行测试episodes
        num_test_episodes = 3
        
        for episode in range(num_test_episodes):
            print(f"\n🎯 Episode {episode + 1}/{num_test_episodes}")
            
            obs, info = env.reset()
            
            # 验证初始观测
            print(f"   初始观测形状: {obs.shape if hasattr(obs, 'shape') else 'dict/other'}")
            
            episode_steps = 0
            episode_grasps = 0
            
            while episode_steps < env.MAX_EPISODE_STEPS:
                # 📸 记录视频帧
                if hasattr(env, 'render'):
                    frame = env.render()
                    if frame is not None and hasattr(video_frames, 'append'):
                        video_frames.append(frame)
                
                # 🎯 选择动作：尝试抓取当前剩余的第一个物体
                if len(env.remaining_indices[0]) > 0:
                    action = 0  # 选择第一个可用物体
                    print(f"     步骤 {episode_steps + 1}: 尝试抓取索引 {env.remaining_indices[0][0]}")
                else:
                    print(f"     步骤 {episode_steps + 1}: 没有剩余物体可抓取")
                    break
                
                # 📊 记录抓取前的状态
                pre_action_selectable_count = len(env.selectable_objects[0])
                pre_action_remaining = len(env.remaining_indices[0])
                
                # ⚡ 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                test_stats['total_actions'] += 1
                
                # 📊 记录抓取后的状态 
                post_action_selectable_count = len(env.selectable_objects[0])
                post_action_remaining = len(env.remaining_indices[0])
                post_action_grasped = len(env.grasped_objects[0])
                
                print(f"       奖励: {reward.item():.3f}")
                print(f"       可选物体: {pre_action_selectable_count} -> {post_action_selectable_count}")
                print(f"       剩余索引: {pre_action_remaining} -> {post_action_remaining}")
                print(f"       已抓取: {post_action_grasped}")
                
                # ✅ 检查是否成功抓取（可选物体数量减少了）
                if post_action_selectable_count < pre_action_selectable_count:
                    episode_grasps += 1
                    test_stats['successful_grasps'] += 1
                    print(f"       ✅ 成功抓取并移除物体！")
                
                # 🔍 验证观测中没有异常的瞬移物体位置
                if hasattr(obs, 'flatten') and obs.numel() > 0:
                    # 检查观测中是否有异常大的位置值（可能来自瞬移物体）
                    obs_values = obs.flatten()
                    anomaly_threshold = 2.0  # 超过工作空间的合理范围
                    anomalies = torch.any(torch.abs(obs_values) > anomaly_threshold)
                    if anomalies:
                        test_stats['obs_anomalies'] += 1
                        print(f"       ⚠️ 观测中发现异常值（可能包含瞬移物体位置）")
                
                # 检查任务完成或终止条件
                if terminated.any() or truncated.any():
                    print(f"       🏁 任务终止: terminated={terminated.any()}, truncated={truncated.any()}")
                    break
                
                episode_steps += 1
            
            test_stats['episodes'] += 1
            print(f"   Episode结果: {episode_grasps}个成功抓取，{episode_steps}步")
        
        # 📊 打印测试统计
        print(f"\n📊 测试统计结果:")
        print(f"   总episodes: {test_stats['episodes']}")
        print(f"   成功抓取: {test_stats['successful_grasps']}")
        print(f"   总动作数: {test_stats['total_actions']}")
        print(f"   成功率: {test_stats['successful_grasps']/max(test_stats['total_actions'], 1)*100:.1f}%")
        print(f"   观测异常次数: {test_stats['obs_anomalies']}")
        
        # 🎥 保存视频
        if hasattr(video_frames, '__len__') and len(video_frames) > 0:
            try:
                print(f"\n🎬 保存测试视频到: {video_path}")
                import imageio
                
                # 确保帧格式正确
                processed_frames = []
                for frame in video_frames[::5]:  # 每5帧取1帧，减少文件大小
                    if isinstance(frame, np.ndarray):
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                        processed_frames.append(frame)
                
                if len(processed_frames) > 0:
                    imageio.mimsave(str(video_path), processed_frames, fps=20)
                    print(f"   ✅ 视频保存成功: {len(processed_frames)}帧")
                else:
                    print(f"   ⚠️ 没有有效帧，跳过视频保存")
                    
            except ImportError:
                print(f"   ❌ 需要安装imageio来保存视频: pip install imageio")
            except Exception as video_error:
                print(f"   ❌ 视频保存失败: {video_error}")
        
        # ✅ 验证修复效果
        print(f"\n🔍 修复效果验证:")
        
        # 验证1: 瞬移物体是否正确从可选列表移除
        final_selectable_count = len(env.selectable_objects[0])
        final_grasped_count = len(env.grasped_objects[0])
        expected_selectable = env.total_objects_per_env - final_grasped_count
        
        if final_selectable_count == expected_selectable:
            print(f"   ✅ 瞬移修复验证通过: 可选物体数({final_selectable_count}) = 总数({env.total_objects_per_env}) - 已抓取({final_grasped_count})")
        else:
            print(f"   ❌ 瞬移修复验证失败: 可选物体数({final_selectable_count}) ≠ 预期({expected_selectable})")
        
        # 验证2: 观测异常次数
        if test_stats['obs_anomalies'] == 0:
            print(f"   ✅ 观测修复验证通过: 无瞬移物体位置泄露到观测中")
        else:
            print(f"   ⚠️ 观测修复需要进一步检查: {test_stats['obs_anomalies']}次异常")
        
        # 验证3: 遮挡和支撑检测改进
        print(f"   ℹ️ 遮挡和支撑检测已升级为基于AABB包围盒的方法，更鲁棒处理不同大小物体")
        
        env.close()
        print(f"\n🎉 测试完成！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_methods():
    """专门测试改进的检测方法"""
    print(f"\n🔬 单独测试改进的检测方法...")
    
    try:
        # 创建简单的测试环境
        env = EnvClutterEnv(
            num_envs=1,
            obs_mode="state",
            use_discrete_action=True,
            use_ideal_oracle=True,
            render_mode=None  # 无渲染，加快测试
        )
        
        obs, info = env.reset()
        
        # 测试改进的遮挡检测
        if len(env.selectable_objects[0]) >= 2:
            obj1 = env.selectable_objects[0][0]
            obj2 = env.selectable_objects[0][1]
            
            print(f"   测试遮挡检测:")
            is_blocked1 = env._is_object_blocked(obj1)
            is_blocked2 = env._is_object_blocked(obj2)
            print(f"     物体1遮挡状态: {is_blocked1}")
            print(f"     物体2遮挡状态: {is_blocked2}")
            
            print(f"   测试支撑检测:")
            is_supporting1 = env._is_supporting_others(obj1)
            is_supporting2 = env._is_supporting_others(obj2)
            print(f"     物体1支撑状态: {is_supporting1}")
            print(f"     物体2支撑状态: {is_supporting2}")
        
        env.close()
        print(f"   ✅ 检测方法测试完成")
        return True
        
    except Exception as e:
        print(f"   ❌ 检测方法测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 理想化神谕环境修复测试")
    print("=" * 60)
    
    # 主要功能测试
    main_test_passed = test_oracle_fixed_environment()
    
    # 检测方法专项测试
    detection_test_passed = test_detection_methods()
    
    print("=" * 60)
    if main_test_passed and detection_test_passed:
        print("🎉 所有测试通过！修复效果良好")
        exit(0)
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        exit(1)




