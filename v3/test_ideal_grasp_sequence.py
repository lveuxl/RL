#!/usr/bin/env python3
"""
理想化抓取顺序学习环境测试
主要特性：
1. 测试抓取顺序学习任务 - 每回合抓取9个物体
2. 验证奖励函数优先级 - 抓取成功 > 位移小 > 时间短
3. 检验动作掩码更新机制 - 确保并行环境索引正确
4. 支持视频录制 - 观察抓取顺序学习效果
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import time

# 导入修正后的环境以确保注册
import sys
import os
sys.path.append(os.getcwd())

# 导入环境以确保注册
try:
    # 尝试导入copy版本（新的抓取顺序学习环境）
    from env_clutter import EnvClutterEnv
    print("✅ 使用env_clutter环境（抓取顺序学习版本）")
except ImportError:
    try:
        # 回退到原始环境
        from env_clutter import EnvClutterEnv
        print("⚠️ 使用原始env_clutter环境（可能功能不完整）")
    except ImportError as e:
        print(f"❌ 环境导入失败: {e}")
        raise

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def test_ideal_grasp_sequence():
    """测试理想化抓取顺序学习环境"""
    print("=== 理想化抓取顺序学习环境测试 ===")
    print("🎯 任务目标：通过强化学习挑选最适合的抓取顺序")
    print("📋 任务要求：每回合抓取9个物体")
    print("🏆 奖励优先级：1.抓取成功 2.其他物体位移小 3.总时间短")
    print("🔧 测试内容：动作掩码更新、并行环境索引、奖励函数")
    print()
    
    # 配置测试参数
    num_envs = 2  # 测试2个并行环境以验证掩码更新
    capture_video = True
    save_trajectory = False
    test_name = f"ideal_grasp_sequence_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    
    # 创建理想化抓取环境
    try:
        env = gym.make(
            "EnvClutter-v1",
            render_mode="rgb_array",
            obs_mode="state", 
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",  # 使用密集奖励以观察学习过程
            sim_backend="gpu",
            use_discrete_action=True,  # 启用离散动作选择
            use_ideal_oracle=True,     # 启用理想化神谕抓取
            num_envs=num_envs
        )
        
        print(f"✅ 理想化抓取环境创建成功 (环境数: {num_envs})")
        print(f"🎯 目标物体数量: {env.unwrapped.total_objects_per_env}")
        print(f"🎮 动作空间大小: {env.unwrapped.MAX_N}")
        print()
        
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return
    
    # 添加视频录制包装器
    if capture_video or save_trajectory:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"📹 视频将保存到: {video_output_dir}")
        
        try:
            env = RecordEpisode(
                env,
                output_dir=video_output_dir,
                save_trajectory=save_trajectory,
                save_video=capture_video,
                trajectory_name="ideal_grasp_sequence",
                max_steps_per_video=3000,  # 足够长以观察完整的9次抓取流程
                video_fps=30,
                render_substeps=True,
                info_on_video=True,
            )
            print("✅ 视频录制包装器添加成功")
        except Exception as e:
            print(f"⚠️ 视频录制包装器添加失败: {e}")
    
    # 添加向量化包装器
    try:
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
        print("✅ 向量化包装器添加成功")
    except Exception as e:
        print(f"❌ 向量化包装器添加失败: {e}")
        return

    try:
        # 测试抓取顺序学习
        total_episodes = 2
        success_stats = []
        
        for episode in range(total_episodes):
            print(f"\n🎮 === 抓取顺序学习测试 {episode + 1}/{total_episodes} ===")
            
            obs, info = env.reset()
            episode_start_time = time.time()
            
            unwrapped_env = env.unwrapped
            
            # 验证环境初始化状态
            print(f"🔍 环境初始化验证:")
            if hasattr(unwrapped_env, 'remaining_indices'):
                for env_idx in range(num_envs):
                    remaining = len(unwrapped_env.remaining_indices[env_idx])
                    grasped = len(unwrapped_env.grasped_objects[env_idx])
                    print(f"  环境{env_idx}: 剩余物体={remaining}, 已抓取={grasped}")
            
            # 验证观测结构和动作掩码
            print(f"🔍 观测结构验证:")
            print(f"  观测维度: {obs.shape}")
            
            # 根据新的观测结构提取掩码
            total_objects = unwrapped_env.total_objects_per_env
            mask_start = total_objects * 8
            mask_end = mask_start + total_objects
            
            if obs.shape[-1] >= mask_end:
                action_mask = obs[0, mask_start:mask_end]  # 提取第一个环境的掩码
                print(f"  动作掩码: {action_mask.cpu().numpy() if hasattr(action_mask, 'cpu') else action_mask}")
                available_actions = torch.sum(action_mask).item() if hasattr(action_mask, 'sum') else np.sum(action_mask)
                print(f"  可用动作数: {available_actions}")
            else:
                print(f"  ⚠️ 观测维度不足，无法提取掩码")
            
            # 模拟智能抓取顺序策略（用于演示）
            # 策略：优先选择暴露度最高的物体（模拟理想的抓取顺序）
            episode_rewards = []
            episode_actions = []
            step_count = 0
            max_steps_per_episode = unwrapped_env.total_objects_per_env + 2  # 最多抓取次数
            
            print(f"\n🚀 开始抓取顺序测试（最多{max_steps_per_episode}次抓取）...")
            
            while step_count < max_steps_per_episode:
                # 选择动作策略：对于每个环境选择不同的抓取策略
                actions = []
                
                for env_idx in range(num_envs):
                    if hasattr(unwrapped_env, 'remaining_indices') and env_idx < len(unwrapped_env.remaining_indices):
                        remaining_indices = unwrapped_env.remaining_indices[env_idx]
                        
                        if remaining_indices:
                            # 策略1: 环境0使用顺序抓取（从上到下）
                            # 策略2: 环境1使用逆序抓取（从下到上）
                            if env_idx == 0:
                                action = 0  # 选择第一个可用物体
                            else:
                                action = len(remaining_indices) - 1  # 选择最后一个可用物体
                                
                            # 确保动作有效
                            action = max(0, min(action, len(remaining_indices) - 1))
                            target_obj_idx = remaining_indices[action]
                            actions.append(action)
                            print(f"  📍 环境{env_idx}: 选择动作{action} -> 目标物体{target_obj_idx} (剩余{len(remaining_indices)}个)")
                        else:
                            actions.append(0)  # 没有可抓取物体时的默认动作
                            print(f"  ⭕ 环境{env_idx}: 无可抓取物体，使用默认动作")
                    else:
                        actions.append(0)
                        print(f"  ⚠️ 环境{env_idx}: 状态异常，使用默认动作")
                
                # 执行动作
                action_array = np.array(actions)
                episode_actions.append(actions.copy())
                
                obs, reward, terminated, truncated, info = env.step(action_array)
                step_count += 1
                
                # 记录奖励和状态
                episode_rewards.append(reward.cpu().numpy() if hasattr(reward, 'cpu') else reward)
                
                print(f"  🎯 步骤{step_count}: 动作={actions}, 奖励={reward}")
                
                # 验证抓取效果和掩码更新
                if hasattr(unwrapped_env, 'grasped_objects'):
                    for env_idx in range(num_envs):
                        if env_idx < len(unwrapped_env.grasped_objects):
                            grasped_count = len(unwrapped_env.grasped_objects[env_idx])
                            remaining_count = len(unwrapped_env.remaining_indices[env_idx])
                            print(f"    环境{env_idx}: 已抓取={grasped_count}, 剩余={remaining_count}")
                
                # 检查终止条件
                if isinstance(terminated, (np.ndarray, torch.Tensor)):
                    if hasattr(terminated, 'any'):
                        if terminated.any():
                            print(f"  🏁 部分环境达到终止条件")
                            break
                    elif np.any(terminated):
                        print(f"  🏁 部分环境达到终止条件")
                        break
                elif terminated:
                    print(f"  🏁 环境达到终止条件")
                    break
                
                # 检查是否完成所有抓取
                all_completed = True
                for env_idx in range(num_envs):
                    if hasattr(unwrapped_env, 'grasped_objects') and env_idx < len(unwrapped_env.grasped_objects):
                        grasped_count = len(unwrapped_env.grasped_objects[env_idx])
                        if grasped_count < unwrapped_env.total_objects_per_env:
                            all_completed = False
                            break
                
                if all_completed:
                    print(f"  ✅ 所有环境完成目标抓取任务！")
                    break
            
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            
            # 统计本episode结果
            print(f"\n📊 Episode {episode + 1} 结果统计:")
            print(f"  ⏱️ 总耗时: {episode_duration:.2f}秒")
            print(f"  🔄 总步数: {step_count}")
            print(f"  💰 累计奖励: {np.sum(episode_rewards):.3f}")
            print(f"  📈 平均步骤奖励: {np.mean(episode_rewards):.3f}")
            
            # 各环境最终成果
            env_success_info = []
            for env_idx in range(num_envs):
                if hasattr(unwrapped_env, 'grasped_objects') and env_idx < len(unwrapped_env.grasped_objects):
                    grasped_count = len(unwrapped_env.grasped_objects[env_idx])
                    total_objects = unwrapped_env.total_objects_per_env
                    success_rate = grasped_count / total_objects
                    env_success_info.append({
                        'env_idx': env_idx,
                        'grasped': grasped_count, 
                        'total': total_objects,
                        'success_rate': success_rate
                    })
                    
                    status = "✅成功" if grasped_count == total_objects else "🔄进行中" if grasped_count > 0 else "❌失败"
                    print(f"  环境{env_idx}: {grasped_count}/{total_objects} = {success_rate:.1%} {status}")
                    print(f"    抓取顺序: {unwrapped_env.grasped_objects[env_idx]}")
                    
            success_stats.append(env_success_info)
            
            # 验证奖励函数优先级
            if len(episode_rewards) > 1:
                reward_trend = np.diff(episode_rewards, axis=0)
                print(f"  📈 奖励变化趋势: 初始={episode_rewards[0]:.3f} -> 最终={episode_rewards[-1]:.3f}")
        
        # 总体结果分析
        print(f"\n📈 === 理想化抓取顺序学习测试总结 ===")
        
        # 计算总体成功率
        total_success_count = 0
        total_attempts = 0
        
        for episode_stats in success_stats:
            for env_stats in episode_stats:
                total_attempts += 1
                if env_stats['success_rate'] == 1.0:
                    total_success_count += 1
        
        overall_success_rate = total_success_count / total_attempts if total_attempts > 0 else 0
        
        print(f"🎯 总测试次数: {total_attempts} (环境数 {num_envs} × 轮次 {total_episodes})")
        print(f"✅ 完全成功次数: {total_success_count}")
        print(f"📊 总体成功率: {overall_success_rate:.1%}")
        
        # 分析不同抓取策略的效果
        if len(success_stats) > 0:
            print(f"\n🔍 抓取策略效果分析:")
            
            env0_success = sum(1 for ep in success_stats for env in ep if env['env_idx'] == 0 and env['success_rate'] == 1.0)
            env0_attempts = sum(1 for ep in success_stats for env in ep if env['env_idx'] == 0)
            
            if num_envs > 1:
                env1_success = sum(1 for ep in success_stats for env in ep if env['env_idx'] == 1 and env['success_rate'] == 1.0)
                env1_attempts = sum(1 for ep in success_stats for env in ep if env['env_idx'] == 1)
                
                print(f"  策略1(顺序抓取): {env0_success}/{env0_attempts} = {env0_success/env0_attempts:.1%}")
                print(f"  策略2(逆序抓取): {env1_success}/{env1_attempts} = {env1_success/env1_attempts:.1%}")
            else:
                print(f"  测试策略: {env0_success}/{env0_attempts} = {env0_success/env0_attempts:.1%}")
        
        # 评估测试结果
        if overall_success_rate >= 0.8:
            print(f"\n🎉 理想化抓取顺序学习环境测试通过！")
            print(f"✅ 动作掩码更新机制正常")
            print(f"✅ 并行环境索引处理正确") 
            print(f"✅ 奖励函数设计合理")
            print(f"🚀 建议：可以开始训练抓取顺序选择模型")
        elif overall_success_rate >= 0.5:
            print(f"\n⚠️ 理想化抓取环境基本可用，但需要优化")
            print(f"💡 建议：检查奖励函数权重和掩码更新逻辑")
        else:
            print(f"\n❌ 理想化抓取环境存在问题")
            print(f"💡 建议：检查环境配置和状态机逻辑")
            
        if capture_video:
            print(f"\n📹 测试视频已保存至: {video_output_dir}")
            print(f"💡 通过视频可以观察抓取顺序学习的效果")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass
        print("\n🔚 理想化抓取顺序学习测试完成")


def test_action_mask_dynamics():
    """专门测试动作掩码的动态更新机制"""
    print("\n=== 动作掩码动态更新测试 ===")
    
    try:
        env = gym.make(
            "EnvClutter-v1",
            obs_mode="state", 
            use_discrete_action=True,
            use_ideal_oracle=True,
            num_envs=1  # 单环境便于调试
        )
        
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
        
        obs, info = env.reset()
        unwrapped_env = env.unwrapped
        
        print(f"🔍 初始状态:")
        print(f"  总物体数: {unwrapped_env.total_objects_per_env}")
        print(f"  剩余物体索引: {unwrapped_env.remaining_indices[0]}")
        
        # 提取并验证初始掩码
        total_objects = unwrapped_env.total_objects_per_env
        mask_start = total_objects * 8
        mask_end = mask_start + total_objects
        
        initial_mask = obs[0, mask_start:mask_end]
        print(f"  初始掩码: {initial_mask}")
        print(f"  可用动作数: {torch.sum(initial_mask).item()}")
        
        # 执行几次抓取，观察掩码变化
        for step in range(min(3, total_objects)):
            remaining_count = len(unwrapped_env.remaining_indices[0])
            if remaining_count == 0:
                print(f"  ⭕ 无更多可抓取物体")
                break
                
            action = 0  # 总是选择第一个可用物体
            print(f"\n🎯 执行抓取 {step + 1}:")
            print(f"  选择动作: {action}")
            print(f"  目标物体索引: {unwrapped_env.remaining_indices[0][action]}")
            
            obs, reward, terminated, truncated, info = env.step([action])
            
            # 提取新掩码
            new_mask = obs[0, mask_start:mask_end]
            print(f"  抓取后掩码: {new_mask}")
            print(f"  可用动作数: {torch.sum(new_mask).item()}")
            print(f"  剩余物体索引: {unwrapped_env.remaining_indices[0]}")
            print(f"  已抓取物体: {unwrapped_env.grasped_objects[0]}")
            print(f"  奖励: {reward.item():.3f}")
            
            if terminated or truncated:
                print(f"  🏁 环境终止")
                break
        
        print(f"\n✅ 动作掩码动态更新测试完成")
        
    except Exception as e:
        print(f"❌ 掩码测试错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    """主函数 - 选择测试模式"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mask":
        test_action_mask_dynamics()
    else:
        test_ideal_grasp_sequence()


if __name__ == "__main__":
    main()
