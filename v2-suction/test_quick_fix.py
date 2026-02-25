#!/usr/bin/env python3
"""
快速测试并行状态机版本的8状态抓取功能
主要特性：
1. 并行有限状态机 - 真正的多环境并行执行
2. 状态机逐步推进 - 避免训练停滞
3. FSM状态监控 - 实时查看状态机执行情况
4. 多环境测试 - 验证并行执行效果
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


def test_parallel_fsm():
    """测试并行状态机版本"""
    print("=== 测试并行状态机版本的8状态抓取 ===")
    print("🔧 新特性：")
    print("1. 并行有限状态机 - 多环境同步推进")
    print("2. 状态逐步执行 - 每step只执行一个状态片段")
    print("3. FSM状态监控 - 实时观察状态转换")
    print("4. 真正并行训练 - 解决训练停滞问题")
    print()
    
    # 配置测试参数
    num_envs = 4  # 测试4个并行环境
    capture_video = True
    save_trajectory = False
    test_name = f"test_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    

    # 创建环境 - 启用并行状态机
    env = EnvClutterEnv(
        render_mode="rgb_array",
        obs_mode="state", 
        control_mode="pd_ee_delta_pose",
        use_discrete_action=True,  # 启用离散动作
        num_envs=num_envs  # 多环境并行
    )
    
    print(f"✅ 并行环境创建成功 (环境数: {num_envs})")
    print(f"🎯 FSM状态: 0-7 (上升->下降->抓取->提升->移动->下降->放下->回归)")
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
            trajectory_name="parallel_fsm_trajectory",
            max_steps_per_video=2000,  # 足够长以观察完整流程
            video_fps=60,
            render_substeps=True,
            info_on_video=True,
        )
        print("✓ 视频录制包装器添加成功")
    
    # 添加向量化包装器
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
    print("✓ 向量化包装器添加成功")

    try:
        # 测试并行状态机
        total_episodes = 2
        success_counts = [0] * num_envs
        
        for episode in range(total_episodes):
            print(f"\n🎮 === 并行状态机测试 {episode + 1}/{total_episodes} ===")
            
            obs, info = env.reset()
            episode_start_time = time.time()
            
            unwrapped_env = env.unwrapped
            
            # 检查FSM状态初始化
            if hasattr(unwrapped_env, 'env_stage'):
                print(f"🔧 FSM状态初始化:")
                print(f"  env_stage: {unwrapped_env.env_stage}")
                print(f"  env_busy: {unwrapped_env.env_busy}")
                print(f"  env_target: {unwrapped_env.env_target}")
            
            # 为每个环境选择不同的目标物体
            actions = []
            for env_idx in range(num_envs):
                if hasattr(unwrapped_env, 'remaining_indices') and env_idx < len(unwrapped_env.remaining_indices):
                    if unwrapped_env.remaining_indices[env_idx]:
                        # 选择该环境的第一个可用物体
                        action_idx = min(env_idx, len(unwrapped_env.remaining_indices[env_idx]) - 1)
                        target_obj_idx = unwrapped_env.remaining_indices[env_idx][action_idx]
                        actions.append(action_idx)
                        print(f"🎯 环境{env_idx}: 选择抓取物体索引 {target_obj_idx} (动作: {action_idx})")
                    else:
                        actions.append(0)  # 默认动作
                        print(f"⚠️ 环境{env_idx}: 没有可抓取物体，使用默认动作")
                else:
                    actions.append(0)
                    print(f"⚠️ 环境{env_idx}: 状态未初始化，使用默认动作")
            
            # 执行并行状态机测试
            max_steps = 500  # 最大步数，足够完成一次抓取
            step_count = 0
            
            print(f"\n🚀 开始并行状态机执行...")
            
            while step_count < max_steps:
                # 执行动作（对于忙碌的环境，动作会被忽略；对于空闲环境，会启动新的抓取流程）
                action_array = np.array(actions)
                obs, reward, terminated, truncated, info = env.step(action_array)
                
                step_count += 1
                
                # 监控FSM状态（每10步打印一次）
                if step_count % 10 == 0 and hasattr(unwrapped_env, 'env_stage'):
                    print(f"📊 步数{step_count}: ", end="")
                    for env_idx in range(num_envs):
                        stage = unwrapped_env.env_stage[env_idx].item()
                        busy = unwrapped_env.env_busy[env_idx].item()
                        tick = unwrapped_env.stage_tick[env_idx].item()
                        print(f"环境{env_idx}[状态{stage},{'忙' if busy else '闲'},步{tick}] ", end="")
                    print()
                
                # 检查是否有环境完成抓取
                completed_envs = []
                if hasattr(info, 'get') and 'success' in info:
                    success_tensor = info.get('success')
                    if isinstance(success_tensor, torch.Tensor):
                        # 多环境情况下，success是张量
                        for env_idx in range(num_envs):
                            if env_idx < len(success_tensor) and success_tensor[env_idx].item():
                                completed_envs.append(env_idx)
                    else:
                        # 单环境情况下，success是标量
                        if success_tensor:
                            completed_envs.append(0)
                
                if completed_envs:
                    print(f"🎉 环境 {completed_envs} 完成抓取!")
                
                # 检查终止条件
                if isinstance(terminated, (np.ndarray, torch.Tensor)):
                    if hasattr(terminated, 'any'):
                        if terminated.any():
                            print(f"📋 部分环境终止: {terminated}")
                            break
                    else:
                        # numpy数组情况
                        if np.any(terminated):
                            print(f"📋 部分环境终止: {terminated}")
                            break
                elif terminated:
                    print(f"📋 环境终止")
                    break
                
                # 如果所有环境都空闲，结束测试
                if hasattr(unwrapped_env, 'env_busy'):
                    if not unwrapped_env.env_busy.any():
                        print(f"✅ 所有环境完成任务，提前结束")
                        break
            
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            
            # 统计结果
            print(f"\n📊 并行测试 {episode + 1} 结果:")
            print(f"  ⏱️  总耗时: {episode_duration:.2f}秒")
            print(f"  🔄 总步数: {step_count}")
            print(f"  📈 平均步长: {episode_duration/step_count:.3f}秒/步")
            
            # 各环境成功率统计
            for env_idx in range(num_envs):
                if hasattr(unwrapped_env, 'grasped_objects') and env_idx < len(unwrapped_env.grasped_objects):
                    grasped_count = len(unwrapped_env.grasped_objects[env_idx])
                    if grasped_count > 0:
                        success_counts[env_idx] += 1
                        print(f"  ✅ 环境{env_idx}: 成功抓取 {grasped_count} 个物体")
                    else:
                        print(f"  ❌ 环境{env_idx}: 未成功抓取物体")
                else:
                    print(f"  ⚠️ 环境{env_idx}: 状态异常")
            
            # FSM最终状态
            if hasattr(unwrapped_env, 'env_stage'):
                print(f"  🔧 FSM最终状态: {unwrapped_env.env_stage.tolist()}")
                print(f"  🔧 FSM忙碌状态: {unwrapped_env.env_busy.tolist()}")
        
        # 总体结果统计
        total_success = sum(success_counts)
        total_attempts = total_episodes * num_envs
        overall_success_rate = total_success / total_attempts * 100
        
        print(f"\n📈 === 并行状态机测试结果 ===")
        print(f"🔢 总测试次数: {total_attempts} (环境数 {num_envs} × 轮次 {total_episodes})")
        print(f"✅ 总成功次数: {total_success}")
        print(f"📊 总体成功率: {overall_success_rate:.1f}%")
        
        # 各环境成功率
        print(f"📊 各环境成功率:")
        for env_idx in range(num_envs):
            env_success_rate = success_counts[env_idx] / total_episodes * 100
            print(f"  环境{env_idx}: {success_counts[env_idx]}/{total_episodes} = {env_success_rate:.1f}%")
        
        if overall_success_rate > 0:
            print(f"🎉 并行状态机工作正常！")
            print(f"💡 训练停滞问题已解决 - 各环境可以并行推进")
            print(f"💡 建议：可以开始大规模并行训练")
        else:
            print(f"⚠️ 并行状态机需要进一步调试")
            print(f"💡 建议：检查状态机逻辑和环境同步")
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔚 并行状态机测试完成")


def test_single_env_fsm():
    """测试单环境状态机以便调试"""
    print("=== 单环境FSM调试测试 ===")
    
    env = EnvClutterEnv(
        render_mode="rgb_array",
        obs_mode="state", 
        control_mode="pd_ee_delta_pose",
        use_discrete_action=True,
        num_envs=1  # 单环境
    )
    
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
    
    try:
        obs, info = env.reset()
        unwrapped_env = env.unwrapped
        
        print("🔧 初始FSM状态:")
        if hasattr(unwrapped_env, 'env_stage'):
            print(f"  env_stage: {unwrapped_env.env_stage}")
            print(f"  env_busy: {unwrapped_env.env_busy}")
            print(f"  env_target: {unwrapped_env.env_target}")
            print(f"  remaining_indices: {unwrapped_env.remaining_indices}")
        
        # 选择第一个物体进行抓取
        action = 0
        print(f"🎯 执行动作: {action}")
        
        # 执行几步观察状态机变化
        for step in range(20):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if hasattr(unwrapped_env, 'env_stage'):
                stage = unwrapped_env.env_stage[0].item()
                busy = unwrapped_env.env_busy[0].item()
                tick = unwrapped_env.stage_tick[0].item()
                target = unwrapped_env.env_target[0].item()
                
                print(f"步{step+1}: 状态{stage}, {'忙碌' if busy else '空闲'}, 步数{tick}, 目标{target}, 奖励{reward.item():.3f}")
                
                if not busy:
                    print("✅ 状态机完成或空闲")
                    break
            else:
                print(f"步{step+1}: FSM状态不可用")
        
    except Exception as e:
        print(f"❌ 单环境测试错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    """主函数 - 选择测试模式"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        test_single_env_fsm()
    else:
        test_parallel_fsm()


if __name__ == "__main__":
    main() 