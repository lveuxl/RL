#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import time
import random

from env_clutter import EnvClutterEnv

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def test_continuous_8_grasps():
    
    # 配置测试参数
    num_envs = 1  # 使用单环境便于观察和调试
    capture_video = True
    save_trajectory = False
    test_name = f"continuous_8_grasps_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    
    # 创建环境 - 启用离散动作模式
    print("🏗️ 创建抓取环境...")
    env = EnvClutterEnv(
        render_mode="rgb_array",
        obs_mode="state", 
        control_mode="pd_ee_delta_pose",
        use_discrete_action=True,  # 启用离散动作
        num_envs=num_envs
    )
    
    print(f"✅ 环境创建成功")
    print(f"📦 总物体数量: {env.total_objects_per_env}")
    print(f"🎯 目标: 连续抓取 {min(8, env.total_objects_per_env)} 个物体")
    print()
    
    # 添加视频录制包装器
    if capture_video or save_trajectory:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"🎥 视频录制已启用")
        print(f"📂 视频保存路径: {video_output_dir}")
        
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=save_trajectory,
            save_video=capture_video,
            trajectory_name="continuous_8_grasps",
            max_steps_per_video=5000,  # 足够长以录制完整的8次抓取
            video_fps=30,  # 降低帧率以减少文件大小
            render_substeps=True,
            info_on_video=True,
        )
        print("✓ 视频录制包装器添加成功")
    
    # 添加向量化包装器
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    print("✓ 向量化包装器添加成功")
    print()

    try:
        # 执行连续抓取测试
        total_test_rounds = 1  # 可以增加轮次来测试稳定性
        
        for round_idx in range(total_test_rounds):
            print(f"🎮 === 连续抓取测试轮次 {round_idx + 1}/{total_test_rounds} ===")
            
            # 重置环境
            obs, info = env.reset()
            round_start_time = time.time()
            
            unwrapped_env = env.unwrapped
            
            # 检查环境状态
            print(f"🔧 环境初始状态:")
            if hasattr(unwrapped_env, 'remaining_indices'):
                remaining_objects = len(unwrapped_env.remaining_indices[0])
                print(f"  剩余可抓取物体: {remaining_objects}")
                print(f"  物体索引列表: {unwrapped_env.remaining_indices[0]}")
            
            if hasattr(unwrapped_env, 'selectable_objects'):
                total_objects = len(unwrapped_env.selectable_objects[0])
                print(f"  场景中总物体数: {total_objects}")
            print()
            
            # 连续抓取循环
            target_grasps = min(8, remaining_objects if 'remaining_objects' in locals() else 8)
            successful_grasps = 0
            total_steps = 0
            grasp_times = []
            
            print(f"🚀 开始连续抓取流程 - 目标: {target_grasps} 个物体")
            print("=" * 60)
            
            for grasp_idx in range(target_grasps):
                grasp_start_time = time.time()
                print(f"\n🎯 === 抓取任务 {grasp_idx + 1}/{target_grasps} ===")
                
                # 检查是否还有可抓取的物体
                if not unwrapped_env.remaining_indices[0]:
                    print("⚠️ 没有剩余物体可抓取，提前结束")
                    break
                
                # 智能选择抓取目标
                action = select_optimal_grasp_target(unwrapped_env, grasp_idx)
                if action == -1:
                    print("❌ 无法找到合适的抓取目标，跳过")
                    continue
                
                actual_target_idx = unwrapped_env.remaining_indices[0][action]
                print(f"📍 选择目标: 动作索引={action}, 实际物体索引={actual_target_idx}")
                
                # 执行单次抓取
                grasp_success, grasp_step_count = execute_single_grasp(
                    env, unwrapped_env, action, grasp_idx + 1, target_grasps
                )
                
                grasp_end_time = time.time()
                grasp_duration = grasp_end_time - grasp_start_time
                grasp_times.append(grasp_duration)
                total_steps += grasp_step_count
                
                # 统计结果
                if grasp_success:
                    successful_grasps += 1
                    print(f"✅ 抓取 {grasp_idx + 1} 成功! 用时: {grasp_duration:.2f}秒, 步数: {grasp_step_count}")
                else:
                    print(f"❌ 抓取 {grasp_idx + 1} 失败! 用时: {grasp_duration:.2f}秒, 步数: {grasp_step_count}")
                
                # 打印进度统计
                current_success_rate = (successful_grasps / (grasp_idx + 1)) * 100
                print(f"📊 当前进度: {successful_grasps}/{grasp_idx + 1} 成功率: {current_success_rate:.1f}%")
                
                # 短暂休息让场景稳定
                time.sleep(0.1)
            
            # 轮次结束统计
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            
            print("\n" + "=" * 60)
            print(f"🎊 === 轮次 {round_idx + 1} 完成统计 ===")
            print(f"🎯 目标抓取数: {target_grasps}")
            print(f"✅ 成功抓取数: {successful_grasps}")
            print(f"📊 成功率: {(successful_grasps/target_grasps)*100:.1f}%")
            print(f"⏱️  总用时: {round_duration:.2f}秒")
            print(f"🔄 总步数: {total_steps}")
            print(f"📈 平均每次抓取用时: {np.mean(grasp_times):.2f}±{np.std(grasp_times):.2f}秒")
            print(f"⚡ 平均步长: {round_duration/total_steps:.3f}秒/步")
            
            # 最终环境状态
            final_grasped = len(unwrapped_env.grasped_objects[0])
            final_remaining = len(unwrapped_env.remaining_indices[0])
            print(f"🏆 最终状态: {final_grasped} 个物体已抓取, {final_remaining} 个物体剩余")
            
            if successful_grasps == target_grasps:
                print("🎉 完美完成！所有目标物体都已成功抓取！")
                print("💡 建议: 环境和状态机工作正常，可以开始强化学习训练")
            elif successful_grasps >= target_grasps // 2:
                print("👍 表现良好！大部分物体成功抓取")
                print("💡 建议: 可以调优抓取策略和状态机参数")
            else:
                print("⚠️ 需要改进！成功率较低")
                print("💡 建议: 检查物体选择策略和状态机逻辑")
        
        print(f"\n🏁 === 连续抓取测试完成 ===")
        if capture_video:
            print(f"🎥 视频已保存至: {video_output_dir}")
            print("💡 可以观看视频分析抓取过程和改进点")
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔚 连续抓取测试完成")


def select_optimal_grasp_target(unwrapped_env, grasp_idx):
    """
    智能选择抓取目标
    
    Args:
        unwrapped_env: 解包的环境
        grasp_idx: 当前抓取索引
        
    Returns:
        int: 选择的动作索引，-1表示无法选择
    """
    if not unwrapped_env.remaining_indices[0]:
        return -1
    
    available_count = len(unwrapped_env.remaining_indices[0])
    
    # 策略1: 顺序选择（确保覆盖所有物体）
    if grasp_idx < available_count:
        return 0  # 总是选择第一个可用的物体
    
    # 策略2: 随机选择（当顺序选择超出范围时）
    return random.randint(0, available_count - 1)


def execute_single_grasp(env, unwrapped_env, action, current_grasp, total_grasps):
    """
    执行单次抓取操作
    
    Args:
        env: 环境实例
        unwrapped_env: 解包的环境
        action: 抓取动作
        current_grasp: 当前抓取序号
        total_grasps: 总抓取数
        
    Returns:
        tuple: (是否成功, 步数)
    """
    print(f"🎯 执行抓取动作: {action}")
    
    # 记录抓取前状态
    prev_grasped_count = len(unwrapped_env.grasped_objects[0])
    
    # 执行抓取动作
    step_count = 0
    max_steps = 2000  # 最大步数限制
    
    print(f"🚀 开始状态机执行...")
    
    # 执行动作并等待完成
    obs, reward, terminated, truncated, info = env.step(np.array([action]))
    step_count += 1
    
    # 监控执行过程
    monitor_frequency = 500  # 每500步监控一次
    last_monitor_step = 0
    
    while step_count < max_steps:
        # 检查是否还有FSM在执行
        if hasattr(unwrapped_env, 'env_busy') and not unwrapped_env.env_busy[0]:
            print(f"✅ 状态机执行完成，用时 {step_count} 步")
            break
        
        # 继续执行（对于忙碌的环境，动作会被忽略）
        obs, reward, terminated, truncated, info = env.step(np.array([0]))
        step_count += 1
        
        # 定期监控进度
        if step_count - last_monitor_step >= monitor_frequency:
            if hasattr(unwrapped_env, 'env_stage') and hasattr(unwrapped_env, 'stage_tick'):
                stage = unwrapped_env.env_stage[0].item()
                tick = unwrapped_env.stage_tick[0].item()
                busy = unwrapped_env.env_busy[0].item()
                print(f"📊 步数{step_count}: 状态{stage}, 步数{tick}, {'执行中' if busy else '完成'}")
            last_monitor_step = step_count
        
        # 检查异常终止条件
        if isinstance(terminated, (np.ndarray, torch.Tensor)):
            if hasattr(terminated, 'any') and terminated.any():
                print(f"⚠️ 环境异常终止，步数: {step_count}")
                break
        elif terminated:
            print(f"⚠️ 环境终止，步数: {step_count}")
            break
    
    if step_count >= max_steps:
        print(f"⚠️ 达到最大步数限制 ({max_steps})，强制结束")
    
    # 检查抓取结果
    current_grasped_count = len(unwrapped_env.grasped_objects[0])
    success = current_grasped_count > prev_grasped_count
    
    if success:
        print(f"🎉 物体成功抓取! 累计抓取: {current_grasped_count}")
    else:
        print(f"😞 物体抓取失败! 累计抓取: {current_grasped_count}")
    
    # 显示剩余物体状态
    remaining_count = len(unwrapped_env.remaining_indices[0])
    print(f"📦 剩余物体数: {remaining_count}")
    
    return success, step_count


def test_quick_single_grasp():
    """快速单次抓取测试（调试用）"""
    print("收到，爱学习的小公主！")
    print("=== 快速单次抓取调试 ===")
    
    env = EnvClutterEnv(
        render_mode="rgb_array",
        obs_mode="state", 
        control_mode="pd_ee_delta_pose",
        use_discrete_action=True,
        num_envs=1
    )
    
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    
    try:
        obs, info = env.reset()
        unwrapped_env = env.unwrapped
        
        print("🔧 初始状态检查:")
        print(f"  剩余物体: {len(unwrapped_env.remaining_indices[0])}")
        print(f"  已抓取物体: {len(unwrapped_env.grasped_objects[0])}")
        
        # 执行一次抓取测试
        action = 0
        success, steps = execute_single_grasp(env, unwrapped_env, action, 1, 1)
        
        print(f"🏆 测试结果: {'成功' if success else '失败'}, 用时 {steps} 步")
        
    except Exception as e:
        print(f"❌ 快速测试错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    """主函数 - 选择测试模式"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_quick_single_grasp()
    elif len(sys.argv) > 1 and sys.argv[1] == "help":
        print("收到，爱学习的小公主！")
        print("用法:")
        print("  python continuous_grasp_8_objects.py        # 完整的8次连续抓取测试")
        print("  python continuous_grasp_8_objects.py quick  # 快速单次抓取调试")
        print("  python continuous_grasp_8_objects.py help   # 显示帮助信息")
    else:
        test_continuous_8_grasps()


if __name__ == "__main__":
    main()
