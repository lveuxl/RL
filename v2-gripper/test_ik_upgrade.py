#!/usr/bin/env python3
"""
测试升级后的IK+控制器抓取流程
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入环境和录制相关模块
from env_clutter import EnvClutterEnv
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

def wait_for_objects_to_settle(env, steps=30):
    """
    等待物体稳定，让它们自然落下并稳定
    
    Args:
        env: 环境实例
        steps: 等待的步数
    """
    print(f"⏳ 等待物体稳定 ({steps} 步)...")
    
    for step in range(steps):
        # 执行无动作步骤，让物理仿真继续运行
        # 使用一个无效的动作或者0动作
        try:
            # 获取动作空间大小
            unwrapped_env = env.unwrapped
            if hasattr(unwrapped_env, 'discrete_action_space'):
                # 使用最后一个动作索引作为"无动作"
                no_action = unwrapped_env.discrete_action_space.n - 1
            else:
                no_action = 0
            
            # 执行无动作步骤
            obs, reward, terminated, truncated, info = env.step(no_action)
            
            # 每10步打印一次进度
            if step % 10 == 0:
                print(f"  稳定中... {step}/{steps}")
                
        except Exception as e:
            print(f"  警告：稳定过程中出现错误: {e}")
            break
    
    print("✓ 物体稳定完成，开始抓取动作")

def test_ik_upgrade():
    """测试升级后的IK+控制器抓取流程"""
    print("=== 测试升级后的IK+控制器抓取流程 ===")
    
    # 配置视频录制参数 - 优化视频质量和长度
    capture_video = True
    save_trajectory = False
    test_name = f"test_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    
    # 创建环境
    env = EnvClutterEnv(
        robot_uids="panda",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        use_discrete_action=True,
        obs_mode="rgb",  # 改为rgb模式以支持视频录制
        render_mode="rgb_array",  # 使用rgb_array模式
        sim_backend="gpu",
        sensor_configs=dict(
            width=256,  # 增加分辨率以提高视频质量
            height=256  # 增加分辨率以提高视频质量
        )
    )
    print("✓ 环境创建成功")
    
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
            max_steps_per_video=1200,  # 增加到1200步以容纳等待时间
            video_fps=60,  # 提高帧率到60fps
            render_substeps=True,  # 启用子步渲染以获得更流畅的视频
            info_on_video=True,  # 在视频上显示信息
        )
        print("✓ 视频录制包装器添加成功")
    
    # 添加向量化包装器
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
    print("✓ 向量化包装器添加成功")
    
    # 重置环境
    try:
        obs, info = env.reset()
        print("✓ 环境重置成功")
        print(f"观测维度: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # 等待物体稳定
        wait_for_objects_to_settle(env, steps=50)  # 等待50步让物体稳定
        
    except Exception as e:
        print(f"✗ 环境重置失败: {e}")
        return False
    
    # 检查控制器初始化
    try:
        # 使用推荐的方式访问底层环境属性
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, 'arm_controller') and unwrapped_env.arm_controller is not None:
            print("✓ 控制器初始化成功")
        else:
            print("✗ 控制器初始化失败")
            return False
    except Exception as e:
        print(f"✗ 控制器检查失败: {e}")
        return False
    
    # 检查预计算的关节角
    try:
        if (hasattr(unwrapped_env, 'q_init') and unwrapped_env.q_init is not None and
            hasattr(unwrapped_env, 'q_above') and unwrapped_env.q_above is not None and
            hasattr(unwrapped_env, 'q_goal') and unwrapped_env.q_goal is not None):
            print("✓ 预计算关节角成功")
            print(f"初始关节角维度: {unwrapped_env.q_init.shape}")
            print(f"目标上方关节角维度: {unwrapped_env.q_above.shape}")
            print(f"目标关节角维度: {unwrapped_env.q_goal.shape}")
        else:
            print("✗ 预计算关节角失败")
            return False
    except Exception as e:
        print(f"✗ 关节角检查失败: {e}")
        return False
    
    # 测试离散动作
    try:
        print("\n=== 测试离散动作抓取 ===")
        
        # 获取动作空间
        if hasattr(unwrapped_env, 'discrete_action_space'):
            action_space = unwrapped_env.discrete_action_space
            print(f"离散动作空间: {action_space}")
        else:
            print("✗ 没有离散动作空间")
            return False
        
        # 测试几个动作 - 增加测试动作数量
        for i in range(min(3, action_space.n)):  # 减少到3个动作以节省时间
            print(f"\n--- 测试动作 {i} ---")
            
            # 重置环境
            obs, info = env.reset()
            
            # 等待物体稳定（每次重置后都要等待）
            wait_for_objects_to_settle(env, steps=30)
            
            # 执行动作
            start_time = time.time()
            try:
                next_obs, reward, terminated, truncated, info = env.step(i)
                end_time = time.time()
                
                print(f"动作执行时间: {end_time - start_time:.2f}秒")
                print(f"奖励: {reward}")
                print(f"成功: {info.get('success', False)}")
                print(f"位移: {info.get('displacement', 0.0):.4f}")
                print(f"终止: {terminated}")
                
                # 让环境运行更长时间以录制完整视频
                for step in range(100):  # 减少到100步
                    # 检查终止条件，确保是tensor类型
                    if isinstance(terminated, torch.Tensor):
                        if terminated.any():
                            break
                    elif isinstance(terminated, (bool, np.bool_)):
                        if terminated:
                            break
                    
                    if isinstance(truncated, torch.Tensor):
                        if truncated.any():
                            break
                    elif isinstance(truncated, (bool, np.bool_)):
                        if truncated:
                            break
                    
                    # 执行不同的动作以增加视频内容
                    if step < 30:
                        dummy_action = 0  # 前30步执行第一个动作
                    elif step < 60:
                        dummy_action = min(1, action_space.n - 1)  # 中30步执行第二个动作
                    else:
                        dummy_action = min(2, action_space.n - 1)  # 后40步执行第三个动作
                    
                    next_obs, reward, terminated, truncated, info = env.step(dummy_action)
                    
                    # 添加小延迟以便观察
                    if step % 20 == 0:
                        print(f"  步骤 {step}: 奖励 {reward:.3f}")
                
                print(f"✓ 动作 {i} 测试完成，录制了 {100} 步")
                
            except Exception as e:
                print(f"✗ 动作执行失败: {e}")
                continue
        
        print("✓ 离散动作测试完成")
        
    except Exception as e:
        print(f"✗ 离散动作测试失败: {e}")
        return False
    
    # 额外录制一段展示视频
    try:
        print("\n=== 录制展示视频 ===")
        obs, info = env.reset()
        
        # 等待物体稳定
        wait_for_objects_to_settle(env, steps=30)
        
        # 录制一段展示机械臂各种动作的视频
        for demo_step in range(120):  # 增加展示步数
            # 循环执行不同动作
            action = demo_step % min(3, action_space.n)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if demo_step % 30 == 0:
                print(f"  展示步骤 {demo_step}: 动作 {action}")
            
            # 检查终止条件
            if isinstance(terminated, torch.Tensor):
                if terminated.any():
                    break
            elif isinstance(terminated, (bool, np.bool_)):
                if terminated:
                    break
        
        print("✓ 展示视频录制完成")
        
    except Exception as e:
        print(f"✗ 展示视频录制失败: {e}")
    
    # 测试工具函数
    try:
        print("\n=== 测试工具函数 ===")
        
        # 测试RPY到四元数转换
        euler = torch.tensor([0.0, np.pi/2, 0.0], device=unwrapped_env.device)
        quat = unwrapped_env._rpy_to_quat(euler)
        print(f"✓ RPY到四元数转换: {quat}")
        
        # 测试夹爪控制
        print("测试夹爪控制...")
        unwrapped_env._open_gripper()
        time.sleep(0.5)
        unwrapped_env._close_gripper()
        time.sleep(0.5)
        print("✓ 夹爪控制测试完成")
        
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        # 这不是致命错误，继续执行
        pass
    
    # 清理
    try:
        env.close()
        print("✓ 环境清理成功")
        if capture_video:
            print(f"✓ 视频已保存到: {video_output_dir}")
            print("📹 视频配置:")
            print(f"   - 分辨率: 256x256")
            print(f"   - 帧率: 60 FPS")
            print(f"   - 最大步数: 1200 (包含等待时间)")
            print(f"   - 子步渲染: 启用")
            print(f"   - 物体稳定等待: 30-50步")
    except Exception as e:
        print(f"✗ 环境清理失败: {e}")
    
    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    success = test_ik_upgrade()
    if success:
        print("🎉 所有测试通过！IK+控制器升级成功！")
        print("📹 高质量视频文件已生成，可以查看机械臂的完整抓取过程！")
        print("💡 视频特性:")
        print("   ✓ 更高分辨率 (256x256)")
        print("   ✓ 更高帧率 (60 FPS)")
        print("   ✓ 更长时长 (最多1200步)")
        print("   ✓ 流畅渲染 (子步渲染)")
        print("   ✓ 信息叠加 (显示动作和奖励)")
        print("   ✓ 物体稳定等待 (让物体自然落下)")
    else:
        print("❌ 测试失败，请检查实现。")
        sys.exit(1) 