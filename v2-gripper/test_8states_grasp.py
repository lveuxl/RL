#!/usr/bin/env python3
"""
测试8状态抓取功能的脚本
使用pd_ee_delta_pose控制模式和super().step()实现类似PyBullet的抓取流程
"""
import os
import gymnasium as gym
import numpy as np
import sys
import torch
from env_clutter3 import EnvClutterEnv
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

def test_8states_grasp():
    """测试8状态抓取功能"""
    print("开始测试8状态抓取功能...")
    
    # 配置视频录制参数 - 优化视频质量和长度
    capture_video = True
    save_trajectory = False
    test_name = f"test_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    

    # 创建环境
    env = EnvClutterEnv(
        robot_uids="panda",
        #render_mode="human",
        render_mode="rgb_array",
        obs_mode="rgb", 
        control_mode="pd_ee_delta_pose",  # 使用pd_ee_delta_pose控制模式
        use_discrete_action=True,  # 启用离散动作
        sim_backend="gpu",
        num_envs=1,
        sensor_configs=dict(
            width=256,  # 增加分辨率以提高视频质量
            height=256  # 增加分辨率以提高视频质量
        )
    )
    
    print(f"环境创建成功")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    print(f"控制模式: pd_ee_delta_pose")
    
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
    
    try:
        # 重置环境
        obs, info = env.reset()
        print(f"环境重置成功，观测维度: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # 执行几次抓取测试
        for episode in range(3):
            print(f"\n=== Episode {episode + 1} ===")
            
            obs, info = env.reset()
            done = False
            step_count = 0
            unwrapped_env = env.unwrapped

            while not done and step_count < 5:  # 最多测试5步
                # 选择一个随机的有效动作
                if hasattr(unwrapped_env, 'remaining_indices') and unwrapped_env.remaining_indices:
                    # 从剩余物体中随机选择一个
                    action_idx = np.random.randint(0, len(unwrapped_env.remaining_indices))
                    target_obj_idx = unwrapped_env.remaining_indices[action_idx]
                    
                    print(f"\nStep {step_count + 1}: 选择抓取物体索引 {target_obj_idx}")
                    print(f"剩余可抓取物体: {len(unwrapped_env.remaining_indices)} 个")
                    
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action_idx)
                    
                    print(f"执行结果:")
                    print(f"  - 奖励: {reward.item():.3f}")
                    print(f"  - 成功: {info.get('success', False)}")
                    print(f"  - 位移: {info.get('displacement', 0):.3f}")
                    print(f"  - 剩余物体: {info.get('remaining_objects', 0)}")
                    print(f"  - 已抓取物体: {info.get('grasped_objects', 0)}")
                    
                    done = terminated or truncated
                    step_count += 1
                    
                    # 如果成功，暂停一下让我们看到结果
                    if info.get('success', False):
                        print("✅ 抓取成功！")
                        input("按回车继续...")
                    else:
                        print("❌ 抓取失败")
                else:
                    print("没有剩余可抓取的物体")
                    break
            
            print(f"Episode {episode + 1} 完成，共执行 {step_count} 步")
    
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("环境已关闭")

if __name__ == "__main__":
    test_8states_grasp() 