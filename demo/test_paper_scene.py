#!/usr/bin/env python3
"""
论文展示场景测试文件
测试12物体分层堆叠场景，用于论文配图
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym

# 添加demo目录到路径
demo_dir = os.path.dirname(os.path.abspath(__file__))
if demo_dir not in sys.path:
    sys.path.insert(0, demo_dir)

# 导入修改后的环境
sys.path.insert(0, '/home/linux/jzh/RL_Robot/demo')

# 由于文件名带空格，需要特殊处理
import importlib.util
spec = importlib.util.spec_from_file_location(
    "env_clutter_copy", 
    "/home/linux/jzh/RL_Robot/demo/env_clutter copy.py"
)
env_clutter_copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_clutter_copy)
PaperStackingSceneEnv = env_clutter_copy.PaperStackingSceneEnv

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def test_paper_scene(scene_config="balanced", capture_video=True, save_trajectory=False):
    """
    测试论文展示场景
    
    Args:
        scene_config: 场景配置 ("balanced", "challenging", "realistic")
        capture_video: 是否录制视频
        save_trajectory: 是否保存轨迹
    """
    print(f"=== 论文展示场景测试 ===")
    print(f"🎬 场景配置: {scene_config}")
    print(f"📹 录制视频: {capture_video}")
    print(f"💾 保存轨迹: {save_trajectory}")
    print()
    
    # 测试参数
    num_envs = 1  # 单环境展示
    test_name = f"paper_scene_{scene_config}_{int(time.time())}"
    video_output_dir = f"test_videos/{test_name}"
    
    try:
        # 创建环境
        print("🏗️  创建论文展示环境...")
        env = PaperStackingSceneEnv(
            render_mode="rgb_array",
            obs_mode="state",
            control_mode="pd_ee_delta_pose", 
            num_envs=num_envs,
            scene_config=scene_config  # 使用指定的场景配置
        )
        
        print(f"✅ 环境创建成功")
        print()
        
        # 添加视频录制包装器
        if capture_video or save_trajectory:
            os.makedirs(video_output_dir, exist_ok=True)
            print(f"📁 视频输出目录: {video_output_dir}")
            
            env = RecordEpisode(
                env,
                output_dir=video_output_dir,
                save_trajectory=save_trajectory,
                save_video=capture_video,
                trajectory_name=f"paper_scene_{scene_config}",
                max_steps_per_video=300,  # 足够长以观察完整场景
                video_fps=30,  # 降低帧率，更适合论文展示
                render_substeps=False,  # 不渲染子步骤
                info_on_video=True,
            )
            print("✓ 视频录制包装器添加成功")
        
        # 添加向量化包装器
        env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=False)
        print("✓ 向量化包装器添加成功")
        print()
        
        # 测试场景展示
        print("🎬 开始场景展示...")
        
        # 重置环境，创建堆叠场景
        obs, info = env.reset()
        print("✅ 场景重置完成，堆叠结构已创建")
        
        # 获取环境信息
        unwrapped_env = env.unwrapped
        print(f"📊 场景统计:")
        print(f"  - 总物体数量: {len(unwrapped_env.all_objects)}")
        if hasattr(unwrapped_env, 'target_object') and unwrapped_env.target_object:
            print(f"  - 目标物体: {unwrapped_env.target_object.name}")
        print(f"  - 场景配置: {unwrapped_env.scene_config_name}")
        print()
        
        # 静态展示 - 让相机环绕拍摄不同角度
        print("📸 开始多角度静态展示...")
        
        # 定义相机运动路径（环绕拍摄）
        total_steps = 240  # 8秒 @ 30fps
        
        for step in range(total_steps):
            # 执行无动作步骤（机器人保持静止）
            no_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 7维零动作
            
            obs, reward, terminated, truncated, info = env.step(no_action)
            
            # 每30步打印一次进度
            if (step + 1) % 30 == 0:
                progress = (step + 1) / total_steps * 100
                print(f"  📹 拍摄进度: {progress:.1f}% ({step + 1}/{total_steps})")
            
            # 检查是否需要提前结束
            if terminated or truncated:
                print(f"  ℹ️  场景在第 {step + 1} 步结束")
                break
        
        print("✅ 多角度展示完成")
        print()
        
        # 场景分析
        print("📋 场景分析:")
        
        # 分析物体位置和堆叠结构
        if hasattr(unwrapped_env, 'all_objects'):
            print("  🏗️  堆叠结构分析:")
            
            # 按高度排序物体
            object_heights = []
            for obj in unwrapped_env.all_objects:
                pos = obj.pose.p
                if pos.dim() > 1:
                    height = pos[0, 2].item()  # 取第一个环境的z坐标
                else:
                    height = pos[2].item()
                object_heights.append((obj.name, height))
            
            # 按高度排序
            object_heights.sort(key=lambda x: x[1])
            
            # 分层显示
            layers = {
                "底层 (L0)": [],
                "中层 (L1)": [],
                "上层 (L2+)": []
            }
            
            for name, height in object_heights:
                if height < 0.06:
                    layers["底层 (L0)"].append(f"{name} (h={height:.3f}m)")
                elif height < 0.12:
                    layers["中层 (L1)"].append(f"{name} (h={height:.3f}m)")
                else:
                    layers["上层 (L2+)"].append(f"{name} (h={height:.3f}m)")
            
            for layer_name, objects in layers.items():
                if objects:
                    print(f"    {layer_name}:")
                    for obj_info in objects:
                        print(f"      - {obj_info}")
        
        # 视频输出信息
        if capture_video:
            print(f"🎥 视频文件:")
            video_files = []
            for file in os.listdir(video_output_dir):
                if file.endswith('.mp4'):
                    video_path = os.path.join(video_output_dir, file)
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                    video_files.append(f"  - {file} ({file_size:.1f} MB)")
            
            if video_files:
                print("\n".join(video_files))
                print(f"  📁 保存位置: {video_output_dir}")
            else:
                print("  ⚠️  未找到视频文件")
        
        print()
        print("🎉 论文展示场景测试完成！")
        print("💡 建议：")
        print("  1. 查看生成的视频文件用于论文配图")
        print("  2. 尝试不同的scene_config参数观察不同堆叠效果")
        print("  3. 可以调整相机参数获得更好的视觉效果")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("\n🔚 测试完成")


def test_all_scene_configs():
    """测试所有场景配置"""
    configs = ["balanced", "challenging", "realistic"]
    
    print("=== 测试所有场景配置 ===")
    
    for i, config in enumerate(configs):
        print(f"\n🔄 测试配置 {i+1}/{len(configs)}: {config}")
        test_paper_scene(
            scene_config=config,
            capture_video=True,
            save_trajectory=False
        )
        
        if i < len(configs) - 1:
            print("⏳ 等待3秒后继续...")
            time.sleep(3)
    
    print("\n🎉 所有场景配置测试完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="论文展示场景测试")
    parser.add_argument(
        "--config", 
        type=str, 
        default="balanced",
        choices=["balanced", "challenging", "realistic", "all"],
        help="场景配置类型"
    )
    parser.add_argument("--no-video", action="store_true", help="不录制视频")
    parser.add_argument("--save-trajectory", action="store_true", help="保存轨迹数据")
    
    args = parser.parse_args()
    
    if args.config == "all":
        # 测试所有配置
        test_all_scene_configs()
    else:
        # 测试单个配置
        test_paper_scene(
            scene_config=args.config,
            capture_video=not args.no_video,
            save_trajectory=args.save_trajectory
        )


if __name__ == "__main__":
    main()