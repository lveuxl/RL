#!/usr/bin/env python3
"""
仿真环境内抓取可视化演示
在ManiSkill渲染的环境图像上直接显示抓取位姿
"""
import os
import sys
import torch

# 设置环境变量
os.environ['MUJOCO_GL'] = 'egl'

# 导入环境
from env_clutter import EnvClutterEnv
from config import get_config

def demo_env_grasp_visualization():
    """演示在仿真环境中可视化抓取"""
    print("🎬 仿真环境抓取可视化演示")
    print("="*50)
    
    try:
        # 1. 创建环境（带render模式）
        print("📋 创建环境...")
        config = get_config("default")
        env = EnvClutterEnv(
            num_envs=1,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",  # ✨ 关键：开启渲染模式
            use_discrete_action=True,
            custom_config=config
        )
        
        # 2. 重置环境
        print("🔄 重置环境...")
        obs, info = env.reset(seed=888)
        
        # 3. 获取目标物体
        target_obj = env.selectable_objects[0][0]
        print(f"🎯 目标物体: {target_obj.name}")
        
        # 4. 进行抓取检测，同时开启环境内可视化
        print("🔍 开始抓取检测和环境可视化...")
        print("💡 这将生成两种可视化:")
        print("  1. 原始点云可视化（如果支持）")
        print("  2. 🌟 环境渲染图像上的抓取位姿标注")
        
        grasps = env._detect_grasps_for_target(
            target_obj, 
            env_idx=0, 
            top_k=5,
            visualize=True,          # 传统可视化
            visualize_in_env=True    # ✨ 新功能：环境内可视化
        )
        
        if grasps and len(grasps) > 0:
            print("✅ 检测和可视化成功！")
            print(f"📊 检测结果:")
            print(f"  - 抓取候选: {len(grasps)} 个")
            print(f"  - 最佳分数: {grasps[0]['score']:.4f}")
            print(f"  - 最佳位置: [{grasps[0]['translation'][0]:.3f}, {grasps[0]['translation'][1]:.3f}, {grasps[0]['translation'][2]:.3f}]")
            
            # 检查生成的环境可视化文件
            env_viz_files = [f for f in os.listdir('.') if f.startswith('grasp_simulation_env_0_') and target_obj.name in f and f.endswith('.png')]
            if env_viz_files:
                print(f"🎬 生成的环境可视化: {env_viz_files[0]}")
                print(f"📋 详细信息文件: {env_viz_files[0].replace('.png', '_info.txt')}")
            
            return True
        else:
            print("❌ 未检测到抓取")
            return False
            
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'env' in locals():
            env.close()

def demo_multiple_objects_env_viz():
    """多物体环境可视化演示"""
    print("\n" + "="*50)
    print("🎬 多物体环境可视化演示")
    
    try:
        config = get_config("default")
        env = EnvClutterEnv(
            num_envs=1,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",
            use_discrete_action=True,
            custom_config=config
        )
        
        obs, info = env.reset(seed=777)
        
        print(f"环境中有 {len(env.selectable_objects[0])} 个可选物体")
        
        # 为前3个物体生成环境可视化
        max_objects = min(3, len(env.selectable_objects[0]))
        successful_viz = 0
        
        for i in range(max_objects):
            target_obj = env.selectable_objects[0][i]
            print(f"\n🎯 处理物体 {i+1}/{max_objects}: {target_obj.name}")
            
            try:
                grasps = env._detect_grasps_for_target(
                    target_obj, 
                    env_idx=0, 
                    top_k=3,
                    visualize=False,         # 跳过传统可视化
                    visualize_in_env=True    # 只使用环境可视化
                )
                
                if grasps and len(grasps) > 0:
                    print(f"  ✅ 环境可视化成功 - {len(grasps)} 个抓取，最佳分数: {grasps[0]['score']:.4f}")
                    successful_viz += 1
                else:
                    print(f"  ❌ 未检测到抓取")
                    
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
        print(f"\n📊 多物体可视化结果:")
        print(f"  - 成功: {successful_viz}/{max_objects}")
        print(f"  - 成功率: {successful_viz/max_objects*100:.1f}%")
        
        # 列出所有环境可视化文件
        env_files = [f for f in os.listdir('.') if f.startswith('grasp_simulation_env_') and f.endswith('.png')]
        if env_files:
            print(f"\n📁 生成的环境可视化文件:")
            for file in sorted(env_files):
                file_size = os.path.getsize(file) / 1024
                print(f"  - {file} ({file_size:.1f} KB)")
        
        return successful_viz > 0
        
    except Exception as e:
        print(f"❌ 多物体演示失败: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    print("🚀 仿真环境抓取可视化功能演示")
    print("这是全新功能：在ManiSkill环境渲染图像上直接标注抓取位姿！")
    
    # 基本演示
    success1 = demo_env_grasp_visualization()
    
    if success1:
        # 多物体演示
        choice = input("\n继续多物体环境可视化演示? (y/n): ").lower().strip()
        if choice == 'y':
            success2 = demo_multiple_objects_env_viz()
        else:
            success2 = True
            print("跳过多物体演示")
    else:
        success2 = False
    
    print("\n" + "="*60)
    print("📋 演示总结:")
    print(f"  基本环境可视化: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"  多物体可视化: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1:
        print("\n🎉 环境可视化功能集成成功！")
        print("\n💡 使用方法:")
        print("```python")
        print("# 创建环境时开启渲染")
        print("env = EnvClutterEnv(render_mode='rgb_array', ...)")
        print()
        print("# 使用环境内可视化")
        print("grasps = env._detect_grasps_for_target(")
        print("    target_obj,")  
        print("    visualize_in_env=True  # 🌟 新功能！")
        print(")")
        print("```")
        
        print("\n🌟 环境可视化的优势:")
        print("✅ 真实环境视角：显示抓取在实际仿真场景中的位置")
        print("✅ 直观理解：可以看到抓取相对于其他物体的空间关系")
        print("✅ 调试友好：便于验证抓取检测的准确性")
        print("✅ 研究价值：适合论文和演示使用")
        print("✅ 对比分析：原始环境vs标注环境并排显示")
        
    else:
        print("\n❌ 演示未完全成功，请检查错误信息")
        print("💡 提示: 确保安装了opencv-python: pip install opencv-python")
