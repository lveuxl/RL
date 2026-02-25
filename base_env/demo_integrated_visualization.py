#!/usr/bin/env python3
"""
集成可视化功能的简单演示
展示如何使用环境内置的可视化功能
"""
import os
import sys
import torch
import gymnasium as gym

# 设置环境变量
os.environ['MUJOCO_GL'] = 'egl'

# 导入环境
from env_clutter import EnvClutterEnv
from config import get_config

def demo_integrated_visualization():
    """演示集成的可视化功能"""
    print("🎨 AnyGrasp集成可视化演示")
    print("="*50)
    
    # 创建环境
    print("📋 创建环境...")
    config = get_config("default")
    env = EnvClutterEnv(
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        use_discrete_action=True,
        custom_config=config
    )
    print("✅ 环境创建完成")
    
    try:
        # 重置环境
        print("\n🔄 重置环境...")
        obs, info = env.reset(seed=12345)
        print("✅ 环境重置完成")
        
        print(f"\n🎯 环境中有 {len(env.selectable_objects[0])} 个可选物体")
        
        # 为第一个物体进行可视化演示
        target_obj = env.selectable_objects[0][0]
        print(f"\n📦 演示目标: {target_obj.name}")
        
        print("\n🎨 开始抓取检测和可视化...")
        print("注意：根据环境配置，会优先尝试Open3D 3D可视化")
        print("      如果失败，会自动降级到matplotlib 2D可视化")
        
        # 调用集成的可视化功能
        grasps = env._detect_grasps_for_target(
            target_obj, 
            env_idx=0, 
            top_k=5, 
            visualize=True  # 🎨 开启可视化
        )
        
        if grasps is not None and len(grasps) > 0:
            print(f"\n✅ 可视化演示完成！")
            print(f"📊 检测结果摘要:")
            print(f"  - 抓取候选数: {len(grasps)}")
            print(f"  - 最佳抓取分数: {grasps[0]['score']:.4f}")
            print(f"  - 最佳抓取位置: [{grasps[0]['translation'][0]:.3f}, {grasps[0]['translation'][1]:.3f}, {grasps[0]['translation'][2]:.3f}]")
            print(f"  - 夹爪宽度: {grasps[0]['width']:.3f}m")
            
            # 检查是否有生成的可视化文件
            visualization_files = [f for f in os.listdir('.') if f.startswith('grasp_env_0_') and f.endswith('.png')]
            if visualization_files:
                print(f"\n📁 生成的可视化文件:")
                for file in visualization_files:
                    print(f"  - {file}")
        else:
            print("❌ 未检测到抓取候选")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
        return False
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()
        print("\n🔚 环境已关闭")

def demo_batch_visualization():
    """批量可视化演示"""
    print("\n" + "="*50)
    print("🎨 批量可视化演示")
    print("为所有物体生成可视化")
    
    config = get_config("default")
    env = EnvClutterEnv(
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        use_discrete_action=True,
        custom_config=config
    )
    
    try:
        obs, info = env.reset(seed=42)
        
        objects_to_visualize = min(3, len(env.selectable_objects[0]))  # 限制到3个物体
        print(f"📦 将为 {objects_to_visualize} 个物体生成可视化")
        
        successful_visualizations = 0
        
        for i in range(objects_to_visualize):
            target_obj = env.selectable_objects[0][i]
            print(f"\n🎯 处理物体 {i+1}/{objects_to_visualize}: {target_obj.name}")
            
            try:
                grasps = env._detect_grasps_for_target(
                    target_obj, 
                    env_idx=0, 
                    top_k=5, 
                    visualize=True
                )
                
                if grasps and len(grasps) > 0:
                    print(f"  ✅ 可视化成功 - {len(grasps)} 个抓取候选，最佳分数: {grasps[0]['score']:.4f}")
                    successful_visualizations += 1
                else:
                    print(f"  ❌ 未检测到抓取")
                    
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
        print(f"\n📊 批量可视化结果:")
        print(f"  - 成功: {successful_visualizations}/{objects_to_visualize}")
        print(f"  - 成功率: {successful_visualizations/objects_to_visualize*100:.1f}%")
        
        # 列出所有生成的文件
        all_files = [f for f in os.listdir('.') if f.startswith('grasp_env_') and f.endswith('.png')]
        if all_files:
            print(f"\n📁 生成的所有可视化文件:")
            for file in sorted(all_files):
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"  - {file} ({file_size:.1f} KB)")
        
        return successful_visualizations > 0
        
    except Exception as e:
        print(f"❌ 批量可视化失败: {e}")
        return False
    finally:
        env.close()

if __name__ == "__main__":
    print("🚀 开始AnyGrasp集成可视化演示")
    print("这个演示将展示环境内置的抓取可视化功能")
    
    # 基本演示
    success1 = demo_integrated_visualization()
    
    if success1:
        # 批量演示
        choice = input("\n继续批量可视化演示? (y/n): ").lower().strip()
        if choice == 'y':
            success2 = demo_batch_visualization()
        else:
            success2 = True
            print("跳过批量演示")
    else:
        success2 = False
    
    print("\n" + "="*50)
    print("📋 演示总结:")
    print(f"  基本可视化: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"  批量可视化: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1:
        print("\n🎉 可视化功能集成成功！")
        print("\n💡 使用方法:")
        print("  # 基本用法")
        print("  grasps = env._detect_grasps_for_target(target_obj, visualize=True)")
        print("\n  # 高级用法")
        print("  grasps = env._detect_grasps_for_target(")
        print("      target_obj=your_object,")
        print("      env_idx=0,")
        print("      top_k=5,")
        print("      visualize=True  # 开启可视化")
        print("  )")
        
        print("\n🔧 特性:")
        print("  ✅ 自动降级：Open3D失败时自动使用matplotlib")
        print("  ✅ 服务器友好：支持无图形界面环境")
        print("  ✅ 详细分析：包含统计信息和多视角")
        print("  ✅ 高质量输出：PNG格式，150 DPI")
    else:
        print("\n❌ 演示未完全成功，请检查错误信息")
