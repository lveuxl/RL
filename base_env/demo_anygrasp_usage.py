#!/usr/bin/env python3
"""
AnyGrasp集成使用示例
展示如何在env_clutter环境中使用AnyGrasp进行抓取点检测
"""

import os
import sys
import numpy as np
import torch
import time

# 添加项目路径
project_root = "/home2/jzh/RL_RobotArm-main"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env_clutter import EnvClutterEnv
from config import get_config

def demo_single_grasp_detection():
    """演示单次抓取检测"""
    print("=" * 60)
    print("演示: 单次抓取检测")
    print("=" * 60)
    
    # 创建环境
    config = get_config("default")
    env = EnvClutterEnv(
        num_envs=1,
        use_discrete_action=True,
        custom_config=config,
        obs_mode="rgb+depth+segmentation",
        render_mode=None
    )
    
    try:
        # 重置环境
        obs, info = env.reset()
        print("✅ 环境初始化完成")
        
        if not env.anygrasp_enabled:
            print("❌ AnyGrasp未启用，无法进行演示")
            return
        
        # 获取第一个目标物体
        target_obj = env.selectable_objects[0][0]
        print(f"目标物体: {target_obj.name}")
        
        # 检测抓取点
        print("正在检测抓取点...")
        start_time = time.time()
        
        grasps = env._detect_grasps_for_target(target_obj, env_idx=0, top_k=10)
        
        detection_time = time.time() - start_time
        print(f"检测耗时: {detection_time:.2f}秒")
        
        if grasps and len(grasps) > 0:
            print(f"✅ 检测到{len(grasps)}个抓取候选")
            
            # 显示最佳抓取点详细信息
            best_grasp = grasps[0]
            print(f"\n最佳抓取点:")
            print(f"  分数: {best_grasp['score']:.4f}")
            print(f"  位置: [{best_grasp['translation'][0]:.3f}, {best_grasp['translation'][1]:.3f}, {best_grasp['translation'][2]:.3f}]")
            print(f"  夹爪宽度: {best_grasp['width']:.3f}m")
            
            # 显示抓取姿态（旋转矩阵的欧拉角）
            from scipy.spatial.transform import Rotation
            R = Rotation.from_matrix(best_grasp['rotation'])
            euler = R.as_euler('xyz', degrees=True)
            print(f"  姿态(欧拉角): [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
            
            # 显示所有候选的分数分布
            scores = [g['score'] for g in grasps]
            print(f"\n抓取分数分布:")
            print(f"  最高分: {max(scores):.4f}")
            print(f"  最低分: {min(scores):.4f}")
            print(f"  平均分: {np.mean(scores):.4f}")
            
        else:
            print("❌ 未检测到抓取点")
    
    finally:
        env.close()

def demo_multi_object_comparison():
    """演示多物体抓取点比较"""
    print("\n" + "=" * 60)
    print("演示: 多物体抓取点比较")
    print("=" * 60)
    
    # 创建环境
    config = get_config("default")
    env = EnvClutterEnv(
        num_envs=1,
        use_discrete_action=True,
        custom_config=config,
        obs_mode="rgb+depth+segmentation",
        render_mode=None
    )
    
    try:
        # 重置环境
        obs, info = env.reset()
        
        if not env.anygrasp_enabled:
            print("❌ AnyGrasp未启用，无法进行演示")
            return
        
        # 获取多个目标物体
        objects_to_test = env.selectable_objects[0][:min(3, len(env.selectable_objects[0]))]
        
        print(f"将比较{len(objects_to_test)}个物体的抓取难度")
        
        object_results = []
        
        for i, target_obj in enumerate(objects_to_test):
            print(f"\n检测物体 {i+1}: {target_obj.name}")
            
            start_time = time.time()
            grasps = env._detect_grasps_for_target(target_obj, env_idx=0, top_k=5)
            detection_time = time.time() - start_time
            
            if grasps and len(grasps) > 0:
                best_score = grasps[0]['score']
                grasp_count = len(grasps)
                avg_score = np.mean([g['score'] for g in grasps])
                
                object_results.append({
                    'name': target_obj.name,
                    'best_score': best_score,
                    'grasp_count': grasp_count,
                    'avg_score': avg_score,
                    'detection_time': detection_time
                })
                
                print(f"  ✅ 检测到{grasp_count}个抓取点")
                print(f"  最佳分数: {best_score:.4f}")
                print(f"  平均分数: {avg_score:.4f}")
                print(f"  检测耗时: {detection_time:.2f}秒")
            else:
                print(f"  ❌ 未检测到抓取点")
                object_results.append({
                    'name': target_obj.name,
                    'best_score': 0.0,
                    'grasp_count': 0,
                    'avg_score': 0.0,
                    'detection_time': detection_time
                })
        
        # 排序并显示结果
        if object_results:
            print(f"\n抓取难度排序（按最佳分数）:")
            sorted_results = sorted(object_results, key=lambda x: x['best_score'], reverse=True)
            
            for i, result in enumerate(sorted_results):
                print(f"{i+1}. {result['name']}")
                print(f"   最佳分数: {result['best_score']:.4f}")
                print(f"   抓取点数: {result['grasp_count']}")
                if result['grasp_count'] > 0:
                    print(f"   平均分数: {result['avg_score']:.4f}")
    
    finally:
        env.close()

def demo_discrete_action_with_anygrasp():
    """演示带AnyGrasp的离散动作执行"""
    print("\n" + "=" * 60)
    print("演示: 带AnyGrasp的离散动作执行")
    print("=" * 60)
    
    # 创建环境
    config = get_config("default")
    env = EnvClutterEnv(
        num_envs=1,
        use_discrete_action=True,
        custom_config=config,
        obs_mode="rgb+depth+segmentation",
        render_mode=None
    )
    
    try:
        # 重置环境
        obs, info = env.reset()
        print("✅ 环境初始化完成")
        
        # 执行几个离散动作
        for step in range(3):
            print(f"\n--- 步骤 {step + 1} ---")
            
            # 选择一个物体（循环选择）
            action = step % len(env.selectable_objects[0])
            target_name = env.selectable_objects[0][action].name if action < len(env.selectable_objects[0]) else "无效"
            
            print(f"执行动作: 选择物体索引 {action} ({target_name})")
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - start_time
            
            print(f"步骤耗时: {step_time:.2f}秒")
            print(f"奖励: {reward.item():.3f}")
            print(f"已抓取物体数: {len(env.grasped_objects[0])}")
            
            if terminated.item() or truncated.item():
                print("环境已终止")
                break
        
        print(f"\n最终结果:")
        print(f"总抓取物体数: {len(env.grasped_objects[0])}")
        print(f"剩余物体数: {len(env.remaining_indices[0])}")
    
    finally:
        env.close()

def main():
    """主函数"""
    print("AnyGrasp集成使用演示")
    print("请确保已正确安装AnyGrasp并下载了模型权重")
    
    # 检查权重文件
    checkpoint_path = "/home2/jzh/RL_RobotArm-main/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
    if not os.path.exists(checkpoint_path):
        print(f"❌ AnyGrasp权重文件不存在: {checkpoint_path}")
        print("请下载权重文件后再运行演示")
        return False
    
    try:
        # 演示1: 单次抓取检测
        demo_single_grasp_detection()
        
        # 演示2: 多物体比较
        demo_multi_object_comparison()
        
        # 演示3: 离散动作执行
        demo_discrete_action_with_anygrasp()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
