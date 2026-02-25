#!/usr/bin/env python3
"""
测试可见物体的点云提取和AnyGrasp检测
"""

import os
import sys
import numpy as np
import torch

# 添加项目路径
project_root = "/home/linux/jzh/RL_Robot"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env_clutter import EnvClutterEnv
from config import get_config

def test_visible_object():
    """测试可见物体的点云提取"""
    print("测试可见物体的点云提取...")
    
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
        print("✅ 环境重置成功")
        
        # 找到一个可见的物体
        visible_objects = []
        for i, obj in enumerate(env.selectable_objects[0]):
            obj_ids = obj.per_scene_id
            if obj_ids.numel() > 0:
                # 获取相机观测
                camera_obs = env._get_camera_observations("base_camera")
                segmentation = camera_obs["sensor_data"]["base_camera"]["segmentation"][0]
                actor_seg = segmentation[..., 0]
                
                obj_id = obj_ids[0].item()
                mask = (actor_seg == obj_id)
                pixel_count = mask.sum().item()
                
                if pixel_count > 0:
                    visible_objects.append((i, obj, pixel_count))
                    print(f"物体{i} ({obj.name}): {pixel_count}个像素")
        
        if not visible_objects:
            print("❌ 没有找到可见的物体")
            return
        
        # 选择像素数最多的物体
        visible_objects.sort(key=lambda x: x[2], reverse=True)
        best_idx, best_obj, pixel_count = visible_objects[0]
        
        print(f"\n选择最可见的物体: {best_obj.name} (索引{best_idx}, {pixel_count}像素)")
        
        # 测试点云提取
        points, colors = env._extract_target_pointcloud(best_obj, env_idx=0)
        
        if points is not None and len(points) > 0:
            print(f"✅ 成功提取点云: {len(points)}个点")
            print(f"点云范围:")
            print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            
            # 测试AnyGrasp检测
            if env.anygrasp_enabled:
                print(f"\n测试AnyGrasp检测...")
                grasps = env._detect_grasps_for_target(best_obj, env_idx=0, top_k=5)
                
                if grasps and len(grasps) > 0:
                    print(f"✅ 检测到{len(grasps)}个抓取候选")
                    best_grasp = grasps[0]
                    print(f"最佳抓取:")
                    print(f"  分数: {best_grasp['score']:.4f}")
                    print(f"  位置: [{best_grasp['translation'][0]:.3f}, {best_grasp['translation'][1]:.3f}, {best_grasp['translation'][2]:.3f}]")
                    print(f"  夹爪宽度: {best_grasp['width']:.3f}m")
                else:
                    print("❌ AnyGrasp未检测到抓取点")
            else:
                print("AnyGrasp未启用")
        else:
            print("❌ 点云提取失败")
        
    finally:
        env.close()

if __name__ == "__main__":
    test_visible_object()


