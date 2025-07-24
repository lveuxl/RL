#!/usr/bin/env python3
"""
测试运动规划器集成的简单脚本
"""

import torch
import numpy as np
from env_clutter2 import EnvClutterEnv

def test_motion_planner_integration():
    """测试运动规划器集成"""
    print("开始测试运动规划器集成...")
    
    try:
        # 创建环境
        env = EnvClutterEnv(
            obs_mode="state",
            reward_mode="dense",
            control_mode="pd_ee_delta_pose",
            num_envs=1,
            use_discrete_action=True,
            render_mode="rgb_array"
        )
        
        print("环境创建成功")
        
        # 重置环境
        obs, info = env.reset(seed=42)
        print("环境重置成功")
        
        # 检查运动规划器是否初始化
        if hasattr(env, 'motion_planner') and env.motion_planner is not None:
            print("运动规划器包装器创建成功")
            if env.motion_planner.initialized:
                print("运动规划器初始化成功")
            else:
                print("运动规划器未初始化，将在首次使用时初始化")
        else:
            print("运动规划器包装器未创建")
        
        # 测试一步离散动作
        if len(env.remaining_indices) > 0:
            action = 0  # 选择第一个可用物体
            print(f"执行动作: 抓取物体索引 {env.remaining_indices[action]}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"动作执行结果:")
            print(f"  奖励: {reward}")
            print(f"  成功: {info.get('success', False)}")
            print(f"  位移: {info.get('displacement', 0.0)}")
            print(f"  剩余物体: {info.get('remaining_objects', 0)}")
            print(f"  已抓取物体: {info.get('grasped_objects', 0)}")
        else:
            print("没有可用的物体进行抓取")
        
        # 清理资源
        env.close()
        print("环境清理完成")
        
        print("运动规划器集成测试完成!")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_motion_planner_integration()
    if success:
        print("✅ 测试通过")
    else:
        print("❌ 测试失败") 