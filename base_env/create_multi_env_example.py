#!/usr/bin/env python3
"""
创建多环境可视化的标准示例
展示如何使用gym.make设置num_envs=4和parallel_in_single_scene=True
"""

import gymnasium as gym
import numpy as np

# 导入ManiSkill环境
import mani_skill.envs
from env_clutter import EnvClutterEnv


def create_multi_env_visualization():
    """
    标准的多环境可视化创建方法
    基于ManiSkill官方API
    """
    
    # 方法1: 直接使用gym.make创建多环境可视化
    env = gym.make(
        "EnvClutter-v1",               # 环境ID
        num_envs=4,                    # 4个并行环境
        parallel_in_single_scene=True, # 关键参数：所有环境显示在同一场景
        obs_mode="state",              # 观察模式（兼容parallel_in_single_scene）
        control_mode="pd_ee_delta_pose", # 控制模式
        reward_mode="dense",           # 奖励模式
        sim_backend="gpu",             # 仿真后端
        render_mode="human",           # 渲染模式
        robot_uids="panda"             # 机械臂类型
    )
    
    return env


def create_custom_multi_env():
    """
    自定义多环境创建方法
    展示更多可配置选项
    """
    
    # 方法2: 使用完整配置创建
    env_config = {
        "num_envs": 4,
        "parallel_in_single_scene": True,
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pose", 
        "reward_mode": "dense",
        "sim_backend": "gpu",
        "render_mode": "human",
        "robot_uids": "panda",
        "enable_shadow": True,           # 启用阴影效果
        "render_backend": "gpu",         # GPU渲染
    }
    
    env = gym.make("EnvClutter-v1", **env_config)
    return env


def usage_examples():
    """使用示例"""
    
    print("=== 多环境可视化创建示例 ===\n")
    
    # 示例1: 基础多环境
    print("🔹 示例1: 基础多环境可视化")
    print("```python")
    print("import gymnasium as gym")
    print("import mani_skill.envs")
    print("from env_clutter import EnvClutterEnv")
    print()
    print("env = gym.make(")
    print("    'EnvClutter-v1',")
    print("    num_envs=4,                    # 4个并行环境")
    print("    parallel_in_single_scene=True, # 所有环境显示在同一场景")
    print("    obs_mode='state',              # 状态观察模式")
    print("    render_mode='human'            # 人机交互模式")
    print(")")
    print("```\n")
    
    # 示例2: 完整配置
    print("🔹 示例2: 完整配置")
    print("```python")
    print("env_config = {")
    print("    'num_envs': 4,")
    print("    'parallel_in_single_scene': True,")
    print("    'obs_mode': 'state',")
    print("    'control_mode': 'pd_ee_delta_pose',")
    print("    'reward_mode': 'dense',")
    print("    'sim_backend': 'gpu',")
    print("    'render_mode': 'human',")
    print("    'robot_uids': 'panda'")
    print("}")
    print()
    print("env = gym.make('EnvClutter-v1', **env_config)")
    print("```\n")
    
    # 示例3: 基本使用流程
    print("🔹 示例3: 基本使用流程")
    print("```python")
    print("# 创建环境")
    print("env = gym.make('EnvClutter-v1', num_envs=4, parallel_in_single_scene=True, render_mode='human')")
    print()
    print("# 重置环境")
    print("obs, info = env.reset()")
    print()
    print("# 运行循环")
    print("for step in range(100):")
    print("    actions = env.action_space.sample()  # 随机动作")
    print("    obs, rewards, terms, truncs, infos = env.step(actions)")
    print("    if (terms | truncs).any():")
    print("        obs, info = env.reset()")
    print()
    print("# 关闭环境")
    print("env.close()")
    print("```\n")
    
    print("📝 重要说明:")
    print("- parallel_in_single_scene=True 会将所有环境显示在一个场景中")
    print("- 使用obs_mode='state'以兼容parallel_in_single_scene")
    print("- num_envs>1时自动使用GPU仿真")
    print("- 适合制作展示视频和多环境对比")


def test_creation():
    """测试环境创建"""
    
    print("=== 测试环境创建 ===")
    
    try:
        # 测试基础创建
        print("🔧 测试基础多环境创建...")
        env1 = create_multi_env_visualization()
        print(f"✅ 基础创建成功: {env1.num_envs} 个环境")
        env1.close()
        
        # 测试自定义创建
        print("🔧 测试自定义多环境创建...")
        env2 = create_custom_multi_env()
        print(f"✅ 自定义创建成功: {env2.num_envs} 个环境")
        env2.close()
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    
    # 显示使用示例
    usage_examples()
    
    # 测试创建
    test_creation()


if __name__ == "__main__":
    main()

