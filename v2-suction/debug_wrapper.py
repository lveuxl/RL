"""
调试包装器 - 检查每一步包装后的观测空间
"""

import gymnasium as gym
import numpy as np
import mani_skill.envs
from env_clutter import EnvClutterEnv  # 直接导入环境类
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from wrappers.mask_wrapper import ExtractMaskWrapper, SB3CompatWrapper


def main():
    print("=== 调试包装器流程 ===")
    
    # 步骤1: 创建原始环境
    print("\n1. 创建原始环境...")
    raw_env = gym.make(
        "EnvClutter-v1",
        num_envs=16,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        use_discrete_action=True,
    )
    print(f"原始环境观测空间: {raw_env.observation_space}")
    print(f"原始环境动作空间: {raw_env.action_space}")
    
    # 步骤2: 添加SB3兼容包装器
    print("\n2. 添加SB3兼容包装器...")
    env = SB3CompatWrapper(raw_env)
    print(f"SB3兼容包装器观测空间: {env.observation_space}")
    print(f"SB3兼容包装器动作空间: {env.action_space}")
    
    # 步骤3: 添加掩码提取包装器
    print("\n3. 添加掩码提取包装器...")
    env = ExtractMaskWrapper(env, max_n=15)
    print(f"掩码提取包装器观测空间: {env.observation_space}")
    print(f"掩码提取包装器动作空间: {env.action_space}")
    
    # 步骤4: 转换为SB3向量环境
    print("\n4. 转换为SB3向量环境...")
    vec_env = ManiSkillSB3VectorEnv(env)
    print(f"SB3向量环境观测空间: {vec_env.observation_space}")
    print(f"SB3向量环境动作空间: {vec_env.action_space}")
    
    # 测试观测
    print("\n5. 测试观测...")
    try:
        obs = vec_env.reset()
        print(f"观测类型: {type(obs)}")
        if isinstance(obs, dict):
            print("观测键:", list(obs.keys()))
            for key, value in obs.items():
                print(f"  {key}: {type(value)} {value.shape if hasattr(value, 'shape') else 'no shape'}")
        else:
            print(f"观测形状: {obs.shape if hasattr(obs, 'shape') else 'no shape'}")
    except Exception as e:
        print(f"观测测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    vec_env.close()
    print("\n调试完成！")


if __name__ == "__main__":
    main() 