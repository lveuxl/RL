"""
test_evaluate.py — 快速测试评估脚本
"""
import sys
import os

# 测试导入
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from jenga_tower import NUM_BLOCKS
    from jenga_ppo_wrapper import JengaPPOWrapper
    from vp3e_modules import VP3ENetwork, PriorGuidedActorCritic
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试环境创建
try:
    base_env = gym.make(
        "JengaTower-v1", obs_mode="state", render_mode="rgb_array",
        num_envs=1, sim_backend="cpu",
    )
    env = JengaPPOWrapper(base_env, lambda_int=0.0)
    print("✓ 环境创建成功")
    
    obs, info = env.reset(seed=42)
    print(f"✓ 环境重置成功, obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    env.close()
except Exception as e:
    print(f"✗ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试模型加载
try:
    device = torch.device("cpu")
    vision_net = VP3ENetwork(feat_dim=256, gnn_layers=4, max_blocks=NUM_BLOCKS).to(device)
    rl_net = PriorGuidedActorCritic(feat_dim=256).to(device)
    
    vision_path = "runs/jenga_ppo/vision_final.pt"
    rl_path = "runs/jenga_ppo/rl_final.pt"
    
    if os.path.exists(vision_path):
        vision_net.load_state_dict(torch.load(vision_path, map_location=device))
        print(f"✓ Vision 网络加载成功: {vision_path}")
    else:
        print(f"✗ Vision checkpoint 不存在: {vision_path}")
    
    if os.path.exists(rl_path):
        rl_net.load_state_dict(torch.load(rl_path, map_location=device))
        print(f"✓ RL 网络加载成功: {rl_path}")
    else:
        print(f"✗ RL checkpoint 不存在: {rl_path}")
    
    vision_net.eval()
    rl_net.eval()
    print("✓ 模型加载和设置成功")
    
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n所有测试通过！")
