"""
heuristic_jenga_baseline.py — Heuristic Baseline for Jenga Block Extraction

基于几何规则的启发式策略，使用完美感知（物理引擎底层位姿）。

策略逻辑:
  1. 获取所有积木的 Z 轴高度和 XY 平面坐标
  2. 过滤规则: 排除最高两层和最底下一层的木块
  3. 选择规则: 在剩余候选中，选择距离塔中心垂线最近的木块
  4. 回退规则: 若候选池为空，随机选择一个非顶层木块

指标:
  MEB (Mean Extracted Blocks): 平均每 episode 成功抽取数
  SR  (Step Success Rate):     成功抽取步 / 总尝试步

Usage:
    conda activate /opt/anaconda3/envs/skill
    python heuristic_jenga_baseline.py --max_episodes 10 --seed 42
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from jenga_tower import NUM_BLOCKS, NUM_LEVELS
from jenga_ppo_wrapper import JengaPPOWrapper


class HeuristicJengaAgent:
    """基于几何规则的 Jenga 抽取策略"""
    
    def __init__(self, tower_center=(0.0, 0.0)):
        """
        Args:
            tower_center: 塔中心的 XY 坐标 (默认原点)
        """
        self.tower_center = np.array(tower_center)
    
    def select_action(self, actors, valid_mask):
        """
        根据几何规则选择要抽取的积木。
        
        Args:
            actors: list of Actor objects (SAPIEN 物理对象)
            valid_mask: np.ndarray, shape (NUM_BLOCKS,), 1 表示可抽取
        
        Returns:
            block_id: int, 选中的积木 ID
        """
        valid_ids = [i for i in range(len(actors)) if valid_mask[i]]
        
        if not valid_ids:
            raise ValueError("没有可抽取的积木")
        
        # ① 获取所有有效积木的位姿
        positions = []
        for idx in valid_ids:
            actor = actors[idx]
            # 获取积木的世界坐标位置
            pose = actor.get_pose()
            pos = pose.p  # 3D position (x, y, z)
            positions.append((idx, pos[0], pos[1], pos[2]))
        
        # ② 计算 Z 轴范围
        z_values = [p[3] for p in positions]
        z_min = min(z_values)
        z_max = max(z_values)
        
        # 计算层高（假设每层高度约 0.015m，根据 Jenga 标准尺寸）
        # 为了鲁棒性，我们用 Z 值排序后分组
        z_sorted = sorted(set(z_values))
        
        # 如果层数太少（<= 3层），无法应用过滤规则
        if len(z_sorted) <= 3:
            # 回退: 随机选择非最高层
            non_top_candidates = [p for p in positions if p[3] < z_max - 0.01]
            if non_top_candidates:
                chosen = np.random.choice(len(non_top_candidates))
                return non_top_candidates[chosen][0]
            else:
                # 极端情况: 只剩顶层，随机选一个
                return np.random.choice(valid_ids)
        
        # ③ 过滤规则: 排除最高两层和最底层
        # 最高两层的 Z 阈值
        top_two_threshold = z_sorted[-2] - 0.005  # 留一点容差
        # 最底层的 Z 阈值
        bottom_threshold = z_sorted[0] + 0.005
        
        candidates = [
            p for p in positions
            if p[3] > bottom_threshold and p[3] < top_two_threshold
        ]
        
        # ④ 如果候选池为空，回退到随机选择非顶层
        if not candidates:
            non_top_candidates = [p for p in positions if p[3] < z_max - 0.01]
            if non_top_candidates:
                chosen = np.random.choice(len(non_top_candidates))
                return non_top_candidates[chosen][0]
            else:
                return np.random.choice(valid_ids)
        
        # ⑤ 选择规则: 计算到塔中心垂线的 2D 距离
        distances = []
        for idx, x, y, z in candidates:
            dist_2d = np.sqrt((x - self.tower_center[0])**2 + 
                            (y - self.tower_center[1])**2)
            distances.append((idx, dist_2d))
        
        # 返回距离最小的积木 ID
        distances.sort(key=lambda x: x[1])
        return distances[0][0]


def parse_args():
    p = argparse.ArgumentParser(description="Heuristic Baseline for Jenga")
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=15,
                   help="每 episode 最大抽取步数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tower_center_x", type=float, default=0.0)
    p.add_argument("--tower_center_y", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建环境
    base_env = gym.make(
        "JengaTower-v1", obs_mode="state", render_mode="rgb_array",
        num_envs=1, sim_backend="cpu",
    )
    env = JengaPPOWrapper(base_env, lambda_int=0.0)
    
    # 创建启发式智能体
    agent = HeuristicJengaAgent(
        tower_center=(args.tower_center_x, args.tower_center_y)
    )
    
    all_meb = []
    total_attempts, total_success = 0, 0
    
    print("=" * 60)
    print("  Heuristic Baseline for Jenga")
    print(f"  Episodes: {args.max_episodes}  |  Max steps/ep: {args.max_steps}")
    print(f"  Strategy: 排除最高两层+最底层, 选择最靠近塔中心的积木")
    print("=" * 60)
    
    t0 = time.time()
    
    for ep in range(args.max_episodes):
        obs, info = env.reset()
        ep_success = 0
        ep_collapsed = False
        
        print(f"\n── Episode {ep + 1}/{args.max_episodes} ──")
        
        for step in range(args.max_steps):
            mask = info["mask"]
            valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
            
            if not valid_ids:
                break
            
            # 获取物理引擎中的积木 actors
            uw = env.unwrapped
            actors = [blk._objs[0] for blk in uw.blocks]
            
            # 使用启发式策略选择动作
            try:
                block_id = agent.select_action(actors, mask)
            except Exception as e:
                print(f"  Step {step + 1}: 策略错误 ({e}), 随机选择")
                block_id = np.random.choice(valid_ids)
            
            # 执行动作
            obs, reward, done, _, info = env.step(block_id)
            total_attempts += 1
            collapsed = info.get("collapsed", False)
            lv, pi = divmod(block_id, 3)
            
            if collapsed:
                ep_collapsed = True
                print(f"  Step {step + 1}: #{block_id} L{lv}[{pi}] -> 坍塌  r={reward:.1f}")
                break
            
            ep_success += 1
            total_success += 1
            print(f"  Step {step + 1}: #{block_id} L{lv}[{pi}] -> 成功  r={reward:.1f}")
            
            if done:
                break
        
        all_meb.append(ep_success)
        print(f"  结果: MEB={ep_success}, {'坍塌' if ep_collapsed else '完整'}")
    
    elapsed = time.time() - t0
    sr = total_success / max(total_attempts, 1)
    
    print("\n" + "=" * 60)
    print("  Heuristic Baseline 汇总")
    print(f"  MEB: {np.mean(all_meb):.2f} +/- {np.std(all_meb):.2f}")
    print(f"  SR:  {sr:.1%}  ({total_success}/{total_attempts})")
    print(f"  各 EP: {all_meb}")
    print(f"  耗时: {elapsed:.1f}s  ({elapsed / max(total_attempts, 1):.1f}s/step)")
    print("=" * 60)
    
    base_env.close()


if __name__ == "__main__":
    main()
