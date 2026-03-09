"""
pointcloud_heuristic_baseline.py — 基于点云的 Heuristic Baseline（公平对比版本）

与原 heuristic_jenga_baseline.py 的区别：
  - 原版：使用完美感知（物理引擎精确位姿）
  - 本版：使用点云估计位姿（与 RL 模型相同的输入）

这样才能公平对比 RL 模型的性能！

Usage:
    conda activate /opt/anaconda3/envs/skill
    python pointcloud_heuristic_baseline.py --max_episodes 10 --seed 42
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from jenga_tower import NUM_BLOCKS, render_point_cloud
from jenga_ppo_wrapper import JengaPPOWrapper


class PointCloudHeuristicAgent:
    """基于点云的几何规则策略（公平对比版本）"""
    
    def __init__(self, tower_center=(0.0, 0.0), n_pts=256, min_points=10):
        """
        Args:
            tower_center: 塔中心的 XY 坐标
            n_pts: 每个积木采样的点数
            min_points: 认为积木可见的最小点数阈值
        """
        self.tower_center = np.array(tower_center)
        self.n_pts = n_pts
        self.min_points = min_points
    
    def _render_point_cloud(self, env):
        """渲染点云（与 RL 模型相同的输入）"""
        uw = env.unwrapped
        cams = {k: v for k, v in uw.scene.sensors.items() if k.startswith("surround")}
        pcd_data = render_point_cloud(uw.scene, cams, uw.blocks)
        return pcd_data["per_block_pcd"]
    
    def _estimate_center_from_pointcloud(self, pointcloud):
        """从点云估计积木中心位置"""
        if len(pointcloud) < self.min_points:
            return None
        
        # 简单方法：计算点云质心
        xyz = pointcloud[:, :3]
        center = xyz.mean(axis=0)
        return center
    
    def select_action(self, env, obs, info):
        """
        根据点云估计的位姿 + 几何规则选择积木
        
        Args:
            env: 环境
            obs: 观测（未使用）
            info: 包含 mask 的信息字典
        
        Returns:
            block_id: int, 选中的积木 ID
        """
        mask = info["mask"]
        valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
        
        if not valid_ids:
            raise ValueError("没有可抽取的积木")
        
        # ① 渲染点云
        per_block_pcd = self._render_point_cloud(env)
        
        # ② 从点云估计每个积木的中心位置
        centers = {}
        for idx in valid_ids:
            center = self._estimate_center_from_pointcloud(per_block_pcd[idx])
            if center is not None:
                centers[idx] = center
        
        if not centers:
            # 所有积木都不可见（极端情况），随机选择
            return np.random.choice(valid_ids)
        
        visible_ids = list(centers.keys())
        positions = [(idx, centers[idx][0], centers[idx][1], centers[idx][2]) 
                     for idx in visible_ids]
        
        # ③ 计算 Z 轴范围
        z_values = [p[3] for p in positions]
        z_min = min(z_values)
        z_max = max(z_values)
        z_sorted = sorted(set(z_values))
        
        # 如果层数太少（<= 3层），无法应用过滤规则
        if len(z_sorted) <= 3:
            # 回退: 随机选择非最高层
            non_top_candidates = [p for p in positions if p[3] < z_max - 0.01]
            if non_top_candidates:
                chosen = np.random.choice(len(non_top_candidates))
                return non_top_candidates[chosen][0]
            else:
                return np.random.choice(visible_ids)
        
        # ④ 过滤规则: 排除最高两层和最底层
        # 注意：由于点云估计有误差，这里的阈值需要更宽松
        top_two_threshold = z_sorted[-2] - 0.01  # 增加容差
        bottom_threshold = z_sorted[0] + 0.01
        
        candidates = [
            p for p in positions
            if p[3] > bottom_threshold and p[3] < top_two_threshold
        ]
        
        # ⑤ 如果候选池为空，回退到随机选择非顶层
        if not candidates:
            non_top_candidates = [p for p in positions if p[3] < z_max - 0.01]
            if non_top_candidates:
                chosen = np.random.choice(len(non_top_candidates))
                return non_top_candidates[chosen][0]
            else:
                return np.random.choice(visible_ids)
        
        # ⑥ 选择规则: 计算到塔中心垂线的 2D 距离
        distances = []
        for idx, x, y, z in candidates:
            dist_2d = np.sqrt((x - self.tower_center[0])**2 + 
                            (y - self.tower_center[1])**2)
            distances.append((idx, dist_2d))
        
        # 返回距离最小的积木 ID
        distances.sort(key=lambda x: x[1])
        return distances[0][0]


def parse_args():
    p = argparse.ArgumentParser(description="Point Cloud Heuristic Baseline for Jenga")
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=15,
                   help="每 episode 最大抽取步数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tower_center_x", type=float, default=0.0)
    p.add_argument("--tower_center_y", type=float, default=0.0)
    p.add_argument("--n_pts", type=int, default=256,
                   help="每个积木采样的点数")
    p.add_argument("--min_points", type=int, default=10,
                   help="认为积木可见的最小点数")
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
    
    # 创建基于点云的启发式智能体
    agent = PointCloudHeuristicAgent(
        tower_center=(args.tower_center_x, args.tower_center_y),
        n_pts=args.n_pts,
        min_points=args.min_points,
    )
    
    all_meb = []
    total_attempts, total_success = 0, 0
    
    print("=" * 60)
    print("  Point Cloud Heuristic Baseline for Jenga")
    print(f"  Episodes: {args.max_episodes}  |  Max steps/ep: {args.max_steps}")
    print(f"  Strategy: 基于点云估计 + 几何规则（公平对比版本）")
    print(f"  Point sampling: {args.n_pts} points/block")
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
            
            # 使用基于点云的启发式策略选择动作
            try:
                block_id = agent.select_action(env, obs, info)
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
    print("  Point Cloud Heuristic Baseline 汇总")
    print(f"  MEB: {np.mean(all_meb):.2f} +/- {np.std(all_meb):.2f}")
    print(f"  SR:  {sr:.1%}  ({total_success}/{total_attempts})")
    print(f"  各 EP: {all_meb}")
    print(f"  耗时: {elapsed:.1f}s  ({elapsed / max(total_attempts, 1):.1f}s/step)")
    print("=" * 60)
    print("\n注意：此版本使用点云估计位姿，与 RL 模型使用相同输入")
    print("      这是公平的对比 baseline！")
    
    base_env.close()


if __name__ == "__main__":
    main()
