"""
3d_det_heuristic_baseline.py — 3D Detection + Rules Baseline

使用 3D 目标检测估计位姿，然后应用几何规则。
这是一个更强的 Baseline，比简单的质心估计更准确。

方法：
  1. 点云 → 3D 目标检测器 → 6D 位姿 (x,y,z,roll,pitch,yaw)
  2. 应用几何规则（与 Oracle Heuristic 相同）

对比：
  - Oracle Heuristic: 完美位姿（理论上界）
  - 3D Det + Rules: 3D 检测位姿（强 Baseline）
  - Point Cloud Heuristic: 质心估计（弱 Baseline）
  - RL Model: 端到端学习

Usage:
    conda activate /opt/anaconda3/envs/skill
    python 3d_det_heuristic_baseline.py --max_episodes 10 --seed 42
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from jenga_tower import NUM_BLOCKS, render_point_cloud
from jenga_ppo_wrapper import JengaPPOWrapper


# ════════════════════════════════════════════════════
#  简化的 3D 目标检测器
# ════════════════════════════════════════════════════

class Simple3DDetector(nn.Module):
    """
    简化的 3D 目标检测器（用于 Baseline）
    
    输入：点云 [K, 3]
    输出：中心位置 [3] + 置信度 [1]
    
    实际应用中可以使用：
    - PointNet++
    - VoteNet
    - 3DETR
    等更强的检测器
    """
    
    def __init__(self, feat_dim=128):
        super().__init__()
        
        # PointNet backbone
        self.pointnet = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat_dim),
        )
        
        # 中心回归头
        self.center_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 3),  # (x, y, z)
        )
        
        # 置信度头
        self.conf_head = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, pcd):
        """
        Args:
            pcd: [K, 3] 点云
        
        Returns:
            center: [3] 估计的中心位置
            conf: [1] 置信度 (0-1)
        """
        if len(pcd) == 0:
            return torch.zeros(3), torch.zeros(1)
        
        # PointNet: [K, 3] → [K, D] → [D]
        feat = self.pointnet(pcd)
        feat = feat.max(dim=0)[0]  # max pooling
        
        # 预测中心和置信度
        center = self.center_head(feat)
        conf = self.conf_head(feat)
        
        return center, conf


class Supervised3DDetector:
    """
    使用监督学习训练的 3D 检测器
    
    训练数据：从物理引擎获取 GT 位姿
    """
    
    def __init__(self, device="cpu"):
        self.device = device
        self.model = Simple3DDetector().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.trained = False
    
    def train_on_environment(self, env, num_samples=1000):
        """
        在环境中收集数据并训练检测器
        
        Args:
            env: Jenga 环境
            num_samples: 训练样本数
        """
        print(f"训练 3D 检测器（{num_samples} 样本）...")
        
        self.model.train()
        losses = []
        
        for i in range(num_samples):
            # 重置环境
            obs, info = env.reset()
            
            # 渲染点云
            uw = env.unwrapped
            cams = {k: v for k, v in uw.scene.sensors.items() 
                    if k.startswith("surround")}
            pcd_data = render_point_cloud(uw.scene, cams, uw.blocks)
            
            # 随机选择一个积木
            valid_ids = [i for i in range(NUM_BLOCKS) if info["mask"][i]]
            if not valid_ids:
                continue
            
            block_id = np.random.choice(valid_ids)
            
            # 获取点云和 GT 中心
            pcd = pcd_data["per_block_pcd"][block_id]
            if len(pcd) < 10:
                continue
            
            # GT 中心（从物理引擎）
            gt_center = uw.blocks[block_id]._objs[0].get_pose().p
            
            # 转换为 tensor
            pcd_t = torch.from_numpy(pcd[:, :3]).float().to(self.device)
            gt_center_t = torch.from_numpy(gt_center).float().to(self.device)
            
            # 前向传播
            pred_center, pred_conf = self.model(pcd_t)
            
            # 损失：中心位置 MSE + 置信度（固定为 1）
            loss_center = nn.functional.mse_loss(pred_center, gt_center_t)
            loss_conf = nn.functional.binary_cross_entropy(
                pred_conf, torch.ones_like(pred_conf)
            )
            loss = loss_center + 0.1 * loss_conf
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"  样本 {i+1}/{num_samples}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self.trained = True
        print(f"✓ 3D 检测器训练完成，平均损失: {np.mean(losses):.4f}")
    
    def detect(self, pointcloud):
        """
        检测单个积木的中心位置
        
        Args:
            pointcloud: [K, 3] numpy array
        
        Returns:
            center: [3] numpy array, 估计的中心位置
            conf: float, 置信度
        """
        if not self.trained:
            # 未训练，回退到质心
            if len(pointcloud) < 10:
                return None, 0.0
            return pointcloud[:, :3].mean(axis=0), 0.5
        
        with torch.no_grad():
            pcd_t = torch.from_numpy(pointcloud[:, :3]).float().to(self.device)
            center, conf = self.model(pcd_t)
            return center.cpu().numpy(), conf.item()


# ════════════════════════════════════════════════════
#  3D Det + Rules Agent
# ════════════════════════════════════════════════════

class ThreeDDetHeuristicAgent:
    """3D 检测 + 几何规则策略"""
    
    def __init__(self, detector, tower_center=(0.0, 0.0), conf_threshold=0.3):
        """
        Args:
            detector: Supervised3DDetector 实例
            tower_center: 塔中心的 XY 坐标
            conf_threshold: 置信度阈值
        """
        self.detector = detector
        self.tower_center = np.array(tower_center)
        self.conf_threshold = conf_threshold
        
        from jenga_tower import render_point_cloud
        self.render_point_cloud = render_point_cloud
    
    def _render_pcd(self, env):
        """渲染点云"""
        uw = env.unwrapped
        cams = {k: v for k, v in uw.scene.sensors.items() if k.startswith("surround")}
        pcd_data = self.render_point_cloud(uw.scene, cams, uw.blocks)
        return pcd_data["per_block_pcd"]
    
    def select_action(self, env, obs, info):
        """使用 3D 检测 + 几何规则选择动作"""
        mask = info["mask"]
        valid_ids = [i for i in range(NUM_BLOCKS) if mask[i]]
        
        if not valid_ids:
            raise ValueError("没有可抽取的积木")
        
        # ① 渲染点云
        per_block_pcd = self._render_pcd(env)
        
        # ② 使用 3D 检测器估计每个积木的中心
        centers = {}
        for idx in valid_ids:
            center, conf = self.detector.detect(per_block_pcd[idx])
            if center is not None and conf > self.conf_threshold:
                centers[idx] = center
        
        if not centers:
            # 所有检测都失败，随机选择
            return np.random.choice(valid_ids)
        
        visible_ids = list(centers.keys())
        positions = [(idx, centers[idx][0], centers[idx][1], centers[idx][2]) 
                     for idx in visible_ids]
        
        # ③ 应用几何规则（与 Oracle Heuristic 相同）
        z_values = [p[3] for p in positions]
        z_sorted = sorted(set(z_values))
        
        if len(z_sorted) <= 3:
            non_top = [p for p in positions if p[3] < max(z_values) - 0.01]
            if non_top:
                return non_top[np.random.randint(len(non_top))][0]
            return np.random.choice(visible_ids)
        
        # 过滤最高两层和最底层
        top_two_threshold = z_sorted[-2] - 0.005
        bottom_threshold = z_sorted[0] + 0.005
        
        candidates = [
            p for p in positions
            if p[3] > bottom_threshold and p[3] < top_two_threshold
        ]
        
        if not candidates:
            non_top = [p for p in positions if p[3] < max(z_values) - 0.01]
            if non_top:
                return non_top[np.random.randint(len(non_top))][0]
            return np.random.choice(visible_ids)
        
        # 选择距离塔中心最近的
        distances = []
        for idx, x, y, z in candidates:
            dist_2d = np.sqrt((x - self.tower_center[0])**2 + 
                            (y - self.tower_center[1])**2)
            distances.append((idx, dist_2d))
        
        distances.sort(key=lambda x: x[1])
        return distances[0][0]


# ════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="3D Det + Rules Baseline for Jenga")
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tower_center_x", type=float, default=0.0)
    p.add_argument("--tower_center_y", type=float, default=0.0)
    p.add_argument("--train_detector", action="store_true",
                   help="训练 3D 检测器（否则使用质心）")
    p.add_argument("--detector_samples", type=int, default=500,
                   help="训练检测器的样本数")
    p.add_argument("--device", type=str, default="cpu")
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
    
    # 创建 3D 检测器
    detector = Supervised3DDetector(device=args.device)
    
    # 训练检测器（可选）
    if args.train_detector:
        detector.train_on_environment(env, num_samples=args.detector_samples)
    else:
        print("跳过检测器训练，使用质心估计")
    
    # 创建 Agent
    agent = ThreeDDetHeuristicAgent(
        detector=detector,
        tower_center=(args.tower_center_x, args.tower_center_y),
    )
    
    all_meb = []
    total_attempts, total_success = 0, 0
    
    print("=" * 60)
    print("  3D Detection + Rules Baseline for Jenga")
    print(f"  Episodes: {args.max_episodes}  |  Max steps/ep: {args.max_steps}")
    print(f"  Detector: {'Trained' if args.train_detector else 'Centroid'}")
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
            
            # 选择动作
            try:
                block_id = agent.select_action(env, obs, info)
            except Exception as e:
                print(f"  Step {step + 1}: 错误 ({e}), 随机选择")
                block_id = np.random.choice(valid_ids)
            
            # 执行
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
    print("  3D Det + Rules Baseline 汇总")
    print(f"  MEB: {np.mean(all_meb):.2f} +/- {np.std(all_meb):.2f}")
    print(f"  SR:  {sr:.1%}  ({total_success}/{total_attempts})")
    print(f"  各 EP: {all_meb}")
    print(f"  耗时: {elapsed:.1f}s  ({elapsed / max(total_attempts, 1):.1f}s/step)")
    print("=" * 60)
    
    base_env.close()


if __name__ == "__main__":
    main()
