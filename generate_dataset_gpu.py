"""
GPU 并行版离线数据生成 Pipeline — 1024 个环境同时仿真

利用 ManiSkill GPU 后端 + PyTorch 张量操作:
  阶段 1 (GPU, 批量): reset → collapse 检查 → support graph → complexity 过滤
  阶段 2 (CPU, 逐个): 对通过过滤的环境提取 stability / potentiality / 点云

Usage:
    conda activate /opt/anaconda3/envs/skill
    python generate_dataset_gpu.py [--per_bucket 1000] [--num_envs 1024] [--out dataset_jenga_3000.h5]
"""
import argparse
import os
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from jenga_tower import (
    BLOCK_H, BLOCK_W, BLOCK_L, BLOCK_DENSITY,
    NUM_LEVELS, NUM_BLOCKS,
    get_gt_stability, get_gt_potentiality,
    render_point_cloud, calculate_topology_complexity,
)
from mani_skill.utils.structs.pose import Pose

N_PTS = 256

BUCKET_CFG = {
    "easy":   {"c_lo": 0.0, "c_hi": 0.3, "layers": (3, 7),  "drop": (0.00, 0.05)},
    "medium": {"c_lo": 0.3, "c_hi": 0.7, "layers": (8, 13), "drop": (0.05, 0.15)},
    "hard":   {"c_lo": 0.7, "c_hi": 1.5, "layers": (14, 18), "drop": (0.10, 0.25)},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_bucket", type=int, default=1000)
    p.add_argument("--out", type=str, default="dataset_jenga_3000.h5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--collapse_threshold", type=float, default=0.03)
    p.add_argument("--flush_every", type=int, default=100)
    return p.parse_args()


# ── GPU 并行塔操作 ──

def batch_reset_tower(uw, num_layers_per_env, rng, p_drop_per_env, device):
    """
    并行 reset 所有环境的塔。

    Args:
        uw: unwrapped env (num_envs=B)
        num_layers_per_env: (B,) int array, 每个环境的层数
        rng: numpy RNG
        p_drop_per_env: (B,) float array, 每个环境的 drop 比例
        device: torch device

    Returns:
        active_masks: (B, NUM_BLOCKS) bool tensor, 哪些 block 是 active 的
    """
    B = uw.num_envs
    gap_w, gap_h = uw.gap_w, uw.gap_h
    cx, cy = 0.2, 0.0

    far = torch.tensor([500.0, 500.0, 0.1], device=device)
    zero3 = torch.zeros(3, device=device)

    num_layers_t = torch.tensor(num_layers_per_env, device=device, dtype=torch.long)

    active_masks = torch.zeros(B, NUM_BLOCKS, dtype=torch.bool, device=device)

    for level in range(NUM_LEVELS):
        z = level * (BLOCK_H + gap_h) + BLOCK_H / 2
        for i in range(3):
            idx = level * 3 + i
            b = uw.blocks[idx]

            env_has_level = num_layers_t > level  # (B,)

            offset = (i - 1) * (0.05 + gap_w)
            if level % 2 == 0:
                pos_val = torch.tensor([cx, cy + offset, z], device=device)
            else:
                pos_val = torch.tensor([cx + offset, cy, z], device=device)

            # 所有 env 先设置到远处
            pos_all = far.unsqueeze(0).expand(B, -1).clone()  # (B, 3)
            pos_all[env_has_level] = pos_val

            b.set_pose(Pose.create_from_pq(pos_all))
            vel = zero3.unsqueeze(0).expand(B, -1)
            b.set_linear_velocity(vel)
            b.set_angular_velocity(vel)

            active_masks[:, idx] = env_has_level

    # 让塔沉降
    for _ in range(20):
        uw.scene.step()

    # 并行 drop blocks
    for env_i in range(B):
        total_active = int(num_layers_per_env[env_i]) * 3
        removable = list(range(3, total_active))
        if not removable:
            continue
        n_drop = int(len(removable) * p_drop_per_env[env_i])
        if n_drop == 0:
            continue
        drop_ids = sorted(rng.choice(removable, n_drop, replace=False).tolist())
        for rid in drop_ids:
            active_masks[env_i, rid] = False

    # 把 inactive blocks 移走 (按 block 维度遍历, 每个 block 一次批量操作)
    for idx in range(NUM_BLOCKS):
        b = uw.blocks[idx]
        inactive_envs = ~active_masks[:, idx]  # (B,)
        if not inactive_envs.any():
            continue
        cur_pos = b.pose.p.clone()  # (B, 3)
        cur_pos[inactive_envs] = far
        b.set_pose(Pose.create_from_pq(cur_pos))
        vel = torch.zeros(B, 3, device=device)
        b.set_linear_velocity(vel)
        b.set_angular_velocity(vel)

    # 沉降
    for _ in range(50):
        uw.scene.step()

    return active_masks


def batch_check_collapse(uw, active_masks, num_layers_per_env, threshold, device):
    """
    并行检查每个环境的塔是否坍塌。

    Returns:
        collapsed: (B,) bool tensor
    """
    B = uw.num_envs
    gap_h = uw.gap_h
    collapsed = torch.zeros(B, dtype=torch.bool, device=device)

    num_layers_t = torch.tensor(num_layers_per_env, device=device, dtype=torch.long)

    for env_i in range(B):
        top_level = int(num_layers_t[env_i].item()) - 1
        expected_z = top_level * (BLOCK_H + gap_h) + BLOCK_H / 2
        for local_i in range(3):
            idx = top_level * 3 + local_i
            if active_masks[env_i, idx]:
                actual_z = uw.blocks[idx].pose.p[env_i, 2].item()
                if abs(actual_z - expected_z) > threshold:
                    collapsed[env_i] = True
                    break

    return collapsed


def batch_build_support_graph(uw, active_masks, device):
    """
    用 get_pairwise_contact_forces 为所有环境并行构建 support graph。

    Returns:
        list of dicts (len = B), 每个 dict 同 get_support_graph 返回格式, 但只包含 active blocks。
        如果某个环境查询失败则返回 None。
    """
    B = uw.num_envs
    dt = uw.scene.timestep
    vol = BLOCK_L * BLOCK_W * BLOCK_H
    threshold = 0.1 * vol * BLOCK_DENSITY * 9.81

    results = [None] * B

    # 收集每个环境的 active block indices
    all_active_ids = []
    for env_i in range(B):
        ids = active_masks[env_i].nonzero(as_tuple=True)[0].cpu().tolist()
        all_active_ids.append(ids)

    # 预计算所有 block pair 的 pairwise contact forces (GPU 批量)
    # 遍历所有可能的 block pair, 查询力
    pair_forces = {}
    for i in range(NUM_BLOCKS):
        for j in range(i + 1, NUM_BLOCKS):
            # 只在至少一个环境里两个 block 都 active 时才查询
            both_active = active_masks[:, i] & active_masks[:, j]
            if not both_active.any():
                continue
            forces = uw.scene.get_pairwise_contact_forces(
                uw.blocks[i], uw.blocks[j]
            )  # (B, 3)
            pair_forces[(i, j)] = forces

    for env_i in range(B):
        aids = all_active_ids[env_i]
        n = len(aids)
        if n == 0:
            continue

        volumes = [vol] * n
        heights = [uw.blocks[aid].pose.p[env_i, 2].item() for aid in aids]

        aid_to_local = {aid: li for li, aid in enumerate(aids)}
        support_matrix = [[0] * n for _ in range(n)]

        for (gi, gj), forces in pair_forces.items():
            if gi not in aid_to_local or gj not in aid_to_local:
                continue
            li, lj = aid_to_local[gi], aid_to_local[gj]
            fz = forces[env_i, 2].item() / dt

            if -fz > threshold:
                support_matrix[li][lj] = 1
            if fz > threshold:
                support_matrix[lj][li] = 1

        results[env_i] = {
            "volumes": volumes,
            "heights": heights,
            "support_matrix": support_matrix,
            "active_ids": aids,
        }

    return results


# ── HDF5 写入 (同 CPU 版) ──

def flush_to_h5(h5path, buffer):
    if not buffer:
        return
    mode = "a" if os.path.exists(h5path) else "w"
    with h5py.File(h5path, mode) as f:
        for sample in buffer:
            gname = f"sample_{sample['global_idx']:05d}"
            g = f.create_group(gname)
            for k, v in sample.items():
                if k == "global_idx":
                    continue
                if isinstance(v, np.ndarray):
                    g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                elif isinstance(v, str):
                    g.attrs[k] = v.encode("utf-8")
                else:
                    g.attrs[k] = v
    buffer.clear()


# ── CPU 环境: 用于精确采集 potentiality + 点云 ──

class CPUCollector:
    """单个 CPU 环境实例, 用于提取 potentiality 和点云。"""

    def __init__(self):
        self.env = gym.make(
            "JengaTower-v1", obs_mode="state", render_mode="rgb_array",
            num_envs=1, sim_backend="cpu",
        )
        self.env.reset(seed=99)
        self.uw = self.env.unwrapped
        self.device = self.uw.device
        self.surround_cams = {
            uid: s for uid, s in self.uw.scene.sensors.items()
            if uid.startswith("surround")
        }

    def collect_sample(self, num_layers, active_ids, removed_ids, rng):
        """
        在 CPU 环境里精确重建塔 → 采集 stability / potentiality / 点云。
        """
        uw = self.uw
        device = self.device
        gap_w, gap_h = uw.gap_w, uw.gap_h
        cx, cy = 0.2, 0.0

        far = torch.tensor([[500.0, 500.0, 0.1]], device=device)
        zero3 = torch.zeros(1, 3, device=device)

        # reset 所有 block 到远处
        for idx in range(NUM_BLOCKS):
            uw.blocks[idx].set_pose(Pose.create_from_pq(far))
            uw.blocks[idx].set_linear_velocity(zero3)
            uw.blocks[idx].set_angular_velocity(zero3)

        # 放置 active blocks
        for idx in active_ids:
            level, i = divmod(idx, 3)
            z = level * (BLOCK_H + gap_h) + BLOCK_H / 2
            offset = (i - 1) * (0.05 + gap_w)
            pos = [cx, cy + offset, z] if level % 2 == 0 else [cx + offset, cy, z]
            uw.blocks[idx].set_pose(
                Pose.create_from_pq(torch.tensor([pos], device=device))
            )
            uw.blocks[idx].set_linear_velocity(zero3)
            uw.blocks[idx].set_angular_velocity(zero3)

        for _ in range(60):
            uw.scene.step()

        active_blocks = [uw.blocks[i] for i in active_ids]

        # support graph (CPU 版, 用原始 get_contacts)
        from jenga_tower import get_support_graph
        graph = get_support_graph(uw.scene, active_blocks)

        poses_np = np.array([b.pose.p[0].cpu().numpy() for b in active_blocks])
        s_load, s_balance = get_gt_stability(
            graph["support_matrix"], graph["volumes"], poses_np
        )
        potentiality = get_gt_potentiality(uw.scene, active_blocks, sim_steps=100)

        pcd_data = render_point_cloud(uw.scene, self.surround_cams, active_blocks)
        obs_pcd = np.zeros((len(active_blocks), N_PTS, 6), dtype=np.float32)
        for bi, pc in enumerate(pcd_data["per_block_pcd"]):
            if len(pc) == 0:
                continue
            if len(pc) > N_PTS:
                obs_pcd[bi] = pc[rng.choice(len(pc), N_PTS, replace=False)]
            else:
                obs_pcd[bi, :len(pc)] = pc

        return {
            "support_matrix": np.array(graph["support_matrix"], dtype=np.int8),
            "volumes": np.array(graph["volumes"], dtype=np.float32),
            "heights": np.array(graph["heights"], dtype=np.float32),
            "poses": poses_np.astype(np.float32),
            "s_load": s_load.astype(np.float32),
            "s_balance": s_balance.astype(np.float32),
            "potentiality": potentiality.astype(np.float32),
            "obs_point_clouds": obs_pcd,
        }

    def close(self):
        self.env.close()


# ── 主循环 ──

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    target = args.per_bucket
    B = args.num_envs

    # GPU 并行环境
    env = gym.make(
        "JengaTower-v1", obs_mode="state", render_mode="rgb_array",
        num_envs=B, sim_backend="gpu",
    )
    env.reset(seed=42)
    uw = env.unwrapped
    device = uw.device

    # CPU 采集器 (精确 potentiality + 点云)
    cpu_collector = CPUCollector()

    counts = {b: 0 for b in BUCKET_CFG}
    skipped = 0
    global_idx = 0

    # ── 断点续传 ──
    if os.path.exists(args.out):
        print(f"[*] 检测到已存在的数据集 {args.out}，正在读取历史进度...")
        with h5py.File(args.out, "r") as f:
            existing_samples = list(f.keys())
            global_idx = len(existing_samples)
            for s_name in existing_samples:
                b_name = f[s_name].attrs.get("bucket")
                if isinstance(b_name, bytes):
                    b_name = b_name.decode("utf-8")
                if b_name in counts:
                    counts[b_name] += 1
        print(f"[*] 进度已恢复！当前已有样本: {counts}")
        print(f"[*] 下一个写入索引: sample_{global_idx:05d}\n")

    write_buf = []
    bucket_names = list(BUCKET_CFG.keys())
    total_target = target * len(BUCKET_CFG)

    t_start = time.time()
    print(f"GPU 并行生成: {B} 个环境 × {len(BUCKET_CFG)} 桶 × {target} = {total_target} 样本")
    print(f"输出: {args.out}\n")

    while sum(counts.values()) < total_target:
        # ===== 阶段 1: GPU 批量 reset + 过滤 =====

        # 为每个环境分配一个未满的桶
        unfilled = [b for b in bucket_names if counts[b] < target]
        if not unfilled:
            break

        env_buckets = rng.choice(unfilled, size=B)
        num_layers_arr = np.zeros(B, dtype=int)
        p_drop_arr = np.zeros(B, dtype=float)

        for i in range(B):
            cfg = BUCKET_CFG[env_buckets[i]]
            num_layers_arr[i] = rng.integers(cfg["layers"][0], cfg["layers"][1] + 1)
            p_drop_arr[i] = rng.uniform(cfg["drop"][0], cfg["drop"][1])

        # 并行 reset
        active_masks = batch_reset_tower(uw, num_layers_arr, rng, p_drop_arr, device)

        # 额外稳定步
        for _ in range(10):
            uw.scene.step()

        # 并行 collapse 检查
        collapsed = batch_check_collapse(
            uw, active_masks, num_layers_arr, args.collapse_threshold, device
        )

        # 并行构建 support graph + 计算 complexity
        graphs = batch_build_support_graph(uw, active_masks, device)

        # 逐环境过滤: 未坍塌 + complexity 在桶范围内
        candidates = []
        for env_i in range(B):
            if collapsed[env_i]:
                skipped += 1
                continue
            if graphs[env_i] is None:
                skipped += 1
                continue

            bucket = env_buckets[env_i]
            cfg = BUCKET_CFG[bucket]
            graph = graphs[env_i]

            complexity = calculate_topology_complexity(
                int(num_layers_arr[env_i]), graph["support_matrix"]
            )

            if not (cfg["c_lo"] <= complexity < cfg["c_hi"]):
                skipped += 1
                continue

            # 检查桶是否还需要样本
            if counts[bucket] >= target:
                continue

            candidates.append({
                "env_i": env_i,
                "bucket": bucket,
                "num_layers": int(num_layers_arr[env_i]),
                "p_drop": float(p_drop_arr[env_i]),
                "complexity": complexity,
                "active_ids": graph["active_ids"],
                "removed_ids": [
                    idx for idx in range(int(num_layers_arr[env_i]) * 3)
                    if not active_masks[env_i, idx].item()
                ],
            })

        # ===== 阶段 2: CPU 精确采集 =====
        for cand in candidates:
            bucket = cand["bucket"]
            if counts[bucket] >= target:
                continue

            detail = cpu_collector.collect_sample(
                cand["num_layers"], cand["active_ids"],
                cand["removed_ids"], rng,
            )

            sample = {
                "global_idx": global_idx,
                "bucket": bucket,
                "num_layers": cand["num_layers"],
                "p_drop": cand["p_drop"],
                "complexity": cand["complexity"],
                "active_ids": np.array(cand["active_ids"], dtype=np.int32),
                "removed_ids": np.array(cand["removed_ids"], dtype=np.int32)
                    if cand["removed_ids"] else np.zeros(0, dtype=np.int32),
                **detail,
            }
            write_buf.append(sample)
            counts[bucket] += 1
            global_idx += 1

            done = sum(counts.values())
            if done % args.flush_every == 0:
                flush_to_h5(args.out, write_buf)

            if done % 50 == 0 or done <= 5:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_target - done) if done else 0
                status = "  ".join(f"{b}={counts[b]}/{target}" for b in bucket_names)
                print(
                    f"[{done:5d}/{total_target}]  {status}  "
                    f"c={cand['complexity']:.3f} L={cand['num_layers']:2d} "
                    f"drop={cand['p_drop']:.2f} skip={skipped} "
                    f"{elapsed:.0f}s ETA={eta:.0f}s"
                )

            if sum(counts.values()) >= total_target:
                break

    # 最终 flush
    flush_to_h5(args.out, write_buf)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  GPU 并行数据生成完成 → {args.out}")
    for b in bucket_names:
        print(f"    {b:8s}: {counts[b]} 样本")
    print(f"  丢弃: {skipped} (坍塌+复杂度不匹配)")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/max(global_idx,1):.2f}s/sample)")
    print(f"{'='*55}")

    cpu_collector.close()
    env.close()


if __name__ == "__main__":
    main()
