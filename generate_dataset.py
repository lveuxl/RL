"""
离线数据生成 Pipeline — 靶向桶调度 + 连续复杂度课程

三个难度桶 (easy / medium / hard), 各 1000 样本, 共 3000。
每 100 个样本 + 结束时写入 HDF5。

Usage:
    conda activate /opt/anaconda3/envs/skill
    python generate_dataset.py [--per_bucket 1000] [--out dataset_jenga_3000.h5]
"""
import argparse
import os
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from jenga_tower import (
    BLOCK_H, NUM_LEVELS, NUM_BLOCKS,
    get_support_graph, get_gt_stability, get_gt_potentiality,
    render_point_cloud, calculate_topology_complexity,
)
from mani_skill.utils.structs.pose import Pose

N_PTS = 256  # 每块点云采样点数

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
    p.add_argument("--collapse_threshold", type=float, default=0.03)
    p.add_argument("--flush_every", type=int, default=100)
    return p.parse_args()


# ── 塔操作 ──

def reset_tower(uw, num_layers, rng, p_drop, device):
    # 修改前：把物体挂在 999 米高空
    # far = torch.tensor([[999.0, 999.0, 999.0]], device=device)

    # 修改后：放在远离塔的坐标 (X=2, Y=2)，并且贴着地面 (Z=0.05 约等于木块厚度一半)
    far = torch.tensor([[500.0, 500.0, 0.1]], device=device)
    zero3 = torch.zeros(1, 3, device=device)
    gap_w, gap_h = uw.gap_w, uw.gap_h
    cx, cy = 0.2, 0.0

    for level in range(NUM_LEVELS):
        z = level * (BLOCK_H + gap_h) + BLOCK_H / 2
        for i in range(3):
            idx = level * 3 + i
            b = uw.blocks[idx]
            if level >= num_layers:
                b.set_pose(Pose.create_from_pq(far))
                b.set_linear_velocity(zero3)
                b.set_angular_velocity(zero3)
                continue
            offset = (i - 1) * (0.05 + gap_w)
            pos = [cx, cy + offset, z] if level % 2 == 0 else [cx + offset, cy, z]
            b.set_pose(Pose.create_from_pq(torch.tensor([pos], device=device)))
            b.set_linear_velocity(zero3)
            b.set_angular_velocity(zero3)

    for _ in range(20):
        uw.scene.step()

    total_active = num_layers * 3
    removable = list(range(3, total_active))
    n_drop = int(len(removable) * p_drop)
    drop_ids = sorted(rng.choice(removable, n_drop, replace=False).tolist()) if n_drop > 0 else []

    for rid in drop_ids:
        uw.blocks[rid].set_pose(Pose.create_from_pq(far))
        uw.blocks[rid].set_linear_velocity(zero3)
        uw.blocks[rid].set_angular_velocity(zero3)

    for _ in range(50):
        uw.scene.step()

    active_ids = [i for i in range(total_active) if i not in drop_ids]
    return active_ids, drop_ids


def check_collapse(uw, active_ids, num_layers, threshold):
    gap_h = uw.gap_h
    top_level = num_layers - 1
    expected_z = top_level * (BLOCK_H + gap_h) + BLOCK_H / 2
    for idx in active_ids:
        if idx // 3 == top_level:
            if abs(uw.blocks[idx].pose.p[0, 2].item() - expected_z) > threshold:
                return True
    return False


# ── HDF5 写入 ──

def flush_to_h5(h5path, buffer):
    """增量追加样本到 HDF5 文件。"""
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


# ── 主循环 ──

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    target = args.per_bucket

    env = gym.make("JengaTower-v1", obs_mode="state", render_mode="rgb_array",
                   num_envs=1, sim_backend="cpu")
    env.reset(seed=42)
    uw = env.unwrapped
    device = uw.device

    surround_cams = {uid: s for uid, s in uw.scene.sensors.items()
                     if uid.startswith("surround")}

    counts = {b: 0 for b in BUCKET_CFG}
    skipped = 0
    global_idx = 0

    # ── 断点续传逻辑 ──
    if os.path.exists(args.out):
        print(f"[*] 检测到已存在的数据集 {args.out}，正在读取历史进度...")
        with h5py.File(args.out, "r") as f:
            existing_samples = list(f.keys())
            global_idx = len(existing_samples)
            
            for s_name in existing_samples:
                # 读取 attr 里的 bucket 标签
                b_name = f[s_name].attrs.get("bucket")
                if isinstance(b_name, bytes):
                    b_name = b_name.decode("utf-8")
                
                if b_name in counts:
                    counts[b_name] += 1
                    
        print(f"[*] 进度已恢复！当前已有样本: {counts}")
        print(f"[*] 下一个写入的索引将从 sample_{global_idx:05d} 开始\n")

    write_buf = []
    bucket_names = list(BUCKET_CFG.keys())

    t_start = time.time()
    total_target = target * len(BUCKET_CFG)

    print(f"靶向生成: {len(BUCKET_CFG)} 桶 × {target} = {total_target} 样本")
    print(f"输出: {args.out}\n")

    while sum(counts.values()) < total_target:
        # 选一个未满的桶
        unfilled = [b for b in bucket_names if counts[b] < target]
        bucket = rng.choice(unfilled)
        cfg = BUCKET_CFG[bucket]

        num_layers = int(rng.integers(cfg["layers"][0], cfg["layers"][1] + 1))
        p_drop = float(rng.uniform(cfg["drop"][0], cfg["drop"][1]))

        active_ids, removed_ids = reset_tower(uw, num_layers, rng, p_drop, device)

        if check_collapse(uw, active_ids, num_layers, args.collapse_threshold):
            skipped += 1
            continue

        active_blocks = [uw.blocks[i] for i in active_ids]

        for _ in range(10):
            uw.scene.step()

        graph = get_support_graph(uw.scene, active_blocks)
        complexity = calculate_topology_complexity(num_layers, graph["support_matrix"])

        # 检查复杂度是否落入目标桶
        if not (cfg["c_lo"] <= complexity < cfg["c_hi"]):
            skipped += 1
            continue

        poses = np.array([b.pose.p[0].cpu().numpy() for b in active_blocks])
        s_load, s_balance = get_gt_stability(
            graph["support_matrix"], graph["volumes"], poses
        )
        potentiality = get_gt_potentiality(uw.scene, active_blocks, sim_steps=100)

        pcd_data = render_point_cloud(uw.scene, surround_cams, active_blocks)

        obs_pcd = np.zeros((len(active_blocks), N_PTS, 6), dtype=np.float32)
        for bi, pc in enumerate(pcd_data["per_block_pcd"]):
            if len(pc) == 0:
                continue
            if len(pc) > N_PTS:
                obs_pcd[bi] = pc[rng.choice(len(pc), N_PTS, replace=False)]
            else:
                obs_pcd[bi, :len(pc)] = pc

        sample = {
            "global_idx": global_idx,
            "bucket": bucket,
            "num_layers": num_layers,
            "p_drop": p_drop,
            "complexity": complexity,
            "active_ids": np.array(active_ids, dtype=np.int32),
            "removed_ids": np.array(removed_ids, dtype=np.int32) if removed_ids else np.zeros(0, dtype=np.int32),
            "support_matrix": np.array(graph["support_matrix"], dtype=np.int8),
            "volumes": np.array(graph["volumes"], dtype=np.float32),
            "heights": np.array(graph["heights"], dtype=np.float32),
            "poses": poses.astype(np.float32),
            "s_load": s_load.astype(np.float32),
            "s_balance": s_balance.astype(np.float32),
            "potentiality": potentiality.astype(np.float32),
            "obs_point_clouds": obs_pcd,
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
            print(f"[{done:5d}/{total_target}]  {status}  "
                  f"c={complexity:.3f} L={num_layers:2d} drop={p_drop:.2f} "
                  f"skip={skipped} {elapsed:.0f}s ETA={eta:.0f}s")

    # 最终 flush
    flush_to_h5(args.out, write_buf)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  数据生成完成 → {args.out}")
    for b in bucket_names:
        print(f"    {b:8s}: {counts[b]} 样本")
    print(f"  丢弃: {skipped} (坍塌+复杂度不匹配)")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/max(global_idx,1):.2f}s/sample)")
    print(f"{'='*55}")

    env.close()


if __name__ == "__main__":
    main()
