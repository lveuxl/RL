"""
GPU 并行版离线数据生成 Pipeline — 全 GPU 渲染 + 多进程 CPU 采集

三阶段流水线:
  阶段 1 (GPU, 批量): reset → collapse → support graph → complexity 过滤
  阶段 1.5 (GPU, 批量): 多视角点云渲染, 一次拍摄 B 个环境
  阶段 2 (多进程 CPU): stability + potentiality 并行计算

Usage:
    conda activate /opt/anaconda3/envs/skill
    python generate_dataset_gpu.py --per_bucket 10000 --num_envs 1024 --workers 16
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp # 导入 mp

import gymnasium as gym
import h5py
import numpy as np
import torch

from jenga_tower import (
    BLOCK_H, BLOCK_W, BLOCK_L, BLOCK_DENSITY,
    NUM_LEVELS, NUM_BLOCKS,
    get_gt_stability, get_gt_potentiality,
    calculate_topology_complexity,
)
from mani_skill.utils.structs.pose import Pose

N_PTS = 256

BUCKET_CFG = {
    "easy":   {"c_lo": 0.0, "c_hi": 0.4, "layers": (3, 7),  "drop": (0.10, 0.25)},
    "medium": {"c_lo": 0.3, "c_hi": 0.7, "layers": (8, 13), "drop": (0.05, 0.15)},
    "hard":   {"c_lo": 0.7, "c_hi": 1.5, "layers": (14, 18), "drop": (0.10, 0.25)},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_bucket", type=int, default=10000)
    p.add_argument("--out", type=str, default="dataset_jenga_30000.h5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--workers", type=int, default=16,
                   help="CPU worker 进程数 (potentiality 并行)")
    p.add_argument("--collapse_threshold", type=float, default=0.03)
    p.add_argument("--flush_every", type=int, default=100)
    return p.parse_args()


# ─────────────────────────────────────────────────────
#  阶段 1: GPU 并行塔操作
# ─────────────────────────────────────────────────────

def batch_reset_tower(uw, num_layers_per_env, rng, p_drop_per_env, device):
    """并行 reset B 个环境的塔, 返回 active_masks (B, NUM_BLOCKS)。"""
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
            env_has_level = num_layers_t > level

            offset = (i - 1) * (0.05 + gap_w)
            pos_val = torch.tensor(
                [cx, cy + offset, z] if level % 2 == 0 else [cx + offset, cy, z],
                device=device,
            )

            pos_all = far.unsqueeze(0).expand(B, -1).clone()
            pos_all[env_has_level] = pos_val

            b.set_pose(Pose.create_from_pq(pos_all))
            vel = zero3.unsqueeze(0).expand(B, -1)
            b.set_linear_velocity(vel)
            b.set_angular_velocity(vel)
            active_masks[:, idx] = env_has_level

    for _ in range(20):
        uw.scene.step()

    # drop blocks (per-env random)
    for env_i in range(B):
        total_active = int(num_layers_per_env[env_i]) * 3
        removable = list(range(3, total_active))
        if not removable:
            continue
        n_drop = int(len(removable) * p_drop_per_env[env_i])
        if n_drop == 0:
            continue
        for rid in sorted(rng.choice(removable, n_drop, replace=False).tolist()):
            active_masks[env_i, rid] = False

    for idx in range(NUM_BLOCKS):
        b = uw.blocks[idx]
        inactive = ~active_masks[:, idx]
        if not inactive.any():
            continue
        cur_pos = b.pose.p.clone()
        cur_pos[inactive] = far
        b.set_pose(Pose.create_from_pq(cur_pos))
        vel = torch.zeros(B, 3, device=device)
        b.set_linear_velocity(vel)
        b.set_angular_velocity(vel)

    for _ in range(50):
        uw.scene.step()

    return active_masks


def batch_check_collapse(uw, active_masks, num_layers_per_env, threshold, device):
    """并行 collapse 检测, 返回 (B,) bool tensor。"""
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
    """GPU 并行构建所有环境的 support graph, 返回 list[dict | None]。"""
    B = uw.num_envs
    dt = uw.scene.timestep
    vol = BLOCK_L * BLOCK_W * BLOCK_H
    threshold = 0.1 * vol * BLOCK_DENSITY * 9.81

    results = [None] * B

    all_active_ids = []
    for env_i in range(B):
        all_active_ids.append(
            active_masks[env_i].nonzero(as_tuple=True)[0].cpu().tolist()
        )

    # GPU 批量查询所有 block pair 接触力
    pair_forces = {}
    for i in range(NUM_BLOCKS):
        for j in range(i + 1, NUM_BLOCKS):
            both_active = active_masks[:, i] & active_masks[:, j]
            if not both_active.any():
                continue
            pair_forces[(i, j)] = uw.scene.get_pairwise_contact_forces(
                uw.blocks[i], uw.blocks[j]
            )

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


# ─────────────────────────────────────────────────────
#  阶段 1.5: GPU 并行点云渲染
# ─────────────────────────────────────────────────────

def batch_render_point_cloud(uw, active_masks, candidate_env_indices, rng):
    """
    GPU 模式下一次渲染所有环境, 按 candidate 索引提取点云。

    Returns:
        dict: env_i → obs_point_clouds (N_active, N_PTS, 6)
    """
    scene = uw.scene
    surround_cams = {
        uid: s for uid, s in scene.sensors.items()
        if uid.startswith("surround")
    }

    scene.update_render(update_sensors=True, update_human_render_cameras=False)
    for sensor in surround_cams.values():
        sensor.capture()

    # 收集所有相机的原始数据 (保持在 GPU tensor 上)
    all_pos, all_rgb, all_seg, all_c2w = [], [], [], []
    for cam_uid, sensor in surround_cams.items():
        images = sensor.get_obs(rgb=True, position=True, segmentation=True)
        params = sensor.get_params()
        all_pos.append(images["position"].float() / 1000.0)   # (B,H,W,3)
        all_rgb.append(images["rgb"].float())                  # (B,H,W,3)
        all_seg.append(images["segmentation"])                 # (B,H,W,1)
        all_c2w.append(params["cam2world_gl"])                 # (B,4,4)

    # 构建 per_scene_id → block 全局索引 的映射
    block_sids = [blk._objs[0].per_scene_id for blk in uw.blocks]

    result = {}
    for env_i in candidate_env_indices:
        aids = active_masks[env_i].nonzero(as_tuple=True)[0].cpu().tolist()
        n_active = len(aids)
        if n_active == 0:
            result[env_i] = np.zeros((0, N_PTS, 6), dtype=np.float32)
            continue

        # 该环境的 active block sid set
        active_sids = {block_sids[idx] for idx in aids}
        sid_to_local = {block_sids[aids[li]]: li for li in range(n_active)}

        env_xyz_list, env_rgb_list, env_seg_list = [], [], []

        for cam_i in range(len(all_pos)):
            pos = all_pos[cam_i]   # (B,H,W,3)
            rgb = all_rgb[cam_i]   # (B,H,W,3)
            seg = all_seg[cam_i]   # (B,H,W,1)
            c2w = all_c2w[cam_i]   # (B,4,4)

            B_cam, H, W, _ = pos.shape
            pos_e = pos[env_i]     # (H,W,3)
            seg_e = seg[env_i, :, :, 0]  # (H,W)
            rgb_e = rgb[env_i]     # (H,W,3)

            valid = seg_e != 0
            if not valid.any():
                continue

            ones = torch.ones(H, W, 1, device=pos_e.device, dtype=pos_e.dtype)
            pts_homo = torch.cat([pos_e, ones], dim=-1).reshape(-1, 4)  # (H*W, 4)
            pts_world = (pts_homo @ c2w[env_i].T)[:, :3]  # (H*W, 3)

            valid_flat = valid.reshape(-1)
            env_xyz_list.append(pts_world[valid_flat].cpu().numpy())
            env_seg_list.append(seg_e.reshape(-1)[valid_flat].cpu().numpy().astype(int))

            rgb_flat = rgb_e.reshape(-1, 3)[valid_flat].cpu().numpy()
            env_rgb_list.append(rgb_flat / 255.0 if rgb_flat.max() > 1.0 else rgb_flat)

        if not env_xyz_list:
            result[env_i] = np.zeros((n_active, N_PTS, 6), dtype=np.float32)
            continue

        global_xyz = np.concatenate(env_xyz_list)
        global_rgb = np.concatenate(env_rgb_list)
        global_seg = np.concatenate(env_seg_list)
        global_pcd = np.concatenate([global_xyz, global_rgb], axis=1)

        obs_pcd = np.zeros((n_active, N_PTS, 6), dtype=np.float32)
        for sid, li in sid_to_local.items():
            mask = global_seg == sid
            if not mask.any():
                continue
            pc = global_pcd[mask]
            if len(pc) > N_PTS:
                obs_pcd[li] = pc[rng.choice(len(pc), N_PTS, replace=False)]
            else:
                obs_pcd[li, :len(pc)] = pc

        result[env_i] = obs_pcd

    return result


# ─────────────────────────────────────────────────────
#  阶段 2: 多进程 CPU — potentiality + stability
# ─────────────────────────────────────────────────────

def _worker_collect(task):
    """
    独立进程 worker: 创建 CPU 环境 → 重建塔 → 计算 potentiality + stability。
    每个 worker 自行管理环境生命周期 (进程级单例由 initializer 初始化)。
    """
    
    active_ids = task["active_ids"]
    active_ids = task["active_ids"]
    num_layers = task["num_layers"]
    seed = task["worker_seed"]

    import gymnasium as gym
    import torch
    from mani_skill.utils.structs.pose import Pose
    from jenga_tower import (
        BLOCK_H, NUM_BLOCKS, get_support_graph,
        get_gt_stability, get_gt_potentiality,
    )

    env = gym.make(
        "JengaTower-v1", obs_mode="state", render_mode=None,
        num_envs=1, sim_backend="cpu",
    )
    env.reset(seed=seed)
    uw = env.unwrapped
    device = uw.device
    gap_w, gap_h = uw.gap_w, uw.gap_h
    cx, cy = 0.2, 0.0

    far = torch.tensor([[500.0, 500.0, 0.1]], device=device)
    zero3 = torch.zeros(1, 3, device=device)

    for idx in range(NUM_BLOCKS):
        uw.blocks[idx].set_pose(Pose.create_from_pq(far))
        uw.blocks[idx].set_linear_velocity(zero3)
        uw.blocks[idx].set_angular_velocity(zero3)

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

    graph = get_support_graph(uw.scene, active_blocks)
    poses_np = np.array([b.pose.p[0].cpu().numpy() for b in active_blocks])
    s_load, s_balance = get_gt_stability(
        graph["support_matrix"], graph["volumes"], poses_np
    )
    potentiality = get_gt_potentiality(uw.scene, active_blocks, sim_steps=100)

    env.close()

    return {
        "env_i": task["env_i"],
        "support_matrix": np.array(graph["support_matrix"], dtype=np.int8),
        "volumes": np.array(graph["volumes"], dtype=np.float32),
        "heights": np.array(graph["heights"], dtype=np.float32),
        "poses": poses_np.astype(np.float32),
        "s_load": s_load.astype(np.float32),
        "s_balance": s_balance.astype(np.float32),
        "potentiality": potentiality.astype(np.float32),
    }


# ─────────────────────────────────────────────────────
#  HDF5 写入
# ─────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────
#  主循环
# ─────────────────────────────────────────────────────

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

    counts = {b: 0 for b in BUCKET_CFG}
    skipped = 0
    global_idx = 0

    # 断点续传
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
    print(f"GPU 并行生成: {B} envs, {args.workers} CPU workers")
    print(f"目标: {len(BUCKET_CFG)} 桶 × {target} = {total_target} 样本")
    print(f"输出: {args.out}\n")

    while sum(counts.values()) < total_target:
        # ===== 阶段 1: GPU 批量 reset + 过滤 =====
        unfilled = [b for b in bucket_names if counts[b] < target]
        if not unfilled:
            break
        
        # 计算每个桶还需要多少样本
        remains = np.array([target - counts[b] for b in unfilled])
        # 计算权重：剩下的越多，生成的概率越大
        #probs = remains / remains.sum()
        probs = (remains ** 2) / (remains ** 2).sum()
        env_buckets = rng.choice(unfilled, size=B, p=probs)
        num_layers_arr = np.zeros(B, dtype=int)
        p_drop_arr = np.zeros(B, dtype=float)

        for i in range(B):
            bucket = env_buckets[i]
            cfg = BUCKET_CFG[env_buckets[i]]
            # num_layers_arr[i] = rng.integers(cfg["layers"][0], cfg["layers"][1] + 1)
            # p_drop_arr[i] = rng.uniform(cfg["drop"][0], cfg["drop"][1])
            # --- 关键修改：强制缩小采样范围 ---
            if bucket == "easy":
                # 产生极简塔：层数少，掉落多，极大概率落入 0.0-0.4
                num_layers_arr[i] = rng.integers(3, 7) # 甚至可以尝试 (3, 6)
                p_drop_arr[i] = rng.uniform(0.10, 0.30) # 增加空洞，降低复杂度
            elif bucket == "medium":
                num_layers_arr[i] = rng.integers(8, 13)
                p_drop_arr[i] = rng.uniform(0.05, 0.15)
            else: # hard
                num_layers_arr[i] = rng.integers(14, 18)
                p_drop_arr[i] = rng.uniform(0.10, 0.25)

        active_masks = batch_reset_tower(uw, num_layers_arr, rng, p_drop_arr, device)

        for _ in range(10):
            uw.scene.step()

        collapsed = batch_check_collapse(
            uw, active_masks, num_layers_arr, args.collapse_threshold, device
        )
        graphs = batch_build_support_graph(uw, active_masks, device)

        # 过滤
        candidates = []
        for env_i in range(B):
            if collapsed[env_i] or graphs[env_i] is None:
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

        if not candidates:
            continue

        # ===== 阶段 1.5: GPU 并行点云渲染 =====
        cand_env_ids = [c["env_i"] for c in candidates]
        pcd_map = batch_render_point_cloud(uw, active_masks, cand_env_ids, rng)

        # ===== 阶段 2: 多进程 CPU — potentiality + stability =====
        tasks = []
        for ci, cand in enumerate(candidates):
            if counts[cand["bucket"]] >= target:
                continue
            tasks.append({
                "env_i": cand["env_i"],
                "active_ids": cand["active_ids"],
                "num_layers": cand["num_layers"],
                "worker_seed": int(rng.integers(0, 2**31)),
            })

        if not tasks:
            continue

        # 多进程并行计算 potentiality + stability
        worker_results = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            for res in pool.map(_worker_collect, tasks):
                worker_results[res["env_i"]] = res

        # 组装最终样本
        for cand in candidates:
            bucket = cand["bucket"]
            if counts[bucket] >= target:
                continue
            env_i = cand["env_i"]
            if env_i not in worker_results:
                continue

            detail = worker_results[env_i]
            obs_pcd = pcd_map.get(env_i, np.zeros((0, N_PTS, 6), dtype=np.float32))

            sample = {
                "global_idx": global_idx,
                "bucket": bucket,
                "num_layers": cand["num_layers"],
                "p_drop": cand["p_drop"],
                "complexity": cand["complexity"],
                "active_ids": np.array(cand["active_ids"], dtype=np.int32),
                "removed_ids": np.array(cand["removed_ids"], dtype=np.int32)
                    if cand["removed_ids"] else np.zeros(0, dtype=np.int32),
                "support_matrix": detail["support_matrix"],
                "volumes": detail["volumes"],
                "heights": detail["heights"],
                "poses": detail["poses"],
                "s_load": detail["s_load"],
                "s_balance": detail["s_balance"],
                "potentiality": detail["potentiality"],
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
                print(
                    f"[{done:5d}/{total_target}]  {status}  "
                    f"c={cand['complexity']:.3f} L={cand['num_layers']:2d} "
                    f"drop={cand['p_drop']:.2f} skip={skipped} "
                    f"{elapsed:.0f}s ETA={eta:.0f}s"
                )

            if sum(counts.values()) >= total_target:
                break

    flush_to_h5(args.out, write_buf)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  GPU 并行数据生成完成 → {args.out}")
    for b in bucket_names:
        print(f"    {b:8s}: {counts[b]} 样本")
    print(f"  丢弃: {skipped} (坍塌+复杂度不匹配)")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/max(global_idx,1):.2f}s/sample)")
    print(f"{'='*55}")

    env.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # 强制使用 spawn
    main()
