"""
残局 Potentiality 验证:
  移除若干关键木块 → 物理稳定 → get_gt_potentiality → 可视化

Usage:
    conda activate /opt/anaconda3/envs/skill
    python test_potentiality.py
"""
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from jenga_tower import (
    BLOCK_H, NUM_LEVELS, NUM_BLOCKS,
    get_gt_potentiality, get_support_graph, get_gt_stability,
)
from mani_skill.utils.structs.pose import Pose

# ─── 初始化环境 ───
env = gym.make("JengaTower-v1", obs_mode="state", render_mode="human",
               num_envs=1, sim_backend="cpu")
obs, _ = env.reset(seed=42)
uw = env.unwrapped

zero_action = np.zeros(env.action_space.shape)
for _ in range(10):
    env.step(zero_action)

# ─── 构造残局: 移除 6 块关键木块 ───
# L1 中间块(#4), L3 两侧块(#9,#11), L5 中间块(#16), L8 边块(#24), L10 中间块(#31)
removed_ids = [4, 9, 11, 16, 24, 31]
far = torch.tensor([[999.0, 999.0, 999.0]], device=uw.device)
zero3 = torch.zeros(1, 3, device=uw.device)

print(f"移除木块: {removed_ids}")
for rid in removed_ids:
    li, pi = divmod(rid, 3)
    print(f"  block #{rid}  L{li}[{pi}]")
    uw.blocks[rid].set_pose(Pose.create_from_pq(far))
    uw.blocks[rid].set_linear_velocity(zero3)
    uw.blocks[rid].set_angular_velocity(zero3)

# 物理稳定 (200 步, 让残局自然沉降)
for _ in range(200):
    uw.scene.step()

# 记录残局稳定后各块位置
remaining_ids = [i for i in range(NUM_BLOCKS) if i not in removed_ids]
positions_after_settle = {}
for i in remaining_ids:
    positions_after_settle[i] = uw.blocks[i].pose.p[0].cpu().numpy().copy()

print(f"\n残局稳定后, 剩余 {len(remaining_ids)} 块木块")

# ─── 提取 Physical Graph (残局) ───
remaining_blocks = [uw.blocks[i] for i in remaining_ids]
graph = get_support_graph(uw.scene, remaining_blocks)

edges = sum(sum(row) for row in graph["support_matrix"])
print(f"残局支撑边数: {edges}")

# ─── 计算 Potentiality ───
import time
print(f"\n计算 Potentiality ({len(remaining_blocks)} 块 × 100 步)...")
t0 = time.time()
potentiality = get_gt_potentiality(uw.scene, remaining_blocks)
elapsed = time.time() - t0
print(f"完成, 耗时 {elapsed:.1f}s\n")

# ─── 打印结果 ───
print(f"{'Block':>8}  {'OrigID':>7}  {'Layer':>6}  {'Potentiality':>14}")
print(f"{'-'*42}")
for local_i, global_i in enumerate(remaining_ids):
    li, pi = divmod(global_i, 3)
    marker = " ◄" if potentiality[local_i] < 1.0 else ""
    print(f"  #{global_i:2d}[{pi}]   ({local_i:2d})   L{li:<4d}  "
          f"{potentiality[local_i]:.4f}{marker}")

# ─── 可视化 ───
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 上图: 残局 Potentiality (按原始 block ID 对齐)
ax = axes[0]
full_pot = np.full(NUM_BLOCKS, np.nan)
for local_i, global_i in enumerate(remaining_ids):
    full_pot[global_i] = potentiality[local_i]

colors = []
for i in range(NUM_BLOCKS):
    if i in removed_ids:
        colors.append("#999999")
    elif np.isnan(full_pot[i]):
        colors.append("#999999")
    elif full_pot[i] < 0.9:
        colors.append("#d62728")
    elif full_pot[i] < 1.0:
        colors.append("#ff7f0e")
    else:
        colors.append("#2ca02c")

bar_vals = [0.0 if np.isnan(v) else v for v in full_pot]
bars = ax.bar(range(NUM_BLOCKS), bar_vals, color=colors, edgecolor="gray", linewidth=0.3)

for rid in removed_ids:
    ax.annotate("✕", (rid, 0.02), ha="center", fontsize=9, color="#666666")

ax.set_ylabel("Potentiality")
ax.set_title(f"Residual Potentiality (removed: {removed_ids})",
             fontsize=12, fontweight="bold")
ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.4)
ax.axhline(0.9, color="red", ls=":", lw=0.5, alpha=0.4)
for lv in range(1, NUM_LEVELS):
    ax.axvline(lv * 3 - 0.5, color="blue", lw=0.3, alpha=0.4, ls="--")
ax.set_xticks(range(0, NUM_BLOCKS, 3))
ax.set_xticklabels([f"L{i}" for i in range(NUM_LEVELS)], fontsize=7)
ax.set_ylim(0, 1.1)
ax.legend(["pot=1.0 line", "pot=0.9 line"], loc="lower right", fontsize=8)

# 下图: 残局侧视图 (X-Z), 按 Potentiality 着色
ax2 = axes[1]
ax2.set_title("Residual Tower Side View (colored by Potentiality)", fontsize=12, fontweight="bold")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Z (m)")

import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

norm = Normalize(vmin=0.5, vmax=1.0)
cmap = plt.cm.RdYlGn

for i in range(NUM_BLOCKS):
    b = uw.blocks[i]
    p = b.pose.p[0].cpu().numpy()
    li = i // 3

    if i in removed_ids:
        continue

    if li % 2 == 0:
        w, h = BLOCK_H, 0.15
    else:
        w, h = 0.15, BLOCK_H

    pot_val = full_pot[i] if not np.isnan(full_pot[i]) else 1.0
    fc = cmap(norm(pot_val))

    rect = mpatches.FancyBboxPatch(
        (p[0] - w / 2, p[2] - BLOCK_H / 2), w, BLOCK_H,
        boxstyle="round,pad=0.001", facecolor=fc, edgecolor="black", linewidth=0.5
    )
    ax2.add_patch(rect)
    ax2.text(p[0], p[2], f"{i}", ha="center", va="center", fontsize=5)

ax2.set_xlim(0.05, 0.35)
ax2.set_ylim(-0.02, NUM_LEVELS * BLOCK_H + 0.05)
ax2.set_aspect("equal")
sm = ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax2, shrink=0.6, label="Potentiality")

plt.tight_layout()
plt.savefig("test_potentiality_residual.png", dpi=150, bbox_inches="tight")
print(f"\n可视化已保存到: test_potentiality_residual.png")
plt.show(block=False)

env.close()
