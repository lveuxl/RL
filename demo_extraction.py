"""
demo_extraction.py — Jenga 连续抽块动画 Demo

自动选出最安全的 5 块核心木块（potentiality 最高），依次平滑抽出并消失。
整个过程中，除正在移动的木块外，塔的其余木块全部冻结为 kinematic，保证不会位移。

Usage:
    conda activate /opt/anaconda3/envs/skill
    python demo_extraction.py [--num_extract 5] [--output demo_phase1.mp4]
"""
import argparse
from collections import deque

import numpy as np
import torch
import gymnasium as gym
import imageio

from mani_skill.utils.structs.pose import Pose
from jenga_tower import (
    BLOCK_L, BLOCK_W, BLOCK_H, BLOCK_DENSITY,
    NUM_LEVELS, NUM_BLOCKS,
    get_support_graph,
)


# ─── 选块: 贪心反事实策略 ───

def _greedy_select(scene, actors, num_extract=5, sim_steps=80, threshold=0.02):
    """
    贪心选择最安全的 num_extract 个木块：每轮模拟移除每个候选块，
    选 potentiality 最高的（移除后塔最稳定），然后将其真正移除，再进入下一轮。

    排除最底层 (L0) 和最顶层 (L17) 的木块，只从中间层选取。
    """
    n = len(actors)
    device = actors[0].device
    far = torch.tensor([[999.0, 999.0, 999.0]], device=device)
    zero3 = torch.zeros(1, 3, device=device)

    removed = set()
    # 排除底层和顶层
    excluded = set(range(3)) | set(range(n - 3, n))
    selected_order = []

    for round_idx in range(num_extract):
        state = scene.get_sim_state()
        init_pos = torch.stack([a.pose.p[0] for a in actors])

        # 从高层向低层遍历: 同 potentiality 时优先选上层木块
        candidates = sorted(
            [i for i in range(n) if i not in removed and i not in excluded],
            key=lambda i: i // 3, reverse=True,
        )
        best_idx, best_pot = -1, -1.0

        for cand in candidates:
            actors[cand].set_pose(Pose.create_from_pq(far))
            actors[cand].set_linear_velocity(zero3)
            actors[cand].set_angular_velocity(zero3)

            for _ in range(sim_steps):
                scene.step()

            stable = sum(
                1 for j in range(n)
                if j != cand and j not in removed
                and torch.norm(actors[j].pose.p[0] - init_pos[j]).item() < threshold
            )
            pot = stable / (n - 1 - len(removed))

            scene.set_sim_state(state)

            # 同分时高层优先 (cand//3 更大)
            if pot > best_pot or (pot == best_pot and cand // 3 > best_idx // 3):
                best_pot = pot
                best_idx = cand

        selected_order.append(best_idx)
        removed.add(best_idx)
        # 真正移除该块 (影响后续轮的模拟)
        actors[best_idx].set_pose(Pose.create_from_pq(far))
        actors[best_idx].set_linear_velocity(zero3)
        actors[best_idx].set_angular_velocity(zero3)
        for _ in range(sim_steps):
            scene.step()

        lv, pi = divmod(best_idx, 3)
        print(f"  第 {round_idx + 1} 轮: 选中 block #{best_idx} L{lv}[{pi}]  potentiality={best_pot:.4f}")

    # 恢复所有块到初始状态 (让外层自行 reset)
    scene.set_sim_state(state)
    return selected_order


# ─── 冻结 / 解冻 ───

def _freeze_all(actors):
    """将所有木块设为 kinematic, 零速度。"""
    zero3 = torch.zeros(1, 3, device=actors[0].device)
    for a in actors:
        a._bodies[0].kinematic = True
        a.set_linear_velocity(zero3)
        a.set_angular_velocity(zero3)


def _disable_collision(actor):
    """关闭单个 actor 的碰撞 (避免抽出时产生接触力)。"""
    actor.set_collision_group(0, 0)
    actor.set_collision_group(1, 0)


# ─── 拍摄 ───

def _capture_frame(scene, camera):
    """使用 human_render_camera 拍一帧 512×512 RGB。"""
    scene.update_render(update_sensors=False, update_human_render_cameras=True)
    camera.capture()
    rgb = camera.get_obs(rgb=True, depth=False, segmentation=False, position=False)["rgb"]
    img = rgb[0].cpu().numpy()
    return img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)


# ─── 单块抽出动画 ───

def _extract_one(scene, camera, actor, actors,
                 steps=60, extraction_distance=0.15, pause_frames=15):
    """
    平滑抽出一块木块并消失, 返回帧列表。
    前提: 所有木块已被 _freeze_all() 冻结为 kinematic。
    """
    device = actor.device

    # 方向判定
    level = int(actor.name.split("_")[1])
    extract_dir = torch.tensor(
        [[1.0, 0.0, 0.0]] if level % 2 == 0 else [[0.0, 1.0, 0.0]],
        device=device,
    )
    delta = extract_dir * (extraction_distance / steps)

    # 关闭该块碰撞 (虽已 kinematic, 双保险)
    _disable_collision(actor)

    initial_p = actor.pose.p.clone()
    frames = []

    # 抽出前静帧
    for _ in range(pause_frames):
        scene.step()
        frames.append(_capture_frame(scene, camera))

    # 平滑抽出
    for i in range(steps):
        actor.set_pose(Pose.create_from_pq(initial_p + delta * (i + 1)))
        scene.step()
        frames.append(_capture_frame(scene, camera))

    # 消失
    far = torch.tensor([[999., 999., 999.]], device=device)
    zero3 = torch.zeros(1, 3, device=device)
    actor.set_pose(Pose.create_from_pq(far))
    actor.set_linear_velocity(zero3)
    actor.set_angular_velocity(zero3)

    # 消失后静帧
    for _ in range(pause_frames):
        scene.step()
        frames.append(_capture_frame(scene, camera))

    return frames


# ─── 主程序 ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jenga 连续抽块动画 Demo")
    parser.add_argument("--num_extract", type=int, default=5, help="抽取木块数")
    parser.add_argument("--steps", type=int, default=60, help="每块抽出帧数")
    parser.add_argument("--distance", type=float, default=0.15, help="抽出距离 (m)")
    parser.add_argument("--pause", type=int, default=15, help="抽出前后静帧数")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--output", type=str, default="demo_phase1.mp4", help="输出路径")
    args = parser.parse_args()

    env = gym.make(
        "JengaTower-v1",
        obs_mode="state",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
    )
    obs, _ = env.reset(seed=42)

    # 预热
    zero_action = np.zeros(env.action_space.shape)
    for _ in range(20):
        env.step(zero_action)

    uw = env.unwrapped

    # ── 第一步: 贪心选块 ──
    print("=" * 50)
    print("  贪心选块 (反事实模拟)...")
    print("=" * 50)
    extract_order = _greedy_select(uw.scene, uw.blocks, num_extract=args.num_extract)

    # 选完后 reset 环境到干净状态
    obs, _ = env.reset(seed=42)
    for _ in range(20):
        env.step(zero_action)

    # ── 第二步: 获取 human render camera ──
    camera = uw.scene.human_render_cameras["render_camera"]

    # ── 第三步: 冻结全塔 + 连续抽出 ──
    print("\n" + "=" * 50)
    print("  开始录制连续抽块动画...")
    print("=" * 50)

    _freeze_all(uw.blocks)

    all_frames = []

    # 开头静帧: 展示完整塔体
    for _ in range(30):
        uw.scene.step()
        all_frames.append(_capture_frame(uw.scene, camera))

    for seq, block_idx in enumerate(extract_order):
        lv, pi = divmod(block_idx, 3)
        direction = "+X" if lv % 2 == 0 else "+Y"
        print(f"  [{seq + 1}/{len(extract_order)}] 抽出 block #{block_idx}  L{lv}[{pi}]  方向={direction}")

        frames = _extract_one(
            uw.scene, camera, uw.blocks[block_idx], uw.blocks,
            steps=args.steps,
            extraction_distance=args.distance,
            pause_frames=args.pause,
        )
        all_frames.extend(frames)

    # 结尾静帧
    for _ in range(45):
        uw.scene.step()
        all_frames.append(_capture_frame(uw.scene, camera))

    # ── 导出视频 ──
    imageio.mimwrite(args.output, all_frames, fps=args.fps)
    duration = len(all_frames) / args.fps
    print(f"\n✓ {args.output}  ({len(all_frames)} 帧, {duration:.1f}s)")

    env.close()
