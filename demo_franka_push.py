"""
demo_franka_push.py — Franka 机器人推 Jenga 木块 Demo

伪物理 + pd_ee_delta_pose 控制:
- Even 层: 长轴 X → 从 -X 小端面沿 +X 推出
- Odd  层: 长轴 Y → 从 -Y 小端面沿 +Y 推出
- EE 夹爪闭合竖直, 同步跟随 kinematic 木块
- 512×512 human render camera (3/4 俯瞰视角)

Usage:
    conda activate /opt/anaconda3/envs/skill
    python demo_franka_push.py [--num_extract 5] [--output demo_phase2.mp4]
"""
import argparse

import numpy as np
import torch
import gymnasium as gym
import imageio
from scipy.spatial.transform import Rotation as R

from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.robots.panda.panda import Panda
from jenga_tower import BLOCK_L, BLOCK_W, BLOCK_H, NUM_BLOCKS

# 缓存两种 EE 姿态 (approaching +X / +Y, closing -Z)
_ROT_CACHE = {}


def _push_geometry(level):
    """
    根据层奇偶返回 (push_dir, contact_offset, target_rot).
    Even 层: 长轴 X, 小端面在 ±X → push -X, dir +X
    Odd  层: 长轴 Y, 小端面在 ±Y → push -Y, dir +Y
    """
    if level % 2 == 0:
        push_dir = np.array([1., 0., 0.])
        contact_offset = np.array([-BLOCK_L / 2, 0., 0.])
        approaching = np.array([1., 0., 0.])
    else:
        push_dir = np.array([0., 1., 0.])
        contact_offset = np.array([0., -BLOCK_L / 2, 0.])
        approaching = np.array([0., 1., 0.])

    key = tuple(approaching)
    if key not in _ROT_CACHE:
        p = Panda.build_grasp_pose(approaching, np.array([0., 0., -1.]), np.zeros(3))
        q = p.q
        _ROT_CACHE[key] = R.from_quat([q[1], q[2], q[3], q[0]])

    return push_dir, contact_offset, _ROT_CACHE[key]


# ─── 渲染 ───

def _capture(scene, camera):
    scene.update_render(update_sensors=False, update_human_render_cameras=True)
    camera.capture()
    rgb = camera.get_obs(rgb=True, depth=False, segmentation=False, position=False)["rgb"]
    img = rgb[0].cpu().numpy()
    return img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)


# ─── 冻结 / 解冻 ───

def _freeze_all(blocks):
    zero3 = torch.zeros(1, 3, device=blocks[0].device)
    for b in blocks:
        b._bodies[0].kinematic = True
        b.set_linear_velocity(zero3)
        b.set_angular_velocity(zero3)


def _unfreeze_remaining(blocks, removed):
    for i, b in enumerate(blocks):
        if i not in removed:
            b._bodies[0].kinematic = False


def _disable_collision(actor):
    actor.set_collision_group(0, 0)
    actor.set_collision_group(1, 0)


# ─── EE 控制 ───

def _ee_action(agent, target_pos, target_rot):
    """
    pd_ee_delta_pose 7D action: [pos_norm(3), rot_norm(3), gripper(1)]
    target_rot: scipy Rotation 对象 (避免 Euler Gimbal lock)
    pos: clip((target-cur)/0.1, -1, 1) → 经 clip_and_scale 还原为 pos_error (≤0.1m)
    rot: -rotvec_err/0.1 → 经 norm-clip * rot_lower(-0.1) 还原为 rotvec_err (≤0.1 rad)
         对于 ≤0.1 rad 的小旋转, rotvec ≈ Euler XYZ, controller 可正确解读
    """
    cur_pos = agent.tcp.pose.p[0].cpu().numpy()
    cur_q = agent.tcp.pose.q[0].cpu().numpy()

    pos_act = np.clip((target_pos - cur_pos) / 0.1, -1, 1)

    cur_r = R.from_quat([cur_q[1], cur_q[2], cur_q[3], cur_q[0]])
    rotvec_err = (target_rot * cur_r.inv()).as_rotvec()
    rot_act = -rotvec_err / 0.1
    n = np.linalg.norm(rot_act)
    if n > 1:
        rot_act /= n

    return torch.tensor(
        np.concatenate([pos_act, rot_act, [-1.0]]),
        dtype=torch.float32,
    ).unsqueeze(0)


def _move_ee(env, scene, camera, target_pos, target_rot,
             max_steps=80, hold=3, thresh=0.005):
    """EE P-tracking 到目标位姿, 到达后保持 hold 帧"""
    agent = env.unwrapped.agent
    frames = []
    for _ in range(max_steps):
        action = _ee_action(agent, target_pos, target_rot)
        env.step(action)
        frames.append(_capture(scene, camera))
        if np.linalg.norm(agent.tcp.pose.p[0].cpu().numpy() - target_pos) < thresh:
            for _ in range(hold):
                env.step(action)
                frames.append(_capture(scene, camera))
            break
    return frames


# ─── 单块推出动画 ───

def animate_franka_pushing(env, selected_actor, steps=60, extraction_distance=0.15):
    """
    Franka 推木块全流程:
    approach → contact → synchronized push → disappear → retract
    """
    uw = env.unwrapped
    scene, agent, blocks = uw.scene, uw.agent, uw.blocks
    device = selected_actor.device
    camera = scene.human_render_cameras["render_camera"]

    level = int(selected_actor.name.split("_")[1])
    block_pos = selected_actor.pose.p[0].cpu().numpy()
    push_dir, c_off, rot = _push_geometry(level)

    contact = block_pos + c_off
    pre_contact = contact - push_dir * 0.02

    # ── 冻结全塔 + 禁用碰撞 ──
    _freeze_all(blocks)
    _disable_collision(selected_actor)

    frames = []

    # ── Phase 1: Approach (standoff → pre-contact → contact) ──
    standoff = pre_contact - push_dir * 0.05
    standoff[2] += 0.06
    frames += _move_ee(env, scene, camera, standoff, rot, max_steps=80)
    frames += _move_ee(env, scene, camera, pre_contact, rot, max_steps=40)
    frames += _move_ee(env, scene, camera, contact, rot, max_steps=20, hold=2)

    # ── Phase 2: Synchronized push ──
    initial_p = selected_actor.pose.p.clone()
    delta_t = torch.tensor(
        push_dir * extraction_distance / steps, device=device
    ).unsqueeze(0)

    for i in range(steps):
        target_p = initial_p + delta_t * (i + 1)
        selected_actor.set_pose(Pose.create_from_pq(target_p))

        ee_target = target_p[0].cpu().numpy() + c_off
        action = _ee_action(agent, ee_target, rot)
        env.step(action)
        frames.append(_capture(scene, camera))

    # ── Phase 3: Disappear ──
    zero3 = torch.zeros(1, 3, device=device)
    selected_actor.set_pose(
        Pose.create_from_pq(torch.tensor([[999., 999., 999.]], device=device))
    )
    selected_actor.set_linear_velocity(zero3)
    selected_actor.set_angular_velocity(zero3)

    # ── Phase 4: Retract ──
    pullback = contact + push_dir * (extraction_distance - 0.05)
    pullback[2] += 0.06
    frames += _move_ee(env, scene, camera, pullback, rot, max_steps=30, hold=0)
    hover = pullback.copy()
    hover[2] += 0.10
    frames += _move_ee(env, scene, camera, hover, rot, max_steps=30, hold=0)

    return frames


# ─── 贪心选块 ───

def _greedy_select(scene, actors, num_extract=5, sim_steps=200, threshold=0.015):
    """
    贪心选块 — 严格保塔不倒:
    - sim_steps=200: 充分暴露延迟崩塌
    - 每层最多抽 1 块 (防止同层抽空)
    - 优先中间块 (index%3==1, Jenga 经典最安全位置)
    - 每轮选块后做累积验证: 移除所有已选块后跑 200 步确认稳定
    - pot < 1.0 直接跳过 (零容忍)
    """
    n = len(actors)
    device = actors[0].device
    far = torch.tensor([[999., 999., 999.]], device=device)
    zero3 = torch.zeros(1, 3, device=device)

    removed = set()
    excluded = set(range(3)) | set(range(n - 3, n))  # L0 + 顶层
    levels_touched = set()
    selected = []

    for rnd in range(num_extract):
        state = scene.get_sim_state()
        init_pos = torch.stack([a.pose.p[0] for a in actors])

        candidates = [
            i for i in range(n)
            if i not in removed and i not in excluded
            and i // 3 not in levels_touched
        ]
        # 排序: 中间块优先 (i%3==1), 同优先级高层优先
        candidates.sort(key=lambda i: (-(i % 3 == 1), -(i // 3)))

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

            if pot < 1.0:
                continue
            if pot > best_pot or (pot == best_pot and cand // 3 > best_idx // 3):
                best_pot = pot
                best_idx = cand

        if best_idx == -1:
            print(f"  第 {rnd + 1} 轮: 无法找到 pot=1.0 的候选块, 停止选块")
            break

        # 累积验证: 移除所有已选 + 本轮块, 跑长模拟
        for idx in selected:
            actors[idx].set_pose(Pose.create_from_pq(far))
            actors[idx].set_linear_velocity(zero3)
            actors[idx].set_angular_velocity(zero3)
        actors[best_idx].set_pose(Pose.create_from_pq(far))
        actors[best_idx].set_linear_velocity(zero3)
        actors[best_idx].set_angular_velocity(zero3)
        for _ in range(sim_steps):
            scene.step()
        all_stable = all(
            torch.norm(actors[j].pose.p[0] - init_pos[j]).item() < threshold
            for j in range(n) if j not in removed and j != best_idx
        )
        scene.set_sim_state(state)

        if not all_stable:
            lv, pi = divmod(best_idx, 3)
            print(f"  第 {rnd + 1} 轮: block #{best_idx} L{lv}[{pi}] 累积验证失败, 跳过")
            excluded.add(best_idx)
            continue

        selected.append(best_idx)
        removed.add(best_idx)
        levels_touched.add(best_idx // 3)
        # 更新场景: 移除已选块后 settle
        for idx in selected:
            actors[idx].set_pose(Pose.create_from_pq(far))
            actors[idx].set_linear_velocity(zero3)
            actors[idx].set_angular_velocity(zero3)
        for _ in range(sim_steps):
            scene.step()

        lv, pi = divmod(best_idx, 3)
        print(f"  第 {rnd + 1} 轮: block #{best_idx} L{lv}[{pi}]  pot={best_pot:.4f} ✓")

    scene.set_sim_state(state)
    return selected


# ─── 主程序 ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka Push Jenga Demo")
    parser.add_argument("--num_extract", type=int, default=5)
    parser.add_argument("--blocks", type=int, nargs="+", default=None,
                        help="跳过选块, 直接指定 block indices")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--distance", type=float, default=0.15)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="demo_phase2.mp4")
    args = parser.parse_args()

    env = gym.make(
        "JengaTower-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
        max_episode_steps=100000,
    )
    obs, _ = env.reset(seed=42)
    zero_action = torch.zeros(1, env.action_space.shape[-1])
    for _ in range(20):
        env.step(zero_action)

    uw = env.unwrapped

    # ── 选块 ──
    if args.blocks:
        extract_order = args.blocks
        print(f"使用指定木块: {extract_order}")
    else:
        print("=" * 50)
        print("  贪心选块 (反事实模拟)...")
        print("=" * 50)
        extract_order = _greedy_select(uw.scene, uw.blocks, num_extract=args.num_extract)

    # Reset 到干净状态
    obs, _ = env.reset(seed=42)
    zero_action = torch.zeros(1, env.action_space.shape[-1])
    for _ in range(20):
        env.step(zero_action)

    camera = uw.scene.human_render_cameras["render_camera"]

    print("\n" + "=" * 50)
    print("  录制 Franka Push Demo...")
    print("=" * 50)

    all_frames = []
    removed = set()
    settle_action = torch.zeros(1, env.action_space.shape[-1])
    settle_action[0, -1] = -1.0

    # Intro
    for _ in range(30):
        uw.scene.step()
        all_frames.append(_capture(uw.scene, camera))

    for seq, block_idx in enumerate(extract_order):
        lv, pi = divmod(block_idx, 3)
        print(f"  [{seq + 1}/{len(extract_order)}] 推出 block #{block_idx} L{lv}[{pi}]")

        frames = animate_franka_pushing(
            env, uw.blocks[block_idx],
            steps=args.steps,
            extraction_distance=args.distance,
        )
        all_frames.extend(frames)
        removed.add(block_idx)

        # Unfreeze → settle → (next iteration re-freezes)
        _unfreeze_remaining(uw.blocks, removed)
        for _ in range(30):
            env.step(settle_action)
            all_frames.append(_capture(uw.scene, camera))

    # Outro
    for _ in range(45):
        env.step(settle_action)
        all_frames.append(_capture(uw.scene, camera))

    imageio.mimwrite(args.output, all_frames, fps=args.fps)
    duration = len(all_frames) / args.fps
    print(f"\n✓ {args.output} ({len(all_frames)} 帧, {duration:.1f}s)")
    env.close()
