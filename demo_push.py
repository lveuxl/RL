"""
demo_push.py — 指定 (layer, position) 让 Panda 推出 Jenga 木块

流程:
  1. 冻结除目标块外的所有木块
  2. 机器人沿安全路径移到目标木块端面
  3. 目标块也设为 kinematic, 用 set_pose 匀速滑出, TCP 同步跟随
  4. 目标块消失(移到远处)
  5. 机器人退回 rest, 解冻所有木块

Usage:
    conda activate /opt/anaconda3/envs/skill
    python demo_push.py --layer 5 --pos 1
"""
import argparse
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation

import jenga_tower
from mani_skill.utils.structs.pose import Pose
from jenga_tower import BLOCK_L, BLOCK_W, BLOCK_H, NUM_LEVELS

STANDOFF = 0.02
SLIDE_DIST = 0.15     # 木块滑出总距离
SLIDE_STEPS = 60      # 滑出帧数
CENTER_X, CENTER_Y = 0.2, 0.0
BASE_OFFSET = np.array([0.615, 0.0, 0.0])
Y_SAFE = -0.16

PUSH_EULER = Rotation.from_matrix(
    np.stack([
        np.cross([1, 0, 0], [0, 0, -1]),
        [1, 0, 0],
        [0, 0, -1],
    ], axis=1)
).as_euler("XYZ").astype(np.float32)


def get_block_geometry(layer, pos, gap_w=0.005, gap_h=0.0005):
    """返回 (face_x_world, by_world, z_world, push_dir_world)"""
    z = layer * (BLOCK_H + gap_h) + BLOCK_H / 2
    offset = (pos - 1) * (BLOCK_W + gap_w)
    if layer % 2 == 0:
        face_x = CENTER_X - BLOCK_L / 2
        by = CENTER_Y + offset
    else:
        face_x = CENTER_X + offset - BLOCK_W / 2
        by = CENTER_Y
    # 统一从 -X 推向 +X
    return face_x, by, z, np.array([1.0, 0.0, 0.0])


def get_current_pose(env):
    ctrl = env.unwrapped.agent.controller.controllers["arm"]
    ee = ctrl.ee_pose_at_base
    p = ee.p[0].cpu().numpy()
    q = ee.q[0].cpu().numpy()
    euler = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("XYZ")
    return p, euler


def make_action(pos, euler, gripper=0.04):
    return np.concatenate([pos, euler, [gripper]]).astype(np.float32)[None]


def move_to(env, base_pos, euler, steps=80):
    action = make_action(base_pos, euler)
    for _ in range(steps):
        env.step(action)
        env.render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--pos", type=int, required=True, help="0=左 1=中 2=右")
    parser.add_argument("--settle", type=int, default=30)
    args = parser.parse_args()

    assert 0 <= args.layer < NUM_LEVELS and args.pos in (0, 1, 2)
    block_idx = args.layer * 3 + args.pos
    face_x, by, z, push_dir = get_block_geometry(args.layer, args.pos)

    # 机器人目标: 端面外侧 STANDOFF 处 (基座系)
    contact_base = np.array([face_x - STANDOFF, by, z]) + BASE_OFFSET
    side_entry = np.array([contact_base[0], Y_SAFE, contact_base[2]])

    print(f"目标: block #{block_idx}  L{args.layer}[{args.pos}]")
    print(f"  contact (base): ({contact_base[0]:.3f}, {contact_base[1]:.3f}, {contact_base[2]:.3f})")

    env = gym.make(
        "JengaTower-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend="cpu",
        robot_uids="panda",
        control_mode="pd_ee_pose",
    )
    obs, _ = env.reset(seed=42)
    uw = env.unwrapped

    rest_p, rest_euler = get_current_pose(env)
    hold = make_action(rest_p, rest_euler)
    for _ in range(args.settle):
        env.step(hold)
    env.render()

    # ── 冻结所有木块 ──
    zero3 = torch.zeros(1, 3, device=uw.device)
    for blk in uw.blocks:
        blk._bodies[0].kinematic = True
        blk.set_linear_velocity(zero3)
        blk.set_angular_velocity(zero3)
    print("已冻结全部木块")

    # ── 机器人移动到接触位置 ──
    side_rest = rest_p.copy(); side_rest[1] = Y_SAFE

    print("\n机器人就位:")
    print("  [1] 侧移")
    move_to(env, side_rest, PUSH_EULER, steps=60)
    print("  [2] 前进")
    move_to(env, side_entry, PUSH_EULER, steps=60)
    print("  [3] 滑入对齐端面")
    move_to(env, contact_base, PUSH_EULER, steps=60)

    # ── 木块匀速滑出 + TCP 同步跟随 ──
    print(f"  [4] 木块滑出 ({SLIDE_STEPS} 帧, {SLIDE_DIST}m)")
    target_block = uw.blocks[block_idx]
    block_p0 = target_block.pose.p.clone()  # (1, 3)
    block_q0 = target_block.pose.q.clone()  # (1, 3) or raw
    delta = torch.tensor([push_dir * (SLIDE_DIST / SLIDE_STEPS)], device=uw.device)

    tcp_p0 = contact_base.copy()

    for step in range(SLIDE_STEPS):
        # 木块匀速前进
        new_p = block_p0 + delta * (step + 1)
        target_block.set_pose(Pose.create_from_pq(new_p))

        # TCP 同步跟随
        tcp_p = tcp_p0 + push_dir * (SLIDE_DIST / SLIDE_STEPS) * (step + 1)
        action = make_action(tcp_p, PUSH_EULER)
        env.step(action)
        env.render()

    # ── 木块消失 ──
    far = torch.tensor([[999.0, 999.0, 999.0]], device=uw.device)
    target_block.set_pose(Pose.create_from_pq(far))
    target_block.set_linear_velocity(zero3)
    target_block.set_angular_velocity(zero3)
    print("  木块已移除")

    # ── 机器人退回 ──
    print("  [5] 退回")
    move_to(env, contact_base, PUSH_EULER, steps=30)
    move_to(env, side_entry, PUSH_EULER, steps=60)
    move_to(env, side_rest, PUSH_EULER, steps=60)
    move_to(env, rest_p, rest_euler, steps=60)

    # ── 解冻 ──
    for blk in uw.blocks:
        blk._bodies[0].kinematic = False
    print("已解冻, 物理模拟恢复")

    hold = make_action(*get_current_pose(env))
    while True:
        env.step(hold)
        try:
            env.render()
        except (TypeError, AttributeError):
            break

    env.close()


if __name__ == "__main__":
    main()
