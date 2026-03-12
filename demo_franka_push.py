"""
demo_franka_push.py — Franka 机器人推 Jenga 木块 Demo

运动规划器 + pd_joint_pos 控制:
- Even 层: 长轴 X → 从 -X 小端面沿 +X 推出
- Odd  层: 长轴 Y → 从 -Y 小端面沿 +Y 推出
- 使用 RecordEpisode 自动录制所有 env.step() 产生的帧
- 航点离散化实现机械臂与木块的同步推进

Usage:
    conda activate /opt/anaconda3/envs/skill
    python demo_franka_push.py --use-planner [--blocks 35 32 28 25 22] [--output demo_phase2.mp4]
"""
import argparse
import sys
import os
import time

import numpy as np
import torch
import gymnasium as gym
import sapien
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2quat

from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.agents.robots.panda.panda import Panda
from jenga_tower import BLOCK_L, BLOCK_W, BLOCK_H, NUM_BLOCKS
import realistic_jenga  # noqa: F401 — 触发 JengaTower-RealisticA-v1 环境注册

sys.path.append(os.path.join(os.path.dirname(__file__), 'examples/motionplanning/panda'))
try:
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
except ImportError:
    print("⚠️ 无法导入运动规划器，将使用简化的控制方法")
    PandaArmMotionPlanningSolver = None

_ROT_CACHE = {}

# 塔中心坐标 (与 jenga_tower.py 保持一致)
TOWER_CENTER_X = 0.2
TOWER_CENTER_Y = 0.0
# 机器人 base 在 x=-0.615, 即机器人从 -X 侧接近塔
ROBOT_BASE_X = -0.615


def _push_geometry(level, block_pos):
    """
    根据层奇偶 + 木块实际位置, 智能选择推动方向.

    原则: 沿长轴推出端面, 方向选择"从机器人侧推向远端"(避免穿越塔体).
      - Even 层: 长轴 X, 机器人在 -X 侧 → 从 -X 端面推向 +X (机器人侧接近)
      - Odd  层: 长轴 Y, 根据木块 Y 偏移选择:
          block_pos.y < tower_center → 从 -Y 端面推向 +Y (向外推)
          block_pos.y >= tower_center → 从 +Y 端面推向 -Y (向外推)
          y == center (中间块) → 默认推向 +Y

    返回 (push_dir, contact_offset, target_rot).
    push_dir: 推动方向单位向量
    contact_offset: 从木块中心到接触端面中心的偏移
    target_rot: EE 目标朝向 (scipy Rotation)
    """
    if level % 2 == 0:
        # Even 层: 长轴沿 X, 从 -X 端面推向 +X (机器人从 -X 侧接近, 不穿越塔)
        push_dir = np.array([1., 0., 0.])
        contact_offset = np.array([-BLOCK_L / 2, 0., 0.])
        approaching = np.array([1., 0., 0.])
    else:
        # Odd 层: 长轴沿 Y, 根据木块偏移选择推出方向
        block_y = block_pos[1] if len(block_pos) > 1 else 0.0
        y_offset = block_y - TOWER_CENTER_Y

        if y_offset < -1e-4:
            # 木块在 Y 负侧 → 从 +Y 端面推向 -Y (向外推, 远离塔中心)
            push_dir = np.array([0., -1., 0.])
            contact_offset = np.array([0., BLOCK_L / 2, 0.])
            approaching = np.array([0., -1., 0.])
        elif y_offset > 1e-4:
            # 木块在 Y 正侧 → 从 -Y 端面推向 +Y (向外推, 远离塔中心)
            push_dir = np.array([0., 1., 0.])
            contact_offset = np.array([0., -BLOCK_L / 2, 0.])
            approaching = np.array([0., 1., 0.])
        else:
            # 中间块: 默认从 -Y 端面推向 +Y
            push_dir = np.array([0., 1., 0.])
            contact_offset = np.array([0., -BLOCK_L / 2, 0.])
            approaching = np.array([0., 1., 0.])

    key = tuple(approaching)
    if key not in _ROT_CACHE:
        p = Panda.build_grasp_pose(approaching, np.array([0., 0., -1.]), np.zeros(3))
        q = p.q
        _ROT_CACHE[key] = R.from_quat([q[1], q[2], q[3], q[0]])

    return push_dir, contact_offset, _ROT_CACHE[key]


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


# ─── 维持动作辅助 ───

def _make_hold_action(env):
    """
    构造"维持当前关节姿态 + 夹爪闭合"的 action。
    pd_joint_pos 模式下, action 前 7 维 = 关节目标位置 (非 delta),
    必须填入当前 qpos, 否则全零会命令机械臂回零位导致剧烈运动。
    """
    uw = env.unwrapped
    qpos = uw.agent.robot.get_qpos()[0].cpu().numpy()
    arm_qpos = qpos[:7]
    gripper_val = -1.0  # 夹爪闭合
    action = np.concatenate([arm_qpos, [gripper_val]])
    return torch.tensor(action, dtype=torch.float32).unsqueeze(0)


def _hold_steps(env, n, action=None):
    """
    执行 n 步"悬停"，维持 PD 控制器在线。
    每步都用当前 qpos 构建 action, 防止关节漂移。
    """
    for _ in range(n):
        act = _make_hold_action(env) if action is None else action
        env.step(act)


# ─── EE 控制 (备用模式) ───

def _ee_action(agent, target_pos, target_rot):
    """pd_ee_delta_pose 7D action: [pos_norm(3), rot_norm(3), gripper(1)]"""
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


def _move_ee(env, target_pos, target_rot, max_steps=80, hold=3, thresh=0.005):
    """EE P-tracking 到目标位姿 (备用模式, 无需手动录帧)"""
    agent = env.unwrapped.agent
    for _ in range(max_steps):
        action = _ee_action(agent, target_pos, target_rot)
        env.step(action)
        if np.linalg.norm(agent.tcp.pose.p[0].cpu().numpy() - target_pos) < thresh:
            for _ in range(hold):
                env.step(action)
            break


# ─── 木块移除辅助 ───

def _force_remove_block(actor, device):
    """将木块直接传送到远处。"""
    zero3 = torch.zeros(1, 3, device=device)
    actor.set_pose(Pose.create_from_pq(torch.tensor([[999., 999., 999.]], device=device)))
    actor.set_linear_velocity(zero3)
    actor.set_angular_velocity(zero3)


def _slide_out_block(env, actor, push_dir, device, dist=0.25, steps=25):
    """
    让木块沿 push_dir 渐进滑出, 产生推出的视觉效果。
    滑出完成后传送到远处。
    """
    delta = push_dir * dist / steps
    current_p = actor.pose.p.clone()
    for i in range(steps):
        offset = torch.tensor(delta * (i + 1), device=device).unsqueeze(0)
        actor.set_pose(Pose.create_from_pq(current_p + offset))
        _hold_steps(env, 1)
    _force_remove_block(actor, device)


# ─── 近似可达点搜索 ───

def _find_reachable_pose(planner, base_pos, quat, push_dir, method="screw"):
    """
    在 base_pos 附近做空间网格搜索, 找到一个 Planner 可达的近似点。
    搜索优先级: 高度微调 → 沿推方向后退 → 侧移 (范围最小)。
    侧移范围控制在 ±1.5cm 以内, 避免机械臂离木块太远。
    返回可达的 sapien.Pose, 或 None。
    """
    up = np.array([0., 0., 1.])
    lateral = np.cross(push_dir, up)
    norm = np.linalg.norm(lateral)
    if norm < 1e-6:
        lateral = np.array([0., 1., 0.])
    else:
        lateral = lateral / norm

    z_offsets = [0., 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04]
    push_offsets = [0., -0.01, -0.02, -0.03, -0.04]
    lat_offsets = [0., 0.01, -0.01, 0.015, -0.015]

    for dz in z_offsets:
        for dp in push_offsets:
            for dl in lat_offsets:
                candidate_pos = base_pos + up * dz + push_dir * dp + lateral * dl
                candidate_pose = sapien.Pose(p=candidate_pos, q=quat)
                if method == "screw":
                    res = planner.move_to_pose_with_screw(candidate_pose, dry_run=True)
                else:
                    res = planner.move_to_pose_with_RRTConnect(candidate_pose, dry_run=True)
                if res != -1:
                    offset_str = f"dz={dz:+.3f} dp={dp:+.3f} dl={dl:+.3f}"
                    print(f"   🔍 找到近似可达点 ({offset_str})")
                    return candidate_pose
    return None


# ─── 单块推出 (使用运动规划器) ───

def animate_franka_pushing_with_planner(env, planner, selected_actor, steps=60, extraction_distance=0.15):
    """
    Franka 推木块全流程 (运动规划器版本).

    【重构核心】:
    - 阶段3采用航点离散化: 将推动距离等分为 steps 个微航点,
      每步先更新木块 Pose, 再让 Planner 跟踪微航点, 实现同步推进。
    - Planner 内部的 follow_path 会调用 env.step(),
      RecordEpisode 自动录制每一帧, 彻底解决丢帧问题。
    - 所有悬停阶段使用 env.step(hold_action) 替代 scene.step(),
      保证 PD 控制器始终在线。
    """
    uw = env.unwrapped
    blocks = uw.blocks
    device = selected_actor.device

    level = int(selected_actor.name.split("_")[1])
    block_pos = selected_actor.pose.p[0].cpu().numpy()
    push_dir, c_off, rot_scipy = _push_geometry(level, block_pos)

    contact = block_pos + c_off

    _freeze_all(blocks)
    _disable_collision(selected_actor)

    # 每次推动前强制确保夹爪闭合 (-1 = 闭合, 1 = 打开)
    planner.gripper_state = -1

    standoff_dist = 0.04

    quat_xyzw = rot_scipy.as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # ── 多角度搜索: 用 dry_run 测试 pre_push_pose 可达性 ──
    push_pose_base = sapien.Pose(p=contact, q=quat_wxyz)
    pre_base_pos = contact - push_dir * standoff_dist

    angles = np.arange(0, np.pi, np.pi / 6)
    angles = np.concatenate([angles, -angles[1:]])

    valid_push_pose = None
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        candidate = push_pose_base * delta_pose
        pre_candidate = sapien.Pose(p=pre_base_pos, q=candidate.q)
        res = planner.move_to_pose_with_screw(pre_candidate, dry_run=True)
        if res != -1:
            valid_push_pose = candidate
            break

    # 备选: 翻转推动方向（从对面推）
    if valid_push_pose is None:
        flip_push_dir = -push_dir
        flip_c_off = -c_off
        flip_contact = block_pos + flip_c_off
        flip_approaching = -np.array(push_dir)
        flip_closing = np.array([0., 0., -1.])
        flip_grasp = Panda.build_grasp_pose(flip_approaching, flip_closing, flip_contact)
        flip_q = flip_grasp.q

        flip_pre_pos = flip_contact - flip_push_dir * standoff_dist
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            candidate = sapien.Pose(p=flip_contact, q=flip_q) * delta_pose
            pre_candidate = sapien.Pose(p=flip_pre_pos, q=candidate.q)
            res = planner.move_to_pose_with_screw(pre_candidate, dry_run=True)
            if res != -1:
                valid_push_pose = candidate
                push_dir = flip_push_dir
                c_off = flip_c_off
                contact = flip_contact
                print(f"   ↻ 使用翻转方向推动 (从对面推)")
                break

    if valid_push_pose is None:
        valid_push_pose = push_pose_base

    push_pose = valid_push_pose

    try:
        # === 阶段1: 移动到接触点前方的预备位置 ===
        pre_push_pos = contact - push_dir * standoff_dist
        pre_push_pose = sapien.Pose(p=pre_push_pos, q=push_pose.q)

        print(f"   → Phase 1: 移动到接触点前方 "
              f"(pos=[{pre_push_pos[0]:.3f}, {pre_push_pos[1]:.3f}, {pre_push_pos[2]:.3f}])")

        reached_phase1 = False
        result = planner.move_to_pose_with_RRTConnect(pre_push_pose)
        if result != -1:
            reached_phase1 = True
        else:
            result = planner.move_to_pose_with_screw(pre_push_pose)
            if result != -1:
                reached_phase1 = True

        if not reached_phase1:
            print(f"   ⚠️ 精确点不可达, 搜索近似可达点...")
            approx = _find_reachable_pose(planner, pre_push_pos, push_pose.q, push_dir, method="screw")
            if approx is None:
                approx = _find_reachable_pose(planner, pre_push_pos, push_pose.q, push_dir, method="rrt")
            if approx is not None:
                result = planner.move_to_pose_with_screw(approx)
                if result == -1:
                    result = planner.move_to_pose_with_RRTConnect(approx)
                if result != -1:
                    reached_phase1 = True
                    pre_push_pos = approx.p

        if not reached_phase1:
            # 即使预备点不可达, 也尝试移动到木块附近的高空位置, 做出"接近"姿态
            print(f"   ⚠️ 预备点不可达, 尝试移动到木块上方...")
            above_pos = block_pos.copy()
            above_pos[2] += 0.15
            above_pose = sapien.Pose(p=above_pos, q=push_pose.q)
            result = planner.move_to_pose_with_screw(above_pose)
            if result == -1:
                result = planner.move_to_pose_with_RRTConnect(above_pose)
            if result == -1:
                approx = _find_reachable_pose(planner, above_pos, push_pose.q, push_dir, method="rrt")
                if approx is not None:
                    planner.move_to_pose_with_RRTConnect(approx)

        _hold_steps(env, 15)

        # === 阶段2: 前进到接触点 ===
        print(f"   → Phase 2: 前进到接触点 "
              f"(pos=[{contact[0]:.3f}, {contact[1]:.3f}, {contact[2]:.3f}])")

        result = planner.move_to_pose_with_screw(push_pose)
        if result == -1:
            approx = _find_reachable_pose(planner, contact, push_pose.q, push_dir, method="screw")
            if approx is not None:
                result = planner.move_to_pose_with_screw(approx)
            if result == -1:
                # 最后尝试用 RRT 到接触点
                result = planner.move_to_pose_with_RRTConnect(push_pose)
            if result == -1:
                print(f"   ⚠️ 接触点不可达, 从当前位置继续推动")

        _hold_steps(env, 10)

        # === 阶段3: 航点离散化同步推动木块 ===
        print(f"   → Phase 3: 同步推动木块 ({extraction_distance}m, {steps} 航点)")

        initial_p = selected_actor.pose.p.clone()
        delta_per_step = push_dir * extraction_distance / steps

        for i in range(steps):
            step_offset = torch.tensor(
                delta_per_step * (i + 1), device=device
            ).unsqueeze(0)
            target_p = initial_p + step_offset
            selected_actor.set_pose(Pose.create_from_pq(target_p))

            waypoint_pos = contact + push_dir * extraction_distance * (i + 1) / steps
            waypoint_pose = sapien.Pose(p=waypoint_pos, q=push_pose.q)

            result = planner.move_to_pose_with_screw(waypoint_pose)
            if result == -1:
                _hold_steps(env, 3)

        # === 阶段4: 木块继续沿推方向滑出视野 ===
        print(f"   → Phase 4: 木块滑出")
        _slide_out_block(env, selected_actor, push_dir, device)

        # === 阶段5: 机械臂安全撤离 (塔保持冻结, 避免碰撞干扰) ===
        print(f"   → Phase 5: 机械臂撤回")

        # 5a: 沿推方向反向撤回, 距离加大到 standoff_dist * 2
        retreat_dist = standoff_dist * 2.5
        retreat_pos = contact - push_dir * retreat_dist
        retreat_pose = sapien.Pose(p=retreat_pos, q=push_pose.q)
        result = planner.move_to_pose_with_screw(retreat_pose)
        if result == -1:
            result = planner.move_to_pose_with_RRTConnect(retreat_pose)
        if result == -1:
            approx = _find_reachable_pose(planner, retreat_pos, push_pose.q, push_dir, method="rrt")
            if approx is not None:
                planner.move_to_pose_with_RRTConnect(approx)

        _hold_steps(env, 5)

        # 5b: 先提升到安全高度, 完全脱离塔体范围
        lift_pos = retreat_pos.copy()
        lift_pos[2] += 0.12
        lift_pose = sapien.Pose(p=lift_pos, q=push_pose.q)
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            result = planner.move_to_pose_with_RRTConnect(lift_pose)
        if result == -1:
            approx = _find_reachable_pose(planner, lift_pos, push_pose.q, push_dir, method="rrt")
            if approx is not None:
                planner.move_to_pose_with_RRTConnect(approx)

        _hold_steps(env, 5)

        print(f"   ✓ 木块推出成功")

    except Exception as e:
        print(f"   ❌ 推动过程出错: {e}")
        import traceback
        traceback.print_exc()
        _slide_out_block(env, selected_actor, push_dir, device)
        print(f"   ✓ 异常兜底: 木块已渐进移除")


# ─── 单块推出 (备用 EE 方法, 无 Planner) ───

def animate_franka_pushing(env, selected_actor, steps=60, extraction_distance=0.15):
    """
    Franka 推木块全流程 (pd_ee_delta_pose 备用版本).
    无需手动录帧, RecordEpisode 自动录制所有 env.step() 的帧。
    """
    uw = env.unwrapped
    agent, blocks = uw.agent, uw.blocks
    device = selected_actor.device

    level = int(selected_actor.name.split("_")[1])
    block_pos = selected_actor.pose.p[0].cpu().numpy()
    push_dir, c_off, rot = _push_geometry(level, block_pos)

    contact = block_pos + c_off
    pre_contact = contact - push_dir * 0.02

    _freeze_all(blocks)
    _disable_collision(selected_actor)

    # Phase 1: Approach
    standoff = pre_contact - push_dir * 0.05
    standoff[2] += 0.06
    _move_ee(env, standoff, rot, max_steps=80)
    _move_ee(env, pre_contact, rot, max_steps=40)
    _move_ee(env, contact, rot, max_steps=20, hold=2)

    # Phase 2: Synchronized push
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

    # Phase 3: Disappear
    zero3 = torch.zeros(1, 3, device=device)
    selected_actor.set_pose(
        Pose.create_from_pq(torch.tensor([[999., 999., 999.]], device=device))
    )
    selected_actor.set_linear_velocity(zero3)
    selected_actor.set_angular_velocity(zero3)

    # Phase 4: Retract
    pullback = contact + push_dir * (extraction_distance - 0.05)
    pullback[2] += 0.06
    _move_ee(env, pullback, rot, max_steps=30, hold=0)
    hover = pullback.copy()
    hover[2] += 0.10
    _move_ee(env, hover, rot, max_steps=30, hold=0)




# ─── 主程序 ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka Push Jenga Demo")
    parser.add_argument("--blocks", type=int, nargs="+",
                        default=[35, 32, 28, 25, 22],
                        help="指定推动的 block indices 顺序")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--distance", type=float, default=0.15)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="demo_phase2.mp4")
    parser.add_argument("--save-video", action="store_true",
                        help="启用视频录制 (使用 RecordEpisode)")
    parser.add_argument("--use-planner", action="store_true",
                        help="使用运动规划器进行更精确的控制")
    args = parser.parse_args()

    # 只要指定了 --output 或 --save-video，就启用录制
    enable_recording = args.save_video or (args.output != "demo_phase2.mp4")

    if args.use_planner and PandaArmMotionPlanningSolver is not None:
        control_mode = "pd_joint_pos"
        print("✓ 使用运动规划器模式 (pd_joint_pos)")
    else:
        control_mode = "pd_ee_delta_pose"
        print("✓ 使用末端执行器控制模式 (pd_ee_delta_pose)")

    env = gym.make(
        "JengaTower-RealisticA-v1",
        obs_mode="state",
        control_mode=control_mode,
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
        max_episode_steps=100000,
    )

    # 【重构】用 RecordEpisode 包裹 env，自动录制所有 env.step() 的帧。
    # 不再需要手动 _capture + frames 列表 + imageio.mimwrite。
    # Planner 内部的 follow_path、open/close_gripper 都调 env.step(),
    # 被 wrapper 自动捕获, 彻底解决丢帧问题。
    output_dir = os.path.dirname(args.output) or "."
    base_name = os.path.splitext(os.path.basename(args.output))[0]
    expected_path = os.path.join(output_dir, f"{base_name}.mp4")
    if os.path.exists(expected_path):
        video_name = f"{base_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        video_name = base_name

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        save_trajectory=False,
        save_video=True,
        video_fps=args.fps,
        save_on_reset=False,  # 不在 reset 时自动保存, 我们在 close 时统一保存
        max_steps_per_video=None,
        trajectory_name=video_name,
    )

    obs, _ = env.reset(seed=42)
    uw = env.unwrapped

    # 初始化运动规划器
    planner = None
    if args.use_planner and PandaArmMotionPlanningSolver is not None:
        print("初始化运动规划器...")
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=uw.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=0.8,
            joint_acc_limits=0.8,
        )
        # 直接设置 gripper_state 为闭合, 不调用 close_gripper() 避免额外步数
        # 后续所有 follow_path / close_gripper / open_gripper 都基于此状态
        planner.gripper_state = -1
        print("✓ 运动规划器初始化完成 (夹爪已设为闭合)")

    # 初始稳定: 先冻结全塔防止物理模拟晃动, 再闭合夹爪
    _freeze_all(uw.blocks)
    if planner is not None:
        planner.close_gripper()
    _hold_steps(env, 5)

    # ── 选块 (直接使用指定顺序) ──
    valid_blocks = []
    for bi in args.blocks:
        if bi < len(uw.blocks) and uw.blocks[bi].pose.p[0, 2].item() < 100:
            valid_blocks.append(bi)
        else:
            print(f"⚠️ 木块 {bi} 在残局中已被移除或索引无效, 跳过")
    extract_order = valid_blocks
    print(f"推动顺序: {extract_order}")

    # Reset 到干净状态
    obs, _ = env.reset(seed=42)

    # reset 后 Planner 对象不会被重建, gripper_state 保持 CLOSED
    # 但物理上夹爪回到了打开状态, 需要重新闭合
    # 先冻结全塔, 防止闭合夹爪过程中塔晃动
    _freeze_all(uw.blocks)
    if planner is not None:
        planner.close_gripper()
    _hold_steps(env, 5)

    print("\n" + "=" * 50)
    print("  录制 Franka Push Demo...")
    print("=" * 50)

    removed = set()

    # Intro: 维持当前姿态 (塔已冻结, 不会晃动)
    _hold_steps(env, 30)

    for seq, block_idx in enumerate(extract_order):
        lv, pi = divmod(block_idx, 3)
        print(f"  [{seq + 1}/{len(extract_order)}] 推出 block #{block_idx} L{lv}[{pi}]")

        if planner is not None:
            animate_franka_pushing_with_planner(
                env, planner, uw.blocks[block_idx],
                steps=args.steps,
                extraction_distance=args.distance,
            )
        else:
            animate_franka_pushing(
                env, uw.blocks[block_idx],
                steps=args.steps,
                extraction_distance=args.distance,
            )
        removed.add(block_idx)

        # 解冻前先维持冻结几步, 让机械臂完全稳定在安全位置
        _hold_steps(env, 10)
        # 解冻剩余木块, 让物理 settle
        _unfreeze_remaining(uw.blocks, removed)
        _hold_steps(env, 40)
        # settle 后重新冻结, 防止下一块推动过程中塔体晃动
        _freeze_all(uw.blocks)

    # Outro
    _hold_steps(env, 45)

    print(f"\n✓ 录制完成, 视频保存中...")

    if planner is not None:
        planner.close()

    # save_on_reset=False 时, close() 不会自动 flush 视频; 传入 name 避免总是保存为 0.mp4
    env.flush_video(name=video_name)
    env.close()

    expected_path = os.path.join(output_dir, f"{video_name}.mp4")
    if os.path.exists(expected_path):
        if video_name != base_name:
            print(f"✓ 视频已保存: {expected_path} (未覆盖已有文件)")
        elif os.path.abspath(expected_path) != os.path.abspath(args.output):
            import shutil
            shutil.move(expected_path, args.output)
            print(f"✓ 视频已保存: {args.output}")
        else:
            print(f"✓ 视频已保存: {args.output}")
    else:
        print(f"✓ 视频应已保存到 {output_dir}/ 目录下")
