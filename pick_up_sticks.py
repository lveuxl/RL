"""
PickUpSticks-v1: 挑棒子 (Pick-up Sticks) 仿真环境 (ManiSkill BaseEnv)

Task Description:
    桌面上散落着 N 根彩色细棒 (圆柱体)。RL 智能体需要学习 **最佳挑取顺序**，
    每一步选择一根棒子 ID，环境内部通过启发式宏动作完成抓取闭环。

Architecture:
    - 离散动作空间 Discrete(N)：选择要挑起的棒子编号
    - 宏动作 (Macro-action)：_execute_heuristic_grasp(stick_id) 内部完成
      多步底层仿真，RL 不控制连续轨迹
    - 物理沉降 + 不可见漏斗：棒子从高处错层生成，经漏斗聚拢后自然堆叠，
      沉降完成后移除漏斗

Usage:
    conda activate <your_env>
    python pick_up_sticks.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from collections import deque

import numpy as np
import sapien
import sapien.physx
import sapien.render
import torch
import gymnasium as gym

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

# ═══════════════════════════════════════════════════════════
#  棒子几何与物理参数
# ═══════════════════════════════════════════════════════════
STICK_RADIUS = 0.004       # 棒子半径 4 mm
STICK_HALF_LENGTH = 0.075  # 棒子半长 7.5 cm (全长 15 cm)
NUM_STICKS = 15            # 默认棒子数量
STICK_DENSITY = 800        # kg/m³ (略重于木头，便于沉降)

# 棒子颜色池 (经典挑棒子游戏的 5 种颜色)
STICK_COLORS = [
    [1.0, 0.0, 0.0, 1.0],   # Red
    [0.0, 0.0, 1.0, 1.0],   # Blue
    [0.0, 0.8, 0.0, 1.0],   # Green
    [1.0, 1.0, 0.0, 1.0],   # Yellow
    [0.6, 0.0, 0.8, 1.0],   # Purple
]

# 漏斗参数
FUNNEL_HALF_THICKNESS = 0.005   # 漏斗壁厚 (半厚度)
FUNNEL_HEIGHT = 0.15            # 漏斗壁高度
FUNNEL_TOP_HALF_WIDTH = 0.12    # 漏斗顶部半宽 (开口大)
FUNNEL_BOTTOM_HALF_WIDTH = 0.04 # 漏斗底部半宽 (出口小)

# 物理沉降步数
SETTLE_STEPS = 500


# ═══════════════════════════════════════════════════════════
#  辅助函数: Dependency Graph (谁压在谁上面)
# ═══════════════════════════════════════════════════════════
def get_dependency_graph(scene, sticks: list, force_threshold_ratio: float = 0.05):
    """
    从 SAPIEN 物理场景提取棒子之间的依赖关系图 (Dependency Graph)。

    通过分析接触力判断 "谁压在谁上面":
      - 如果 stick_i 对 stick_j 施加了向上的法向力 (超过阈值)，
        则认为 stick_i 支撑 stick_j，即 stick_j 依赖 stick_i。

    Args:
        scene:  ManiSkillScene，提供 get_contacts() / timestep
        sticks: list[Actor]，场景中所有棒子
        force_threshold_ratio: 判定支撑的力阈值占单根棒子重力的比例

    Returns:
        dict:
          "heights":          list[float]      — 每根棒子重心的 Z 高度 (m)
          "dependency_matrix": list[list[int]] — D[i][j]=1 表示 stick_j 压在 stick_i 上
                                                 (即移除 stick_i 可能影响 stick_j)
          "on_top_count":     list[int]        — 每根棒子上方被压着的棒子数量
    """
    n = len(sticks)
    dt = scene.timestep

    # 单根棒子的体积和重力
    vol = np.pi * STICK_RADIUS ** 2 * (2 * STICK_HALF_LENGTH)
    single_weight = vol * STICK_DENSITY * 9.81
    threshold = force_threshold_ratio * single_weight

    # entity → 棒子索引映射
    entity_to_idx = {s._bodies[0].entity: i for i, s in enumerate(sticks)}

    heights = [float(s.pose.p[0, 2]) for s in sticks]

    dependency_matrix = [[0] * n for _ in range(n)]

    for contact in scene.get_contacts():
        e0, e1 = contact.bodies[0].entity, contact.bodies[1].entity
        if e0 not in entity_to_idx or e1 not in entity_to_idx:
            continue
        idx0, idx1 = entity_to_idx[e0], entity_to_idx[e1]

        # PhysX 约定: point.impulse 施加在 bodies[0] 上
        fz = sum(pt.impulse[2] for pt in contact.points) / dt

        if -fz > threshold:
            # bodies[0] 向上推 bodies[1] → stick_0 支撑 stick_1
            dependency_matrix[idx0][idx1] = 1
        if fz > threshold:
            # bodies[1] 向上推 bodies[0] → stick_1 支撑 stick_0
            dependency_matrix[idx1][idx0] = 1

    # 统计每根棒子上方被压着的棒子数量
    on_top_count = [sum(row) for row in dependency_matrix]

    return {
        "heights": heights,
        "dependency_matrix": dependency_matrix,
        "on_top_count": on_top_count,
    }


def get_removal_difficulty(dependency_matrix: list, heights: list) -> np.ndarray:
    """
    基于依赖图计算每根棒子的 "移除难度" 分数。

    难度越高 → 该棒子上方压着越多其他棒子，移除风险越大。

    Args:
        dependency_matrix: (N, N) D[i][j]=1 表示 stick_i 支撑 stick_j
        heights:           (N,) 每根棒子重心 Z 高度

    Returns:
        difficulty: np.ndarray (N,) — 移除难度分数 ∈ [0, 1]
    """
    n = len(heights)
    A = np.asarray(dependency_matrix, dtype=int)

    # BFS: 对每根棒子 i，求其传递支撑闭包 (移除 i 后可能受影响的所有棒子)
    children = [np.where(A[i] > 0)[0] for i in range(n)]

    cascade_sizes = []
    for i in range(n):
        visited = set(children[i].tolist())
        queue = deque(visited)
        while queue:
            for c in children[queue.popleft()]:
                if c not in visited:
                    visited.add(c)
                    queue.append(c)
        cascade_sizes.append(len(visited))

    cascade_sizes = np.array(cascade_sizes, dtype=float)
    # Normalize to [0, 1]
    if cascade_sizes.max() > 0:
        difficulty = cascade_sizes / cascade_sizes.max()
    else:
        difficulty = np.zeros(n)

    return difficulty


# ═══════════════════════════════════════════════════════════
#  PickUpSticks-v1 环境主体
# ═══════════════════════════════════════════════════════════
@register_env("PickUpSticks-v1", max_episode_steps=100)
class PickUpSticksEnv(BaseEnv):
    """
    **Task Description:**
    挑棒子 (Pick-up Sticks) — 桌面上散落 N 根彩色细棒，
    RL 智能体学习最佳挑取顺序 (离散宏动作)。

    **Action Space:**
    Discrete(N) — 选择要挑起的棒子编号。
    环境内部通过 _execute_heuristic_grasp() 完成抓取闭环。

    **Observation:**
    Privileged State — 所有棒子的 6DoF 姿态 + 移除状态掩码。

    **Success Conditions:**
    所有棒子均被成功挑起 (removed_mask 全为 True)。
    """

    SUPPORTED_ROBOTS = ["panda", "panda_stick"]
    agent: Panda
    num_sticks = NUM_STICKS

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_sticks: int = NUM_STICKS,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_sticks = num_sticks
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ─────────────────────────────────────────────────────
    #  仿真配置: 高频 + 高迭代 + 防爆
    # ─────────────────────────────────────────────────────
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=500,          # 500 Hz 物理仿真频率 (防穿模)
            control_freq=20,       # 20 Hz 控制频率
            scene_config=SceneConfig(
                solver_position_iterations=100,   # 高位置求解迭代
                solver_velocity_iterations=100,   # 高速度求解迭代
                bounce_threshold=2.0,
                enable_enhanced_determinism=True,  # 增强确定性
            ),
        )

    # ─────────────────────────────────────────────────────
    #  相机配置
    # ─────────────────────────────────────────────────────
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.5], target=[0.0, 0.0, 0.1]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.6, 0.5, 0.5], target=[0.0, 0.0, 0.15]
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ─────────────────────────────────────────────────────
    #  动作空间: 离散 Discrete(N)
    # ─────────────────────────────────────────────────────
    @property
    def action_space(self):
        """离散动作空间: 选择要挑起的棒子编号 [0, N)"""
        return gym.spaces.Discrete(self.num_sticks)

    @action_space.setter
    def action_space(self, value):
        """允许 BaseEnv 在初始化时设置 action_space (兼容父类)"""
        # 我们覆盖了 getter，所以 setter 只做占位，不实际存储
        pass

    @property
    def single_action_space(self):
        return gym.spaces.Discrete(self.num_sticks)

    @single_action_space.setter
    def single_action_space(self, value):
        pass

    # ─────────────────────────────────────────────────────
    #  加载机器人
    # ─────────────────────────────────────────────────────
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ─────────────────────────────────────────────────────
    #  加载场景: 桌子 + 棒子 + 不可见漏斗
    # ─────────────────────────────────────────────────────
    def _load_scene(self, options: dict):
        # ── 1. 桌面场景 ──
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # ── 2. 棒子物理材质 (restitution=0, 防弹跳) ──
        self.stick_phys_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,  # 恢复系数严格为 0
        )

        # ── 3. 创建 N 根棒子 ──
        rng = np.random.default_rng(42)
        self.sticks: List[sapien.Entity] = []

        for i in range(self.num_sticks):
            color = STICK_COLORS[i % len(STICK_COLORS)]
            builder = self.scene.create_actor_builder()

            # 碰撞体
            builder.add_cylinder_collision(
                radius=STICK_RADIUS,
                half_length=STICK_HALF_LENGTH,
                material=self.stick_phys_mat,
                density=STICK_DENSITY,
            )
            # 视觉体
            builder.add_cylinder_visual(
                radius=STICK_RADIUS,
                half_length=STICK_HALF_LENGTH,
                material=sapien.render.RenderMaterial(base_color=color),
            )

            # 初始姿态: 在漏斗上方错层生成 (避免初始穿模)
            # Z 方向每根棒子间隔 3cm，从 0.25m 开始
            init_z = 0.25 + i * 0.03
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, init_z])

            stick = builder.build(name=f"stick_{i}")
            self.sticks.append(stick)

        # ── 4. 构建不可见漏斗 (4 个倾斜的静态 Box 围成倒锥形) ──
        # 漏斗中心在桌面上方 (0, 0, 0)
        # 每面墙: 从 (top_half_width, funnel_height) 倾斜到 (bottom_half_width, 0)
        self.funnel_walls: List[sapien.Entity] = []
        self._build_funnel()

    def _build_funnel(self):
        """
        构建不可见漏斗: 4 个倾斜的静态 Box 碰撞体围成倒锥形。
        漏斗仅有碰撞体，无视觉体 (不可见)。
        """
        funnel_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.3,
            dynamic_friction=0.2,
            restitution=0.0,
        )

        # 漏斗壁的几何参数
        wall_length = np.sqrt(
            (FUNNEL_TOP_HALF_WIDTH - FUNNEL_BOTTOM_HALF_WIDTH) ** 2
            + FUNNEL_HEIGHT ** 2
        )
        tilt_angle = np.arctan2(
            FUNNEL_TOP_HALF_WIDTH - FUNNEL_BOTTOM_HALF_WIDTH,
            FUNNEL_HEIGHT,
        )

        # 壁的中心高度和水平偏移
        mid_z = FUNNEL_HEIGHT / 2
        mid_offset = (FUNNEL_TOP_HALF_WIDTH + FUNNEL_BOTTOM_HALF_WIDTH) / 2

        # 壁的半尺寸: 长方向 = wall_length/2, 宽方向 = 漏斗沿轴向的半长度, 厚度
        wall_half_h = wall_length / 2
        wall_half_w = STICK_HALF_LENGTH + 0.03  # 比棒子长一些，确保兜住
        wall_half_t = FUNNEL_HALF_THICKNESS

        # 4 面墙的配置: (位置偏移方向, 旋转轴, 旋转角度)
        # +X 面, -X 面, +Y 面, -Y 面
        configs = [
            # (position_offset, quaternion)
            # +X wall: 沿 X 正方向偏移, 绕 Y 轴倾斜
            ([mid_offset, 0, mid_z],
             self._quat_from_axis_angle([0, 1, 0], -tilt_angle)),
            # -X wall: 沿 X 负方向偏移, 绕 Y 轴反向倾斜
            ([-mid_offset, 0, mid_z],
             self._quat_from_axis_angle([0, 1, 0], tilt_angle)),
            # +Y wall: 沿 Y 正方向偏移, 绕 X 轴倾斜
            ([0, mid_offset, mid_z],
             self._quat_from_axis_angle([1, 0, 0], tilt_angle)),
            # -Y wall: 沿 Y 负方向偏移, 绕 X 轴反向倾斜
            ([0, -mid_offset, mid_z],
             self._quat_from_axis_angle([1, 0, 0], -tilt_angle)),
        ]

        for idx, (pos, quat) in enumerate(configs):
            builder = self.scene.create_actor_builder()

            # 根据墙面朝向选择合适的 half_size
            if idx < 2:
                # +X / -X 墙: 厚度沿 X, 宽度沿 Y, 高度沿 Z
                half_size = [wall_half_t, wall_half_w, wall_half_h]
            else:
                # +Y / -Y 墙: 宽度沿 X, 厚度沿 Y, 高度沿 Z
                half_size = [wall_half_w, wall_half_t, wall_half_h]

            builder.add_box_collision(
                half_size=half_size,
                material=funnel_mat,
            )
            # 不添加视觉体 → 漏斗不可见

            builder.initial_pose = sapien.Pose(p=pos, q=quat)
            wall = builder.build_static(name=f"funnel_wall_{idx}")
            self.funnel_walls.append(wall)

    @staticmethod
    def _quat_from_axis_angle(axis: list, angle: float) -> list:
        """轴角 → 四元数 (wxyz 格式, SAPIEN 约定)"""
        axis = np.array(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        half = angle / 2.0
        w = np.cos(half)
        xyz = axis * np.sin(half)
        return [w, xyz[0], xyz[1], xyz[2]]

    # ─────────────────────────────────────────────────────
    #  Episode 初始化: 随机撒棒 + 物理沉降 + 移除漏斗
    # ─────────────────────────────────────────────────────
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ── 初始化移除状态掩码 (False = 未移除) ──
            self.removed_mask = torch.zeros(
                (b, self.num_sticks), dtype=torch.bool, device=self.device
            )
            # 记录成功挑起的棒子数
            self.num_removed = torch.zeros(b, dtype=torch.long, device=self.device)

            # ── 随机生成棒子初始姿态 ──
            # 棒子在漏斗上方错层生成，带随机 XY 偏移和随机旋转
            for i in range(self.num_sticks):
                xyz = torch.zeros((b, 3))
                # XY 随机偏移 ±3cm (在漏斗口范围内)
                xyz[:, :2] = (torch.rand((b, 2)) - 0.5) * 0.06
                # Z 方向错层: 从 0.20m 开始，每根间隔 2.5cm
                xyz[:, 2] = 0.20 + i * 0.025

                # 随机旋转: 随机 roll + pitch + yaw
                # 生成随机四元数 (简化: 仅随机绕 Z 轴旋转 + 小幅 XY 倾斜)
                yaw = torch.rand(b) * 2 * np.pi  # 绕 Z 轴 [0, 2π)
                pitch = (torch.rand(b) - 0.5) * 0.6  # 绕 Y 轴 ±0.3 rad
                roll = (torch.rand(b) - 0.5) * 0.6   # 绕 X 轴 ±0.3 rad

                # 简化四元数: 先 yaw 再小幅 pitch/roll
                # q = qz * qy * qx (ZYX 欧拉角)
                qs = self._euler_to_quat_batch(roll, pitch, yaw)

                self.sticks[i].set_pose(Pose.create_from_pq(xyz, qs))
                # 清零速度
                self.sticks[i].set_linear_velocity(torch.zeros(b, 3))
                self.sticks[i].set_angular_velocity(torch.zeros(b, 3))

            # ── 设置棒子阻尼 (使其快速静止) ──
            for stick in self.sticks:
                stick.set_linear_damping(5.0)   # 较大线性阻尼
                stick.set_angular_damping(5.0)  # 较大角阻尼

            # ── 恢复漏斗碰撞 (确保沉降时漏斗生效) ──
            self._enable_funnel_collision(True)

            # ── 物理沉降: 空跑多步让棒子受重力自然堆叠 ──
            for _ in range(SETTLE_STEPS):
                self.scene.step()

            # ── 沉降完成后移除漏斗碰撞 (不阻碍后续抓取) ──
            self._enable_funnel_collision(False)

    def _enable_funnel_collision(self, enable: bool):
        """
        启用/禁用漏斗碰撞体。

        对于静态 Actor，通过将其移到极远处来 "禁用" 碰撞，
        或移回原位来 "启用" 碰撞。
        """
        if not hasattr(self, '_funnel_original_poses'):
            # 首次调用时保存原始位姿
            self._funnel_original_poses = []
            for wall in self.funnel_walls:
                self._funnel_original_poses.append(
                    wall.pose.raw_pose.clone()
                )

        if enable:
            # 恢复到原始位置
            for wall, orig_pose in zip(self.funnel_walls, self._funnel_original_poses):
                wall.set_pose(Pose.create(orig_pose))
        else:
            # 移到极远处 (禁用碰撞)
            far_away = torch.tensor([[999.0, 999.0, -999.0]], device=self.device)
            for wall in self.funnel_walls:
                wall.set_pose(Pose.create_from_pq(far_away))

    @staticmethod
    def _euler_to_quat_batch(
        roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
    ) -> torch.Tensor:
        """
        ZYX 欧拉角 → 四元数 (wxyz), batch 版本。

        Args:
            roll, pitch, yaw: (B,) tensors

        Returns:
            q: (B, 4) wxyz 四元数
        """
        cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
        cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    # ─────────────────────────────────────────────────────
    #  step: 离散宏动作
    # ─────────────────────────────────────────────────────
    def step(self, action):
        """
        覆盖 BaseEnv.step() 以支持离散宏动作。

        Args:
            action: int 或 Tensor — 要挑起的棒子编号 [0, N)

        Returns:
            obs, reward, terminated, truncated, info (标准 Gym 接口)
        """
        # 将 action 转为 tensor
        if isinstance(action, (int, np.integer)):
            action = torch.tensor([action], device=self.device, dtype=torch.long)
        elif isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device=self.device, dtype=torch.long)
        elif isinstance(action, torch.Tensor):
            action = action.to(device=self.device, dtype=torch.long)

        if action.dim() == 0:
            action = action.unsqueeze(0)

        # 对每个并行环境执行宏动作
        grasp_success = self._execute_heuristic_grasp(action)

        # 更新移除状态
        for env_i in range(len(action)):
            stick_id = action[env_i].item()
            if grasp_success[env_i] and not self.removed_mask[env_i, stick_id]:
                self.removed_mask[env_i, stick_id] = True
                self.num_removed[env_i] += 1

        # 步数递增
        self._elapsed_steps += 1

        # 获取 info 和 obs
        info = self.get_info()
        obs = self.get_obs(info, unflattened=True)
        reward = self.get_reward(obs=obs, action=action, info=info)
        obs = self._flatten_raw_obs(obs)

        # 终止条件: 所有棒子都被移除
        if "success" in info:
            terminated = info["success"].clone()
        else:
            terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        truncated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────
    #  启发式抓取 (宏动作占位)
    # ─────────────────────────────────────────────────────
    def _execute_heuristic_grasp(self, stick_ids: torch.Tensor) -> torch.Tensor:
        """
        启发式宏动作: 尝试挑起指定编号的棒子。

        当前为占位实现 — 简化逻辑:
        1. 检查目标棒子是否已被移除 → 若已移除则失败
        2. 检查目标棒子上方是否有其他棒子压着 (通过高度判断) → 若被压则失败
        3. 若可挑取，将棒子移到远处 (模拟移除)，并空跑若干步让剩余棒子重新沉降

        TODO: 替换为真正的运动规划 + 力控抓取闭环:
            - IK 求解 TCP → 棒子抓取点
            - 闭合夹爪 + 力反馈判断是否夹稳
            - 提升 + 放置到收集区
            - 多步底层 scene.step() 完成整个抓取流程

        Args:
            stick_ids: (B,) 每个并行环境要挑起的棒子编号

        Returns:
            success: (B,) bool tensor — 每个环境是否成功挑起
        """
        b = len(stick_ids)
        success = torch.zeros(b, dtype=torch.bool, device=self.device)

        for env_i in range(b):
            sid = stick_ids[env_i].item()

            # 检查: 棒子是否已被移除
            if self.removed_mask[env_i, sid]:
                continue

            # 检查: 棒子是否是 "自由" 的 (上方没有其他棒子压着)
            # 简化判断: 比较目标棒子高度与其他未移除棒子的高度
            target_z = self.sticks[sid].pose.p[env_i, 2].item()
            is_free = True

            for j in range(self.num_sticks):
                if j == sid or self.removed_mask[env_i, j]:
                    continue
                other_z = self.sticks[j].pose.p[env_i, 2].item()
                # 如果有其他棒子在目标棒子正上方且距离很近
                other_xy = self.sticks[j].pose.p[env_i, :2]
                target_xy = self.sticks[sid].pose.p[env_i, :2]
                xy_dist = torch.norm(other_xy - target_xy).item()

                if other_z > target_z + STICK_RADIUS and xy_dist < STICK_HALF_LENGTH:
                    # 有棒子在上方且水平距离较近 → 可能被压着
                    # 这里用简化的启发式，实际应使用接触力判断
                    is_free = False
                    break

            if not is_free:
                continue

            # 挑取成功: 将棒子移到远处
            far_pos = torch.tensor(
                [[999.0, 999.0, 999.0]], device=self.device
            )
            self.sticks[sid].set_pose(Pose.create_from_pq(far_pos))
            self.sticks[sid].set_linear_velocity(torch.zeros(1, 3, device=self.device))
            self.sticks[sid].set_angular_velocity(torch.zeros(1, 3, device=self.device))

            success[env_i] = True

        # 移除后让剩余棒子重新沉降 (短暂物理步进)
        if success.any():
            for _ in range(50):
                self.scene.step()

        return success

    # ─────────────────────────────────────────────────────
    #  评估: 成功条件
    # ─────────────────────────────────────────────────────
    def evaluate(self):
        # 所有棒子都被移除 → 成功
        all_removed = self.removed_mask.all(dim=1)
        return {
            "success": all_removed,
            "num_removed": self.num_removed,
            "removed_mask": self.removed_mask,
        }

    # ─────────────────────────────────────────────────────
    #  观测: Privileged State
    # ─────────────────────────────────────────────────────
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            removed_mask=self.removed_mask.float(),
            num_removed=self.num_removed.float().unsqueeze(-1),
        )
        if "state" in self.obs_mode:
            # Privileged state: 所有棒子的 6DoF 姿态 (7D raw_pose × N)
            stick_poses = torch.stack(
                [s.pose.raw_pose for s in self.sticks], dim=1
            )  # (B, N, 7)
            obs["stick_poses"] = stick_poses.flatten(start_dim=1)  # (B, N*7)
        return obs

    # ─────────────────────────────────────────────────────
    #  奖励: 模板 (Dense / Normalized)
    # ─────────────────────────────────────────────────────
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        模板奖励函数。

        当前设计:
        - 每成功挑起一根棒子: +1.0
        - 挑取失败 (被压住或已移除): -0.1
        - 全部挑完 bonus: +5.0
        """
        reward = torch.zeros(self.num_envs, device=self.device)

        # 基于 num_removed 的增量奖励
        # (这里简化为: 当前 num_removed / total 作为进度奖励)
        progress = self.num_removed.float() / self.num_sticks
        reward += progress

        # 成功 bonus
        if "success" in info:
            reward[info["success"]] = 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0


# ═══════════════════════════════════════════════════════════
#  独立运行: 可视化验证 + Dependency Graph 提取
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    env = gym.make(
        "PickUpSticks-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend="cpu",
    )

    obs, _ = env.reset(seed=42)
    uw = env.unwrapped
    print(f"观测维度: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"棒子数量: {NUM_STICKS}")

    # 预热: 额外物理步进, 让接触力稳定 (不执行宏动作)
    for _ in range(50):
        uw.scene.step()

    # ─── 提取 Dependency Graph ───
    graph = get_dependency_graph(uw.scene, uw.sticks)

    print(f"\n{'='*50}")
    print(f"  Dependency Graph  (threshold = 5% stick weight)")
    print(f"{'='*50}")
    print(f"  高度范围: [{min(graph['heights']):.4f}, {max(graph['heights']):.4f}] m")

    print(f"\n  依赖边 (i → j 表示 stick_i 支撑 stick_j):")
    edge_count = 0
    for i, row in enumerate(graph["dependency_matrix"]):
        for j, v in enumerate(row):
            if v:
                print(f"    stick_{i} → stick_{j}")
                edge_count += 1
    print(f"  共 {edge_count} 条依赖边\n")

    # ─── 计算移除难度 ───
    difficulty = get_removal_difficulty(
        graph["dependency_matrix"], graph["heights"]
    )
    print(f"  移除难度分数:")
    for i in range(NUM_STICKS):
        print(f"    stick_{i}: {difficulty[i]:.4f}")
    print()

    # ─── 交互式可视化 ───
    viewer = env.render()
    if hasattr(viewer, "paused"):
        viewer.paused = True

    while True:
        # 随机选择一根棒子尝试挑起
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: stick_{action}, Reward: {reward.item():.3f}, "
              f"Removed: {uw.num_removed.item()}/{NUM_STICKS}")
        try:
            env.render()
        except (TypeError, AttributeError):
            break
        if (terminated | truncated).any():
            print("Episode finished! Resetting...")
            obs, _ = env.reset()

    env.close()
