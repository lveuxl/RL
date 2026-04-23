"""
PickUpSticks-v1: 挑棒子 (Mikado / Pick-up Sticks) 仿真环境 (ManiSkill BaseEnv)

设计要点:
    1. 宏动作 (Macro-action) 接口: action 是离散的 stick_id, RL 智能体只学习"挑哪根"的顺序。
       底层抓取轨迹由 _execute_heuristic_grasp(stick_id) 占位实现。
    2. 防爆物理调优: 高 sim_freq + 高 solver iterations + 零恢复系数 + 高阻尼。
    3. 不可见漏斗 (Invisible Funnel): 4 块倾斜静态 Box 形成倒锥, 让棒子掉落后聚拢于桌面中心。
    4. 物理沉降阶段: 空跑数百步令棒子静止, 然后**移除漏斗碰撞体**以不阻碍机械臂。

Usage:
    conda activate /opt/anaconda3/envs/skill
    python pick_up_sticks.py
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import sapien.render
import torch
import gymnasium as gym
from gymnasium import spaces

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig


# ─── 棒子几何参数 (m) ───────────────────────────────
STICK_LENGTH = 0.18      # 棒子长度 18 cm (类似真实 Mikado 牙签放大版)
STICK_RADIUS = 0.003     # 棒子半径 3 mm (用细长 Box 近似, half_size = [L/2, R, R])
STICK_HALF = [STICK_LENGTH / 2, STICK_RADIUS, STICK_RADIUS]
NUM_STICKS = 15          # 棒子数量 N (= 离散动作空间维度)
STICK_DENSITY = 500      # kg/m³ (木质轻棒)

# 颜色配置: 每根棒子一种醒目颜色, 便于视觉辨识
STICK_COLORS = [
    [0.90, 0.15, 0.15, 1.0],  # 红
    [0.20, 0.70, 0.25, 1.0],  # 绿
    [0.15, 0.35, 0.85, 1.0],  # 蓝
    [0.95, 0.80, 0.15, 1.0],  # 黄
    [0.85, 0.40, 0.85, 1.0],  # 品红
    [0.20, 0.80, 0.85, 1.0],  # 青
    [0.95, 0.55, 0.15, 1.0],  # 橙
    [0.55, 0.30, 0.75, 1.0],  # 紫
    [0.15, 0.15, 0.15, 1.0],  # 黑
    [0.95, 0.95, 0.95, 1.0],  # 白
    [0.60, 0.40, 0.20, 1.0],  # 棕
    [0.85, 0.60, 0.70, 1.0],  # 粉
    [0.40, 0.60, 0.30, 1.0],  # 军绿
    [0.30, 0.50, 0.80, 1.0],  # 淡蓝
    [0.80, 0.20, 0.50, 1.0],  # 玫红
]

# ─── 漏斗几何参数 ───────────────────────────────
FUNNEL_CENTER_XY = (0.0, 0.0)    # 漏斗水平中心 (桌面坐标系)
FUNNEL_TOP_Z = 0.35              # 漏斗上沿高度
FUNNEL_BOTTOM_Z = 0.05           # 漏斗下沿高度 (紧贴桌面)
FUNNEL_TOP_HALF = 0.15           # 上沿半宽
FUNNEL_BOTTOM_HALF = 0.06        # 下沿半宽 (引导棒子聚拢至此半径内)
SETTLING_STEPS = 400             # 物理沉降步数

# ─── 抓取成功判定阈值 ───────────────────────────────
PERTURB_THRESHOLD = 0.003        # 其他棒子位移 < 3mm 视为"没扰动"


@register_env("PickUpSticks-v1", max_episode_steps=NUM_STICKS)
class PickUpSticksEnv(BaseEnv):
    """
    **Task Description:**
    挑棒子 (Pick-up Sticks / Mikado) 任务.
    桌面上堆叠 N 根彩色细棒, 互相交叠压迫. RL 智能体需要学习**最佳抽取顺序**,
    每次选择一根棒子尝试挑起, 目标是: 不扰动其他棒子的前提下, 尽可能多地移除棒子.

    **Action Space (Macro):**
    Discrete(N) — 离散动作, 选择要挑起的棒子索引 ∈ [0, N).
    底层轨迹由 _execute_heuristic_grasp(stick_id) 启发式实现 (占位).

    **Observation (Privileged State):**
    每根棒子的 6DoF 姿态 (position + quaternion) + 是否已被移除的标志.

    **Success Conditions:**
    全部 N 根棒子被成功依次取走 (不扰动剩余棒子).
    """

    SUPPORTED_ROBOTS = ["panda", "panda_stick"]
    agent: Panda
    num_sticks = NUM_STICKS

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        num_sticks: int = NUM_STICKS,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_sticks = num_sticks
        # 用于在 step 中追踪哪些棒子已被移除
        self._removed_mask: Optional[torch.Tensor] = None
        # 保存漏斗 actor 引用, 方便沉降后禁用
        self._funnel_walls: List[Any] = []
        self._funnel_disabled: bool = False
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # =====================================================================
    # 1. Sim Config — 防爆调优
    # =====================================================================
    @property
    def _default_sim_config(self):
        """
        高 sim_freq + 高 solver 迭代 = 碰撞稳定性好 (防止穿模爆炸).
        bounce_threshold 拉高 → 小速度碰撞不触发反弹.
        enable_enhanced_determinism → 复现实验.
        """
        return SimConfig(
            sim_freq=1000,             # 1 kHz 物理步进 (比默认 500Hz 更稳)
            control_freq=20,
            scene_config=SceneConfig(
                solver_position_iterations=150,
                solver_velocity_iterations=150,
                bounce_threshold=4.0,  # 提高阈值, 减少抖动/反弹
                enable_enhanced_determinism=True,
            ),
        )

    # =====================================================================
    # 2. 相机配置 (基础观测 + 俯视)
    # =====================================================================
    @property
    def _default_sensor_configs(self):
        # 俯视相机 (桌面正上方, 看堆)
        top_pose = sapien_utils.look_at(eye=[0.0, 0.0, 0.55], target=[0.0, 0.0, 0.05])
        # 斜视相机 (用于 RL obs)
        side_pose = sapien_utils.look_at(eye=[0.35, 0.0, 0.45], target=[0.0, 0.0, 0.08])
        return [
            CameraConfig("base_camera", side_pose, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("top_camera", top_pose, 128, 128, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.08])
        return CameraConfig("render_camera", pose, 1024, 1024, 1.0, 0.01, 100)

    # =====================================================================
    # 3. 机械臂加载
    # =====================================================================
    def _load_agent(self, options: dict):
        # 机械臂向 -x 方向退一步, 给桌面中心区域留出操作空间
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # =====================================================================
    # 4. 场景构建 — 桌子 + 漏斗 + 棒子
    # =====================================================================
    def _load_scene(self, options: dict):
        # ── 4.1 桌面场景 ──
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # ── 4.2 构建不可见漏斗 (Invisible Funnel) ──
        # 由 4 块倾斜静态 Box 围成倒锥, 引导棒子聚拢到桌面中心.
        # 这些 Box 只有碰撞体, 无可见材质 (alpha=0), 沉降后会被禁用碰撞.
        self._build_invisible_funnel()

        # ── 4.3 棒子物理材质: 零恢复系数 + 高摩擦 ──
        stick_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,   # 关键: 零反弹, 抑制爆炸
        )

        # ── 4.4 逐根构建棒子 ──
        self.sticks = []
        rng = np.random.default_rng(42)

        for i in range(self.num_sticks):
            builder = self.scene.create_actor_builder()

            # 碰撞体: 细长 Box 近似圆柱 (Box 接触更稳定, capsule 在密堆下易穿模)
            builder.add_box_collision(
                half_size=STICK_HALF,
                material=stick_mat,
                density=STICK_DENSITY,
            )

            # 可视材质
            color = STICK_COLORS[i % len(STICK_COLORS)]
            render_mat = sapien.render.RenderMaterial(
                base_color=color, roughness=0.7, specular=0.1, metallic=0.0
            )
            builder.add_box_visual(half_size=STICK_HALF, material=render_mat)

            # 初始错层位姿: 在漏斗上方螺旋分布, 避免初始穿模
            # (实际沉降位姿由 _initialize_episode 设置)
            spawn_z = FUNNEL_TOP_Z + 0.15 + i * (STICK_RADIUS * 2.5)
            angle = i * (2 * np.pi / self.num_sticks)
            spawn_xy = [
                0.04 * np.cos(angle) + FUNNEL_CENTER_XY[0],
                0.04 * np.sin(angle) + FUNNEL_CENTER_XY[1],
            ]
            builder.initial_pose = sapien.Pose(p=[spawn_xy[0], spawn_xy[1], spawn_z])

            stick = builder.build(name=f"stick_{i}")

            # 通过 set_damping 增加阻尼, 使掉落后能迅速静止 (防爆关键)
            # ManiSkill / SAPIEN: 可通过 physx body 的 linear/angular damping 设置
            try:
                for body in stick._bodies:
                    body.set_linear_damping(0.5)
                    body.set_angular_damping(0.5)
            except Exception:
                # 某些后端接口不支持, 忽略即可; 恢复系数=0 已足够防爆
                pass

            self.sticks.append(stick)

    def _build_invisible_funnel(self):
        """
        构建 4 块倾斜静态 Box 形成的倒锥形漏斗.
        每块 Box 的姿态: 绕 X 或 Y 轴倾斜, 使内表面形成 45° 斜面.
        """
        # 漏斗斜面: 从 (±TOP_HALF, TOP_Z) 倾斜到 (±BOTTOM_HALF, BOTTOM_Z)
        wall_length = 0.4            # 每块壁的长度
        wall_thickness = 0.01        # 壁厚
        # 斜面高度 & 水平跨度
        dz = FUNNEL_TOP_Z - FUNNEL_BOTTOM_Z
        dx = FUNNEL_TOP_HALF - FUNNEL_BOTTOM_HALF
        slant_length = float(np.sqrt(dz * dz + dx * dx))
        tilt_angle = float(np.arctan2(dx, dz))   # 与 Z 轴夹角

        # 4 个方向: +x, -x, +y, -y
        # 每块墙的朝向 = 绕 Y 轴 (+/- tilt_angle) 或绕 X 轴 (+/- tilt_angle)
        walls_cfg = [
            # 方向法向量 (指向漏斗内部), 倾斜轴, 倾斜角符号
            ("+x", "y",  -1),
            ("-x", "y",  +1),
            ("+y", "x",  +1),
            ("-y", "x",  -1),
        ]

        mid_x = (FUNNEL_TOP_HALF + FUNNEL_BOTTOM_HALF) / 2
        mid_z = (FUNNEL_TOP_Z + FUNNEL_BOTTOM_Z) / 2

        for direction, axis, sign in walls_cfg:
            builder = self.scene.create_actor_builder()

            # 半尺寸: [薄厚, 壁长/2, 斜面长度/2]
            half = [wall_thickness / 2, wall_length / 2, slant_length / 2]
            # 静态 (kinematic) 以避免被撞飞
            funnel_mat = sapien.pysapien.physx.PhysxMaterial(
                static_friction=0.3, dynamic_friction=0.2, restitution=0.0
            )
            builder.add_box_collision(half_size=half, material=funnel_mat)

            # 位置 & 姿态
            if direction == "+x":
                pos = [FUNNEL_CENTER_XY[0] + mid_x, FUNNEL_CENTER_XY[1], mid_z]
                # 绕 Y 轴旋转 (sign * tilt_angle), 使斜面朝向中心
                half_a = sign * tilt_angle / 2
                q = [np.cos(half_a), 0.0, np.sin(half_a), 0.0]
            elif direction == "-x":
                pos = [FUNNEL_CENTER_XY[0] - mid_x, FUNNEL_CENTER_XY[1], mid_z]
                half_a = sign * tilt_angle / 2
                q = [np.cos(half_a), 0.0, np.sin(half_a), 0.0]
            elif direction == "+y":
                pos = [FUNNEL_CENTER_XY[0], FUNNEL_CENTER_XY[1] + mid_x, mid_z]
                half_a = sign * tilt_angle / 2
                q = [np.cos(half_a), np.sin(half_a), 0.0, 0.0]
            else:  # "-y"
                pos = [FUNNEL_CENTER_XY[0], FUNNEL_CENTER_XY[1] - mid_x, mid_z]
                half_a = sign * tilt_angle / 2
                q = [np.cos(half_a), np.sin(half_a), 0.0, 0.0]

            builder.initial_pose = sapien.Pose(p=pos, q=q)
            wall = builder.build_kinematic(name=f"funnel_{direction}")
            self._funnel_walls.append(wall)

    def _disable_funnel(self):
        """
        沉降完成后, 禁用漏斗碰撞体 / 移出场景, 避免阻碍机械臂.
        策略: 把漏斗 kinematic actor 瞬移到远处 (因为 ManiSkill 中动态删除 actor 较复杂).
        """
        if self._funnel_disabled:
            return
        far_pose = sapien.Pose(p=[0.0, 0.0, -10.0])  # 藏到地面下方
        for wall in self._funnel_walls:
            try:
                wall.set_pose(far_pose)
            except Exception:
                pass
        self._funnel_disabled = True

    # =====================================================================
    # 5. Episode 初始化: 棒子撒落 + 沉降 + 移除漏斗
    # =====================================================================
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 漏斗复位 (Episode 之间重置)
            self._funnel_disabled = False
            self._funnel_reset()

            # 重置"已移除"掩码
            self._removed_mask = torch.zeros(
                (b, self.num_sticks), dtype=torch.bool, device=self.device
            )

            # ── 5.1 在漏斗上方错层随机撒落棒子 ──
            # 错层高度 + 随机 yaw + 随机 xy 微扰, 避免生成时穿模
            rng = np.random.default_rng(options.get("seed", 0) if options else 0)

            for i in range(self.num_sticks):
                # 高度错层, 避免同一高度生成两根棒子
                layer = i
                z = FUNNEL_TOP_Z + 0.08 + layer * (STICK_RADIUS * 2.5 + 0.01)

                # XY 在漏斗上沿范围内随机
                r = float(rng.uniform(0.0, FUNNEL_TOP_HALF * 0.6))
                phi = float(rng.uniform(0.0, 2 * np.pi))
                x = FUNNEL_CENTER_XY[0] + r * np.cos(phi)
                y = FUNNEL_CENTER_XY[1] + r * np.sin(phi)

                # 随机水平朝向 yaw
                yaw = float(rng.uniform(0.0, 2 * np.pi))
                half_y = yaw / 2
                # 加一点微小的 pitch/roll, 使棒子掉落时旋转
                pitch = float(rng.uniform(-0.15, 0.15))
                roll = float(rng.uniform(-0.15, 0.15))

                # 欧拉角 → 四元数 (z-y-x 顺序)
                cy, sy = np.cos(half_y), np.sin(half_y)
                cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
                cr, sr = np.cos(roll / 2), np.sin(roll / 2)
                qw = cr * cp * cy + sr * sp * sy
                qx = sr * cp * cy - cr * sp * sy
                qy = cr * sp * cy + sr * cp * sy
                qz = cr * cp * sy - sr * sp * cy
                q = torch.tensor([[qw, qx, qy, qz]], device=self.device).expand(b, -1)

                xyz = torch.tensor([[x, y, z]], device=self.device).expand(b, -1)
                self.sticks[i].set_pose(Pose.create_from_pq(xyz, q))
                # 重置速度 (防止残留动量)
                zero3 = torch.zeros(b, 3, device=self.device)
                self.sticks[i].set_linear_velocity(zero3)
                self.sticks[i].set_angular_velocity(zero3)

            # ── 5.2 物理沉降 ──
            # 空跑 SETTLING_STEPS 步, 让棒子在漏斗引导下落到桌面并静止
            for _ in range(SETTLING_STEPS):
                self.scene.step()

            # ── 5.3 沉降完成, 移除漏斗 (关键步骤!) ──
            self._disable_funnel()

    def _funnel_reset(self):
        """Episode 重置时把漏斗移回原位 (若上一个 episode 已被禁用)."""
        if not hasattr(self, "_funnel_init_poses"):
            # 第一次调用, 记录初始位姿
            self._funnel_init_poses = [w.pose for w in self._funnel_walls]
            return
        for wall, init_pose in zip(self._funnel_walls, self._funnel_init_poses):
            try:
                wall.set_pose(init_pose)
            except Exception:
                pass

    # =====================================================================
    # 6. 动作空间: 离散宏动作 (选哪根棒子)
    # =====================================================================
    @property
    def single_action_space(self):
        """Discrete(N): 选择要挑起的棒子索引."""
        return spaces.Discrete(self.num_sticks)

    @property
    def action_space(self):
        if self.num_envs == 1:
            return self.single_action_space
        # 多环境: Vector
        return spaces.MultiDiscrete([self.num_sticks] * self.num_envs)

    # =====================================================================
    # 7. 启发式抓取 (占位): 接收 stick_id, 执行底层轨迹
    # =====================================================================
    def _execute_heuristic_grasp(self, stick_id: int) -> Dict[str, Any]:
        """
        启发式抓取占位方法 — 由环境内部调用底层仿真步进完成抓取闭环.

        TODO: 完整实现需要
            1. 读取目标棒子的当前 pose (position + yaw)
            2. 规划抓取姿态: gripper 朝向棒子中点, 垂直于棒子长轴
            3. 分阶段运动: approach → descend → close gripper → lift → retreat
            4. 每阶段调用 N 步底层仿真 (self.scene.step() + self.agent.controller.set_action(...))
            5. 检查: (a) 目标棒子是否被举起 (z > threshold)
                     (b) 其他棒子是否被扰动 (位移 > PERTURB_THRESHOLD)

        当前占位: 简单瞬移 (teleport) 目标棒子到"回收区" + 记录成功.
        """
        # ── 记录干预前所有棒子位置 (用于判定扰动) ──
        positions_before = torch.stack([s.pose.p.clone() for s in self.sticks], dim=0)

        # ── 占位: 瞬移目标棒子到远处 (模拟"抓走") ──
        far_pose = torch.tensor([[1.0, 1.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        zero3 = torch.zeros(self.num_envs, 3, device=self.device)
        self.sticks[stick_id].set_pose(Pose.create_from_pq(far_pose))
        self.sticks[stick_id].set_linear_velocity(zero3)
        self.sticks[stick_id].set_angular_velocity(zero3)

        # 空跑几步让其他棒子响应重心变化 (可能塌陷)
        for _ in range(30):
            self.scene.step()

        # ── 判定其他棒子是否被扰动 ──
        positions_after = torch.stack([s.pose.p.clone() for s in self.sticks], dim=0)
        displacements = torch.norm(positions_after - positions_before, dim=-1)
        # 排除被抓起的那根自己
        mask = torch.ones(self.num_sticks, dtype=torch.bool, device=self.device)
        mask[stick_id] = False
        # 同时排除已经移除的棒子 (它们在远处, 不计入扰动)
        if self._removed_mask is not None:
            # removed_mask 形状 (B, N); 简单起见这里取 env 0
            mask = mask & (~self._removed_mask[0])

        max_disp = displacements[mask].max().item() if mask.any() else 0.0
        others_disturbed = max_disp > PERTURB_THRESHOLD

        # ── 标记移除 ──
        if self._removed_mask is not None:
            self._removed_mask[:, stick_id] = True

        return {
            "grasped_stick_id": stick_id,
            "others_max_displacement": max_disp,
            "others_disturbed": others_disturbed,
            "success": not others_disturbed,
        }

    # =====================================================================
    # 8. step 重写: 接收离散动作, 调用启发式抓取
    # =====================================================================
    def step(self, action):
        """
        宏动作 step:
            action: int / np.ndarray / torch.Tensor — stick_id ∈ [0, N)
        """
        # 规范化 action
        if isinstance(action, torch.Tensor):
            stick_id = int(action.flatten()[0].item())
        elif isinstance(action, np.ndarray):
            stick_id = int(action.flatten()[0])
        else:
            stick_id = int(action)

        # ── 非法动作检查 ──
        if self._removed_mask is not None and bool(self._removed_mask[0, stick_id]):
            # 选了一根已经被移除的棒子 → 惩罚 + 不前进
            obs = self.get_obs()
            info = {
                "illegal_action": True,
                "grasped_stick_id": stick_id,
                "others_disturbed": False,
                "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
            reward = torch.full((self.num_envs,), -1.0, device=self.device)
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return obs, reward, terminated, truncated, info

        # ── 执行启发式抓取 ──
        grasp_info = self._execute_heuristic_grasp(stick_id)

        # ── 获取观测 & 计算奖励 ──
        obs = self.get_obs()
        info = self.evaluate()
        info.update(grasp_info)

        reward = self.compute_dense_reward(obs=obs, action=action, info=info)

        # ── 终止条件 ──
        # (a) 全部取走 → success
        # (b) 任意一步扰动其他棒子 → terminate (失败)
        all_removed = self._removed_mask.all(dim=-1) if self._removed_mask is not None else \
                      torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        disturbed = torch.tensor(
            [grasp_info["others_disturbed"]] * self.num_envs,
            dtype=torch.bool, device=self.device,
        )
        terminated = all_removed | disturbed
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        info["success"] = all_removed & (~disturbed)
        return obs, reward, terminated, truncated, info

    # =====================================================================
    # 9. 观测: Privileged State (所有棒子的 6DoF)
    # =====================================================================
    def _get_obs_extra(self, info: Dict):
        """
        特权观测: 每根棒子的位姿 (7 维: pos + quat) + removed 标志.
        共计 N * (7 + 1) = N * 8 维.
        """
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            stick_poses = torch.stack(
                [s.pose.raw_pose for s in self.sticks], dim=1
            )  # (B, N, 7)
            removed = (
                self._removed_mask.float()
                if self._removed_mask is not None
                else torch.zeros(self.num_envs, self.num_sticks, device=self.device)
            )
            obs["stick_poses"] = stick_poses.flatten(start_dim=1)       # (B, N*7)
            obs["stick_removed"] = removed                              # (B, N)
        return obs

    # =====================================================================
    # 10. Dependency Graph (辅助函数框架 — 哪根棒子压在哪根上面)
    # =====================================================================
    def compute_dependency_graph(self, force_threshold_ratio: float = 0.1) -> np.ndarray:
        """
        通过 SAPIEN 接触力求解 "i 压在 j 上面" 的有向图 (adjacency matrix).
        参考 jenga_tower.py::get_support_graph 的设计.

        Returns:
            dep: np.ndarray (N, N), dep[i, j] = 1 表示 stick_i 压在 stick_j 上 (即 j 支撑 i).
        """
        n = self.num_sticks
        dt = self.scene.timestep

        # entity → index 映射
        entity_to_idx = {s._bodies[0].entity: i for i, s in enumerate(self.sticks)}

        # 单根棒子重力阈值
        vol = STICK_LENGTH * (2 * STICK_RADIUS) ** 2
        threshold = force_threshold_ratio * vol * STICK_DENSITY * 9.81

        dep = np.zeros((n, n), dtype=int)
        for contact in self.scene.get_contacts():
            e0, e1 = contact.bodies[0].entity, contact.bodies[1].entity
            if e0 not in entity_to_idx or e1 not in entity_to_idx:
                continue
            idx0, idx1 = entity_to_idx[e0], entity_to_idx[e1]
            fz = sum(pt.impulse[2] for pt in contact.points) / dt

            # fz > 0: body1 → body0 的冲量 +Z, 即 body1 对 body0 向上推力
            # ∴ body0 压在 body1 上 (body1 在下支撑 body0)
            if fz > threshold:
                dep[idx0, idx1] = 1
            if -fz > threshold:
                dep[idx1, idx0] = 1

        return dep

    # =====================================================================
    # 11. 评估 & 奖励
    # =====================================================================
    def evaluate(self) -> Dict[str, torch.Tensor]:
        if self._removed_mask is None:
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            n_removed = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            success = self._removed_mask.all(dim=-1)
            n_removed = self._removed_mask.sum(dim=-1).to(torch.int32)
        return {
            "success": success,
            "num_removed": n_removed,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        奖励设计 (参考 Jenga Tower 的密集奖励结构):
            + 基础奖励: 每成功挑起一根棒子 +1.0
            - 扰动惩罚: 若其他棒子位移超过阈值 -2.0
            + 完成奖励: 全部取走 +5.0
        """
        reward = torch.zeros(self.num_envs, device=self.device)

        # 成功抓起 (未扰动) → +1
        if info.get("others_disturbed", False) is False and not info.get("illegal_action", False):
            reward += 1.0

        # 扰动惩罚
        if info.get("others_disturbed", False):
            reward -= 2.0

        # 全部完成 → +5 (one-shot)
        if "success" in info:
            success = info["success"]
            if isinstance(success, torch.Tensor):
                reward = torch.where(success, reward + 5.0, reward)
            elif success:
                reward += 5.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # 理论最大单步奖励 = 1 (抓起) + 5 (完成) = 6
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0


# ═══════════════════════════════════════════════════════════════════════════
#  独立运行入口: 可视化验证
# ═══════════════════════════════════════════════════════════════════════════
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

    print(f"\n{'='*60}")
    print(f"  PickUpSticks-v1 环境初始化完成")
    print(f"{'='*60}")
    print(f"  棒子数量 N = {uw.num_sticks}")
    print(f"  动作空间  = {env.action_space}")
    print(f"  观测空间  = {env.observation_space}")
    print(f"  沉降步数  = {SETTLING_STEPS}")
    print(f"{'='*60}\n")

    # ── 提取 Dependency Graph ──
    dep = uw.compute_dependency_graph()
    edges = int(dep.sum())
    print(f"  Dependency Graph: {edges} 条 '压在上面' 的边")
    for i in range(uw.num_sticks):
        for j in range(uw.num_sticks):
            if dep[i, j]:
                print(f"    stick_{i:2d}  压在  stick_{j:2d}  上面")
    print()

    # ── 交互可视化 ──
    viewer = env.render()
    if hasattr(viewer, "paused"):
        viewer.paused = True

    # ── 随机顺序挑棒子 (演示宏动作接口) ──
    remaining = list(range(uw.num_sticks))
    rng = np.random.default_rng(0)
    rng.shuffle(remaining)

    try:
        for sid in remaining:
            print(f"  尝试挑起 stick_{sid} ...")
            obs, reward, terminated, truncated, info = env.step(sid)
            env.render()
            print(f"    reward = {float(reward):.2f}  "
                  f"扰动 max_disp = {info.get('others_max_displacement', 0):.4f}m  "
                  f"disturbed = {info.get('others_disturbed', False)}")
            if bool(terminated.any() if hasattr(terminated, 'any') else terminated):
                print(f"\n  ⚠ Episode 终止. 成功移除: {int(info.get('num_removed', 0))}/{uw.num_sticks}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
