"""
JengaTower-v1: Jenga 塔仿真环境 (ManiSkill BaseEnv)
Usage:
    conda activate /opt/anaconda3/envs/skill
    python jenga_tower.py
"""
from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

# ─── Jenga 标准木块尺寸*2 (m) ───
BLOCK_L = 0.15   # 长 7.5 cm
BLOCK_W = 0.05   # 宽 2.5 cm
BLOCK_H = 0.03   # 高 1.5 cm
NUM_LEVELS = 18
NUM_BLOCKS = NUM_LEVELS * 3

# 每层 3 块木块的颜色 (浅木 / 中木 / 深木)
BLOCK_COLORS = [
    [0.82, 0.68, 0.47, 1.0],
    [0.65, 0.50, 0.32, 1.0],
    [0.48, 0.35, 0.20, 1.0],
]

BLOCK_DENSITY = 600  # kg/m³


def get_support_graph(scene, actors, force_threshold_ratio=0.1):
    """
    从 SAPIEN 物理场景提取 Physical Graph (节点属性 + 支撑邻接矩阵)。

    Args:
        scene:  ManiSkillScene，提供 get_contacts() / timestep
        actors: list[Actor]，场景中所有木块
        force_threshold_ratio: 判定支撑的力阈值占单块重力的比例 (默认 10%)

    Returns:
        dict:
          "volumes":        list[float]      — 每个木块的包围盒体积 v_i (m³)
          "heights":        list[float]      — 每个木块重心的 Z 高度 h_i (m)
          "support_matrix": list[list[int]]  — A[i][j]=1 表示 block_i 支撑 block_j
    """
    n = len(actors)
    dt = scene.timestep

    vol = BLOCK_L * BLOCK_W * BLOCK_H
    volumes = [vol] * n
    heights = [float(b.pose.p[0, 2]) for b in actors]

    # entity 对象 → 木块索引
    entity_to_idx = {b._bodies[0].entity: i for i, b in enumerate(actors)}

    # 阈值 = 单块重力 × ratio
    threshold = force_threshold_ratio * vol * BLOCK_DENSITY * 9.81

    support_matrix = [[0] * n for _ in range(n)]

    for contact in scene.get_contacts():
        e0, e1 = contact.bodies[0].entity, contact.bodies[1].entity
        if e0 not in entity_to_idx or e1 not in entity_to_idx:
            continue
        idx0, idx1 = entity_to_idx[e0], entity_to_idx[e1]

        # PhysX 约定: point.impulse 施加在 bodies[0] 上
        # ∴ bodies[0] 对 bodies[1] 的力 = −Σimpulse / dt
        fz = sum(pt.impulse[2] for pt in contact.points) / dt

        if -fz > threshold:   # bodies[0] 向上推 bodies[1]
            support_matrix[idx0][idx1] = 1
        if fz > threshold:    # bodies[1] 向上推 bodies[0]
            support_matrix[idx1][idx0] = 1

    return {"volumes": volumes, "heights": heights, "support_matrix": support_matrix}


def get_gt_stability(support_matrix, volumes, poses, alpha=2.3, beta=30.0):
    """
    计算每个木块的 Ground Truth Stability 标签 (纯 NumPy/SciPy, 无引擎依赖)。

    Args:
        support_matrix: (N, N) array-like, A[i][j]=1 表示 block_i 支撑 block_j
        volumes:        (N,) 每个木块的体积 (m³)
        poses:          (N, 3) 每个木块重心的 [x, y, z] 坐标 (m)
        alpha:          载荷衰减系数
        beta:           平衡裕度 sigmoid 斜率

    Returns:
        s_load:    np.ndarray (N,) — 载荷因子 ∈(0,1], 值小=上方载荷重
        s_balance: np.ndarray (N,) — 平衡裕度 ∈(0,1), 值大=移除后重心仍在支撑凸包内
    """
    from collections import deque
    from scipy.spatial import ConvexHull

    n = len(volumes)
    A = np.asarray(support_matrix, dtype=int)
    vols = np.asarray(volumes, dtype=float)
    pos = np.asarray(poses, dtype=float)

    # ── BFS: 对每个块 i, 求其传递支撑闭包 S_i ──
    children = [np.where(A[i] > 0)[0] for i in range(n)]

    supported_sets = []
    for i in range(n):
        visited = set(children[i].tolist())
        queue = deque(visited)
        while queue:
            for c in children[queue.popleft()]:
                if c not in visited:
                    visited.add(c)
                    queue.append(c)
        supported_sets.append(visited)

    # ── s_load = exp(−α · |S_i|/N_total) ──
    s_load = np.array([
        np.exp(-alpha * len(S) / n) for S in supported_sets
    ])

    # ── s_balance ──
    level_of = np.arange(n) // 3

    s_balance = np.ones(n)
    for i in range(n):
        S_i = supported_sets[i]
        if not S_i:
            continue

        S_idx = np.array(list(S_i))

        # S_i 的体积加权 XY 联合重心
        w = vols[S_idx]
        com_xy = (pos[S_idx, :2] * w[:, None]).sum(0) / w.sum()

        # 同层 co-supporters: j ≠ i, same level, 且直接支撑 S_i 中至少一块
        layer_i = level_of[i]
        co_sup = [j for j in range(n)
                  if j != i and level_of[j] == layer_i and A[j, S_idx].any()]

        if not co_sup:
            # 无共同支撑者 → 移除 i 后上方全部失去支撑
            margin = -1.0
        else:
            # 用 co-supporters 的矩形足迹角点构建 2D 凸包
            pts = []
            for j in co_sup:
                cx, cy = pos[j, 0], pos[j, 1]
                hx = BLOCK_L / 2 if level_of[j] % 2 == 0 else BLOCK_W / 2
                hy = BLOCK_W / 2 if level_of[j] % 2 == 0 else BLOCK_L / 2
                pts.extend([(cx - hx, cy - hy), (cx + hx, cy - hy),
                            (cx + hx, cy + hy), (cx - hx, cy + hy)])
            try:
                hull = ConvexHull(np.array(pts))
                # equations[:,:-1] = 外向单位法线, equations[:,-1] = 偏移
                # 内部点满足 n·p + d ≤ 0, 因此 margin = −max(n·com+d)
                margin = -(hull.equations[:, :2] @ com_xy
                           + hull.equations[:, 2]).max()
            except Exception:
                margin = -1.0

        s_balance[i] = 1.0 / (1.0 + np.exp(-beta * margin))

    return s_load, s_balance


def get_gt_potentiality(scene, actors, threshold=0.03, sim_steps=100):
    """
    反事实干预: 逐个移除木块并物理前推, 评估塔的剩余稳定度。

    Args:
        scene:     ManiSkillScene
        actors:    list[Actor], 场景中所有木块
        threshold: 位移阈值 (m), 低于此视为未坍塌
        sim_steps: 每次干预后的物理推演步数

    Returns:
        potentiality: np.ndarray (N,) — 移除 block_i 后剩余稳定木块比例 ∈[0,1]
    """
    n = len(actors)
    device = actors[0].device

    initial_state = scene.get_sim_state()
    init_pos = torch.stack([a.pose.p[0] for a in actors])  # (N, 3)

    far = torch.tensor([[999.0, 999.0, 999.0]], device=device)
    zero3 = torch.zeros(1, 3, device=device)

    potentiality = np.zeros(n)
    for i in range(n):
        actors[i].set_pose(Pose.create_from_pq(far))
        actors[i].set_linear_velocity(zero3)
        actors[i].set_angular_velocity(zero3)

        for _ in range(sim_steps):
            scene.step()

        stable = 0
        for j in range(n):
            if j == i:
                continue
            disp = torch.norm(actors[j].pose.p[0] - init_pos[j]).item()
            if disp < threshold:
                stable += 1
        potentiality[i] = stable / (n - 1)

        scene.set_sim_state(initial_state)

    return potentiality


def calculate_topology_complexity(num_layers, support_matrix, h_max=18, w1=0.4, w2=0.6):
    """
    计算当前塔构型的连续拓扑复杂度分数。

    Args:
        num_layers:     当前塔的实际层数 H
        support_matrix: (N, N) array-like, A[i][j]=1 表示 i 支撑 j
        h_max:          最大层数 (用于归一化)
        w1, w2:         高度项与入度项的权重

    Returns:
        float: 复杂度分数 c ∈ (0, 1+)
    """
    A = np.asarray(support_matrix, dtype=int)
    in_degree = A.sum(axis=0)  # 列求和 = 每个节点的入度
    return w1 * (num_layers / h_max) + w2 * np.mean(np.exp(-in_degree))


def render_point_cloud(scene, cameras, actors):
    """
    多视角 RGB-D + Segmentation → 世界坐标系点云 → 按木块分割。

    Args:
        scene:   ManiSkillScene，提供 update_render()
        cameras: dict {uid: Camera}，要使用的相机传感器
        actors:  list[Actor]，需要分割的木块列表

    Returns:
        dict:
          "global_pcd":     np.ndarray (N, 6)   — [x, y, z, r, g, b] 世界坐标
          "global_seg_ids": np.ndarray (N,)     — per-point per_scene_id
          "per_block_pcd":  list[np.ndarray]    — 每个木块 (M_i, 6); 被完全遮挡则 (0, 6)
          "camera_rgbs":    dict {uid: np.ndarray (H,W,3)}  — 各相机 RGB 图
    """
    scene.update_render(update_sensors=True, update_human_render_cameras=False)
    for sensor in cameras.values():
        sensor.capture()

    all_xyz, all_rgb, all_seg = [], [], []
    camera_rgbs = {}

    for cam_uid, sensor in cameras.items():
        images = sensor.get_obs(rgb=True, position=True, segmentation=True)
        params = sensor.get_params()

        pos = images["position"].float() / 1000.0  # (B,H,W,3) OpenGL cam, mm→m
        seg = images["segmentation"]                # (B,H,W,1) per_scene_id
        rgb = images["rgb"]                         # (B,H,W,3)
        cam2world = params["cam2world_gl"]          # (B,4,4)

        B, H, W, _ = pos.shape

        # 保存 RGB 图 (第一个 batch)
        rgb_np = rgb[0].cpu().numpy()
        camera_rgbs[cam_uid] = (rgb_np / 255.0 if rgb_np.max() > 1.0 else rgb_np).astype(np.float32)

        # 齐次坐标 → 世界坐标
        ones = torch.ones(B, H, W, 1, device=pos.device, dtype=pos.dtype)
        pts_world = (
            torch.cat([pos, ones], dim=-1)
            .reshape(B, -1, 4)
            @ cam2world.transpose(1, 2)
        )  # (B, H*W, 4)

        seg_flat = seg.reshape(B, -1)     # (B, H*W)
        valid = seg_flat[0] != 0          # 去除背景

        all_xyz.append(pts_world[0, valid, :3].cpu().numpy())
        all_seg.append(seg_flat[0, valid].cpu().numpy().astype(int))

        rgb_v = rgb.float().reshape(B, -1, 3)[0, valid].cpu().numpy()
        all_rgb.append(rgb_v / 255.0 if rgb_v.max() > 1.0 else rgb_v)

    global_xyz = np.concatenate(all_xyz)
    global_rgb = np.concatenate(all_rgb)
    global_seg = np.concatenate(all_seg)
    global_pcd = np.concatenate([global_xyz, global_rgb], axis=1)

    # 按木块分割: per_scene_id → block index
    per_block_pcd = []
    for block in actors:
        sid = block._objs[0].per_scene_id
        mask = global_seg == sid
        per_block_pcd.append(global_pcd[mask] if mask.any() else np.zeros((0, 6)))

    return {
        "global_pcd": global_pcd,
        "global_seg_ids": global_seg,
        "per_block_pcd": per_block_pcd,
        "camera_rgbs": camera_rgbs,
    }


@register_env("JengaTower-v1", max_episode_steps=100)
class JengaTowerEnv(BaseEnv):
    """
    **Task Description:**
    Jenga 塔环境 — 18 层、每层 3 块、相邻层 90° 交替。
    塔放置在桌面上，Panda 机械臂可与之交互。

    **Success Conditions:**
    Placeholder — 后续定义抽取/堆叠任务。
    """

    SUPPORTED_ROBOTS = ["panda","panda_stick"]
    agent: Panda
    num_levels = NUM_LEVELS

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=500,
            control_freq=20,
            scene_config=SceneConfig(
                solver_position_iterations=100,
                solver_velocity_iterations=100,
                bounce_threshold=2.0,
                enable_enhanced_determinism=True,
            ),
        )

    @property
    def _default_sensor_configs(self):
        tower_h = self.num_levels * BLOCK_H
        pose = sapien_utils.look_at(
            eye=[0.3, 0, tower_h + 0.3], target=[0, 0, tower_h * 0.5]
        )
        configs = [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

        cx, cy, mid_z = 0.2, 0.0, tower_h * 0.5
        cam_radius = 0.35
        for i in range(4):
            angle = i * np.pi / 2
            eye = [cx + cam_radius * np.cos(angle), cy + cam_radius * np.sin(angle), mid_z]
            p = sapien_utils.look_at(eye, [cx, cy, mid_z])
            configs.append(CameraConfig(f"surround_{i}", p, 256, 256, np.pi / 3, 0.01, 10))

        return configs

    @property
    def _default_human_render_camera_configs(self):
        tower_h = self.num_levels * BLOCK_H
        pose = sapien_utils.look_at(
            eye=[0.45, 0.35, tower_h * 0.8], target=[0.0, 0.0, tower_h * 0.4]
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        wood_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        )

        #render_mats = [sapien.render.RenderMaterial(base_color=c) for c in BLOCK_COLORS]

        # ─── 核心修改: 加载多张不同的木纹贴图 ───
        # 定义你要使用的纹理文件名列表 (确保这些文件在脚本同目录下)
        try:
            tex_side = sapien.render.RenderTexture2D(filename="wood_texture4.jpg", srgb=True)
            tex_end = sapien.render.RenderTexture2D(filename="wood_texture2.jpg", srgb=True)
        except Exception as e:
            raise RuntimeError(f"⚠️ 贴图加载失败: {e}。请确保目录下有 wood_texture3.jpeg 和 wood_texture4.png")

        # 侧面材质 (面积最大的 4 个面)
        mat_side = sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0], roughness=0.85, specular=0.1, metallic=0.0
        )
        mat_side.base_color_texture = tex_side

        # 端面材质 (面积最小的 2 个截面)
        mat_end = sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0], roughness=0.9, specular=0.05, metallic=0.0
        )
        mat_end.base_color_texture = tex_end

        self.blocks = []
        center_x = 0.2  
        center_y = 0.0

        # 使用随机数生成器在后面的循环中随机选择材质
        rng = np.random.default_rng(42)
        # 定义间隙和端面厚度参数
        self.gap_w = 0.005  # 水平间隙 2cm
        self.gap_h = 0.0005 # 垂直间隙 0.5mm，让积木自然沉降避免穿模弹飞

        # for level in range(self.num_levels):
        #     z = level * BLOCK_H + BLOCK_H / 2
        #     for i in range(3):
        #         offset = (i - 1) * BLOCK_W

        #         if level % 2 == 0:
        #             half = [BLOCK_L / 2, BLOCK_W / 2, BLOCK_H / 2]
        #             # 加上中心点偏移
        #             pos = [center_x, center_y + offset, z] 
        #         else:
        #             half = [BLOCK_W / 2, BLOCK_L / 2, BLOCK_H / 2]
        #             # 加上中心点偏移
        #             pos = [center_x + offset, center_y, z]

        #         builder = self.scene.create_actor_builder()
        #         builder.add_box_collision(
        #             half_size=half, material=wood_mat, density=600
        #         )

        #         chosen_mat = render_mats[rng.integers(0, len(render_mats))]
        #         builder.add_box_visual(half_size=half, material=chosen_mat)

        #         builder.initial_pose = sapien.Pose(p=pos)
        #         block = builder.build(name=f"block_{level}_{i}")
        #         self.blocks.append(block)
        for level in range(self.num_levels):
            # 加入垂直空隙
            z = level * (BLOCK_H + self.gap_h) + BLOCK_H / 2
            for i in range(3):
                # 加入水平空隙
                offset = (i - 1) * (BLOCK_W + self.gap_w)

                builder = self.scene.create_actor_builder()
                eps = 1e-4  # 极小值，用于生成端面"贴片"且避免Z-fighting闪烁

                if level % 2 == 0:
                    half = [BLOCK_L / 2, BLOCK_W / 2, BLOCK_H / 2]
                    pos = [center_x, center_y + offset, z] 
                    
                    # 侧面主视觉
                    builder.add_box_visual(half_size=half, material=mat_side)
                    # 两端的贴片 (沿 X 轴)
                    cap_half = [eps, half[1], half[2]]
                    builder.add_box_visual(half_size=cap_half, material=mat_end, pose=sapien.Pose([half[0] + eps, 0, 0]))
                    builder.add_box_visual(half_size=cap_half, material=mat_end, pose=sapien.Pose([-half[0] - eps, 0, 0]))
                else:
                    half = [BLOCK_W / 2, BLOCK_L / 2, BLOCK_H / 2]
                    pos = [center_x + offset, center_y, z]
                    
                    # 侧面主视觉
                    builder.add_box_visual(half_size=half, material=mat_side)
                    # 两端的贴片 (沿 Y 轴)
                    cap_half = [half[0], eps, half[2]]
                    builder.add_box_visual(half_size=cap_half, material=mat_end, pose=sapien.Pose([0, half[1] + eps, 0]))
                    builder.add_box_visual(half_size=cap_half, material=mat_end, pose=sapien.Pose([0, -half[1] - eps, 0]))

                # 物理碰撞体保持为一个完整的 Box (不加贴片)
                builder.add_box_collision(
                    half_size=half, material=wood_mat, density=BLOCK_DENSITY
                )
                
                builder.initial_pose = sapien.Pose(p=pos)
                block = builder.build(name=f"block_{level}_{i}")
                self.blocks.append(block)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            center_x = 0.2  
            center_y = 0.0  
            
            for level in range(self.num_levels):
                # 同样加上垂直间隙
                z = level * (BLOCK_H + self.gap_h) + BLOCK_H / 2
                for i in range(3):
                    # 同样加上水平间隙
                    offset = (i - 1) * (BLOCK_W + self.gap_w)
                    
                    if level % 2 == 0:
                        pos = [center_x, center_y + offset, z]
                    else:
                        pos = [center_x + offset, center_y, z]
                        
                    idx = level * 3 + i
                    xyz = torch.tensor([pos], device=self.device).expand(b, -1)
                    self.blocks[idx].set_pose(Pose.create_from_pq(xyz))
    
    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            block_poses = torch.stack(
                [b.pose.raw_pose for b in self.blocks], dim=1
            )
            obs["block_poses"] = block_poses.flatten(start_dim=1)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)


# ─── 独立运行: 可视化验证 + Physical Graph 提取 ───
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "JengaTower-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend="cpu",
    )

    obs, _ = env.reset(seed=42)
    print(f"观测维度: {env.observation_space}")
    print(f"动作维度: {env.action_space}")
    print(f"Jenga 塔: {NUM_LEVELS} 层, {NUM_BLOCKS} 块")

    # 预热: 零动作步进, 让接触力稳定
    zero_action = np.zeros(env.action_space.shape)
    for _ in range(10):
        env.step(zero_action)

    # ─── 提取 Physical Graph ───
    uw = env.unwrapped
    graph = get_support_graph(uw.scene, uw.blocks)

    print(f"\n{'='*50}")
    print(f"  Physical Graph  (threshold = 10% block weight)")
    print(f"{'='*50}")
    print(f"  木块体积: {graph['volumes'][0]:.4e} m³  (共 {len(graph['volumes'])} 块)")
    print(f"  高度范围: [{min(graph['heights']):.4f}, {max(graph['heights']):.4f}] m")

    print(f"\n  支撑边 (i → j 表示 block_i 支撑 block_j):")
    edge_count = 0
    for i, row in enumerate(graph["support_matrix"]):
        for j, v in enumerate(row):
            if v:
                li, pi = divmod(i, 3)
                lj, pj = divmod(j, 3)
                print(f"    L{li}[{pi}] (#{i:2d}) → L{lj}[{pj}] (#{j:2d})")
                edge_count += 1
    print(f"  共 {edge_count} 条支撑边\n")

    # ─── 可视化 Physical Graph ───
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    n = len(graph["heights"])
    # 每块的 (x, y, z) 重心坐标
    block_xyz = []
    for b in uw.blocks:
        p = b.pose.p[0].cpu().numpy()
        block_xyz.append(p)
    block_xyz = np.array(block_xyz)

    sup = graph["support_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # ────── 左图: 正面侧视图 (X-Z 平面) ──────
    ax = axes[0]
    ax.set_title("Support Graph — Front View (X-Z)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    # 画木块矩形
    for idx in range(n):
        level, pos_in_level = divmod(idx, 3)
        cx, cy, cz = block_xyz[idx]
        if level % 2 == 0:
            hw, hh = BLOCK_L / 2, BLOCK_H / 2
        else:
            hw, hh = BLOCK_W / 2, BLOCK_H / 2
        rect = patches.FancyBboxPatch(
            (cx - hw, cz - hh), 2 * hw, 2 * hh,
            boxstyle="round,pad=0.001",
            linewidth=0.8, edgecolor="saddlebrown", facecolor="bisque", alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(cx, cz, f"{idx}", ha="center", va="center", fontsize=6, fontweight="bold")

    # 画支撑边
    for i in range(n):
        for j in range(n):
            if sup[i][j]:
                ax.annotate(
                    "", xy=(block_xyz[j, 0], block_xyz[j, 2]),
                    xytext=(block_xyz[i, 0], block_xyz[i, 2]),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=0.7, alpha=0.5),
                )

    ax.set_aspect("equal")
    ax.margins(0.1)

    # ────── 右图: 俯视图 (X-Y 平面), 按层着色 ──────
    ax = axes[1]
    ax.set_title("Support Graph — Top View (X-Y)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    cmap = plt.cm.viridis
    for idx in range(n):
        level, pos_in_level = divmod(idx, 3)
        cx, cy, cz = block_xyz[idx]
        if level % 2 == 0:
            hw, hd = BLOCK_L / 2, BLOCK_W / 2
        else:
            hw, hd = BLOCK_W / 2, BLOCK_L / 2
        color = cmap(level / max(NUM_LEVELS - 1, 1))
        rect = patches.Rectangle(
            (cx - hw, cy - hd), 2 * hw, 2 * hd,
            linewidth=0.6, edgecolor="black", facecolor=color, alpha=0.6
        )
        ax.add_patch(rect)
        ax.text(cx, cy, f"{idx}", ha="center", va="center", fontsize=5, fontweight="bold", color="white")

    for i in range(n):
        for j in range(n):
            if sup[i][j]:
                dx = block_xyz[j, 0] - block_xyz[i, 0]
                dy = block_xyz[j, 1] - block_xyz[i, 1]
                ax.annotate(
                    "", xy=(block_xyz[j, 0], block_xyz[j, 1]),
                    xytext=(block_xyz[i, 0], block_xyz[i, 1]),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=0.6, alpha=0.4),
                )

    ax.set_aspect("equal")
    ax.margins(0.15)

    # ────── 颜色条表示层号 ──────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, NUM_LEVELS - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], shrink=0.6, label="Layer Index")

    plt.tight_layout()
    save_path = "support_graph.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  支撑图已保存到: {save_path}")
    plt.show(block=False)

    # ─── 邻接矩阵热力图 ───
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    sup_arr = np.array(sup, dtype=float)
    ax2.imshow(sup_arr, cmap="OrRd", interpolation="nearest", origin="upper")
    ax2.set_title(f"Support Adjacency Matrix  ({edge_count} edges)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Block j  (supported)")
    ax2.set_ylabel("Block i  (supporter)")

    for lv in range(1, NUM_LEVELS):
        pos = lv * 3 - 0.5
        ax2.axhline(pos, color="blue", lw=0.4, alpha=0.5)
        ax2.axvline(pos, color="blue", lw=0.4, alpha=0.5)

    ax2.set_xticks(range(0, n, 3))
    ax2.set_xticklabels([f"L{i}" for i in range(NUM_LEVELS)], fontsize=7)
    ax2.set_yticks(range(0, n, 3))
    ax2.set_yticklabels([f"L{i}" for i in range(NUM_LEVELS)], fontsize=7)

    plt.tight_layout()
    save_path2 = "support_matrix_heatmap.png"
    plt.savefig(save_path2, dpi=200, bbox_inches="tight")
    print(f"  邻接矩阵热力图已保存到: {save_path2}\n")
    plt.show(block=False)

    # ─── 计算 Stability 标签 ───
    poses = np.array([b.pose.p[0].cpu().numpy() for b in uw.blocks])
    s_load, s_balance = get_gt_stability(
        graph["support_matrix"], graph["volumes"], poses
    )

    print(f"{'='*50}")
    print(f"  Ground Truth Stability Labels")
    print(f"{'='*50}")
    print(f"  {'Block':>8}  {'Layer':>6}  {'s_load':>8}  {'s_balance':>10}")
    print(f"  {'-'*38}")
    for i in range(NUM_BLOCKS):
        li, pi = divmod(i, 3)
        print(f"  #{i:2d}[{pi}]   L{li:<4d}  {s_load[i]:.4f}    {s_balance[i]:.6f}")
    print()

    # ─── 可视化: Stability 双柱图 ───
    fig_stab, (ax_sl, ax_sb) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x_idx = np.arange(NUM_BLOCKS)

    ax_sl.bar(x_idx, s_load, color="#1f77b4", edgecolor="gray", linewidth=0.3)
    ax_sl.set_ylabel("s_load")
    ax_sl.set_title("Load Factor  s_load = exp(−α·V_sum)", fontsize=12, fontweight="bold")
    for lv in range(1, NUM_LEVELS):
        ax_sl.axvline(lv * 3 - 0.5, color="blue", lw=0.3, alpha=0.4, ls="--")

    cmap_sb = ["#d62728" if v < 0.5 else "#2ca02c" for v in s_balance]
    ax_sb.bar(x_idx, s_balance, color=cmap_sb, edgecolor="gray", linewidth=0.3)
    ax_sb.set_ylabel("s_balance")
    ax_sb.set_xlabel("Block Index")
    ax_sb.set_title("Balance Margin  s_balance = σ(β·M_margin)", fontsize=12, fontweight="bold")
    ax_sb.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.5)
    for lv in range(1, NUM_LEVELS):
        ax_sb.axvline(lv * 3 - 0.5, color="blue", lw=0.3, alpha=0.4, ls="--")
    ax_sb.set_xticks(range(0, NUM_BLOCKS, 3))
    ax_sb.set_xticklabels([f"L{i}" for i in range(NUM_LEVELS)], fontsize=7)

    plt.tight_layout()
    plt.savefig("stability_labels.png", dpi=150, bbox_inches="tight")
    print(f"  Stability 标签柱状图已保存到: stability_labels.png\n")
    plt.show(block=False)

    # ─── 计算 Potentiality 标签 (反事实干预) ───
    import time as _time
    print(f"{'='*50}")
    print(f"  Computing Potentiality (counterfactual, 36×100 steps)...")
    t0 = _time.time()
    potentiality = get_gt_potentiality(uw.scene, uw.blocks)
    elapsed = _time.time() - t0
    print(f"  完成, 耗时 {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"  {'Block':>8}  {'Layer':>6}  {'Potentiality':>14}")
    print(f"  {'-'*32}")
    for i in range(NUM_BLOCKS):
        li, pi = divmod(i, 3)
        print(f"  #{i:2d}[{pi}]   L{li:<4d}  {potentiality[i]:.4f}")
    print()

    # ─── 可视化: Potentiality 柱状图 ───
    fig_pot, ax_pot = plt.subplots(figsize=(14, 4))
    pot_colors = ["#d62728" if v < 0.9 else "#2ca02c" for v in potentiality]
    ax_pot.bar(range(NUM_BLOCKS), potentiality, color=pot_colors,
               edgecolor="gray", linewidth=0.3)
    ax_pot.set_ylabel("Potentiality")
    ax_pot.set_xlabel("Block Index")
    ax_pot.set_title("GT Potentiality (fraction of remaining stable blocks after removal)",
                     fontsize=11, fontweight="bold")
    ax_pot.axhline(0.9, color="black", ls=":", lw=0.8, alpha=0.5)
    for lv in range(1, NUM_LEVELS):
        ax_pot.axvline(lv * 3 - 0.5, color="blue", lw=0.3, alpha=0.4, ls="--")
    ax_pot.set_xticks(range(0, NUM_BLOCKS, 3))
    ax_pot.set_xticklabels([f"L{i}" for i in range(NUM_LEVELS)], fontsize=7)
    ax_pot.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("potentiality_labels.png", dpi=150, bbox_inches="tight")
    print(f"  Potentiality 柱状图已保存到: potentiality_labels.png\n")
    plt.show(block=False)

    # ─── 提取多视角点云 ───
    surround_cams = {uid: s for uid, s in uw.scene.sensors.items() if uid.startswith("surround")}
    pcd_data = render_point_cloud(uw.scene, surround_cams, uw.blocks)

    global_pcd = pcd_data["global_pcd"]
    per_block = pcd_data["per_block_pcd"]
    cam_rgbs = pcd_data["camera_rgbs"]

    counts = [len(p) for p in per_block]
    visible = sum(1 for c in counts if c > 0)
    occluded_ids = [i for i, c in enumerate(counts) if c == 0]

    print(f"{'='*50}")
    print(f"  Point Cloud  ({len(surround_cams)} cameras, 256×256)")
    print(f"{'='*50}")
    print(f"  全局点数: {len(global_pcd)}")
    print(f"  可见木块: {visible}/{NUM_BLOCKS}")
    if occluded_ids:
        for oid in occluded_ids:
            li, pi = divmod(oid, 3)
            print(f"    ⚠ 完全遮挡: block #{oid}  (L{li}[{pi}])")
    else:
        print(f"    所有木块均有可见点")
    print(f"  每块点数: min={min(counts)}, max={max(counts)}, "
          f"mean={np.mean(counts):.0f}, median={np.median(counts):.0f}\n")

    # ─── 可视化: 4 个相机 RGB 图 ───
    fig_cams, axes_cams = plt.subplots(1, len(cam_rgbs), figsize=(4 * len(cam_rgbs), 4))
    if len(cam_rgbs) == 1:
        axes_cams = [axes_cams]
    for ax_c, (uid, img) in zip(axes_cams, cam_rgbs.items()):
        ax_c.imshow(img)
        ax_c.set_title(uid, fontsize=10)
        ax_c.axis("off")
    plt.tight_layout()
    plt.savefig("surround_cameras_rgb.png", dpi=150, bbox_inches="tight")
    print(f"  相机 RGB 图已保存到: surround_cameras_rgb.png")
    plt.show(block=False)

    # ─── 可视化: 3D 点云 (仅木块, 下采样) ───
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    block_sid_set = {blk._objs[0].per_scene_id for blk in uw.blocks}
    block_sid_to_level = {blk._objs[0].per_scene_id: bi // 3 for bi, blk in enumerate(uw.blocks)}
    block_mask = np.isin(pcd_data["global_seg_ids"], list(block_sid_set))
    block_pcd = global_pcd[block_mask]
    block_seg = pcd_data["global_seg_ids"][block_mask]

    max_vis_pts = 30000
    if len(block_pcd) > max_vis_pts:
        idx_sample = np.random.choice(len(block_pcd), max_vis_pts, replace=False)
        vis_pcd = block_pcd[idx_sample]
        vis_seg = block_seg[idx_sample]
    else:
        vis_pcd = block_pcd
        vis_seg = block_seg

    fig3d = plt.figure(figsize=(16, 7))

    ax3 = fig3d.add_subplot(121, projection="3d")
    ax3.scatter(vis_pcd[:, 0], vis_pcd[:, 1], vis_pcd[:, 2],
                c=vis_pcd[:, 3:6].clip(0, 1), s=0.3, alpha=0.6)
    ax3.set_title("Block Point Cloud (RGB)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_box_aspect([1, 1, 2])

    ax3b = fig3d.add_subplot(122, projection="3d")
    level_colors = np.array([block_sid_to_level[s] for s in vis_seg], dtype=float)
    sc = ax3b.scatter(vis_pcd[:, 0], vis_pcd[:, 1], vis_pcd[:, 2],
                      c=level_colors, cmap="viridis", s=0.3, alpha=0.6,
                      vmin=0, vmax=NUM_LEVELS - 1)
    ax3b.set_title("Block Point Cloud (by Layer)", fontsize=12, fontweight="bold")
    ax3b.set_xlabel("X"); ax3b.set_ylabel("Y"); ax3b.set_zlabel("Z")
    ax3b.set_box_aspect([1, 1, 2])
    fig3d.colorbar(sc, ax=ax3b, shrink=0.5, label="Layer")

    plt.tight_layout()
    plt.savefig("point_cloud_3d.png", dpi=150, bbox_inches="tight")
    print(f"  3D 点云图已保存到: point_cloud_3d.png")
    plt.show(block=False)

    # ─── 可视化: 每块点数柱状图 ───
    fig_bar, ax_bar = plt.subplots(figsize=(14, 4))
    colors_bar = ["#d62728" if c == 0 else "#2ca02c" for c in counts]
    ax_bar.bar(range(NUM_BLOCKS), counts, color=colors_bar, edgecolor="gray", linewidth=0.3)
    ax_bar.set_xlabel("Block Index")
    ax_bar.set_ylabel("Point Count")
    ax_bar.set_title("Per-Block Point Count (red = fully occluded)", fontsize=12, fontweight="bold")
    for lv in range(1, NUM_LEVELS):
        ax_bar.axvline(lv * 3 - 0.5, color="blue", lw=0.3, alpha=0.4, ls="--")
    ax_bar.set_xticks(range(0, NUM_BLOCKS, 3))
    ax_bar.set_xticklabels([f"L{i}" for i in range(NUM_LEVELS)], fontsize=7)
    plt.tight_layout()
    plt.savefig("per_block_point_count.png", dpi=150, bbox_inches="tight")
    print(f"  每块点数柱状图已保存到: per_block_point_count.png\n")
    plt.show(block=False)

    viewer = env.render()
    if hasattr(viewer, "paused"):
        viewer.paused = True

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        try:
            env.render()
        except (TypeError, AttributeError):
            break
        if (terminated | truncated).any():
            obs, _ = env.reset()

    env.close()
