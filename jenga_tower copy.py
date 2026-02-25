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
NUM_LEVELS = 12

# 每层 3 块木块的颜色 (浅木 / 中木 / 深木)
BLOCK_COLORS = [
    [0.82, 0.68, 0.47, 1.0],
    [0.65, 0.50, 0.32, 1.0],
    [0.48, 0.35, 0.20, 1.0],
]


@register_env("JengaTower-v1", max_episode_steps=100)
class JengaTowerEnv(BaseEnv):
    """
    **Task Description:**
    Jenga 塔环境 — 18 层、每层 3 块、相邻层 90° 交替。
    塔放置在桌面上，Panda 机械臂可与之交互。

    **Success Conditions:**
    Placeholder — 后续定义抽取/堆叠任务。
    """

    SUPPORTED_ROBOTS = ["panda"]
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
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

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
                    half_size=half, material=wood_mat, density=600
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


# ─── 独立运行: 可视化验证 ───
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
    print(f"Jenga 塔: {NUM_LEVELS} 层, {NUM_LEVELS * 3} 块")

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
