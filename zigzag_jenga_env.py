"""
ZigZagJengaEnv: Z字形脊柱 Jenga 塔 — 用于测试 OOD 泛化与长程物理推理。

第 4~10 层形成交替的 Z 字形单边支撑结构，
第 11 层及以上保持完整作为顶部配重。
"""
import torch
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from jenga_tower import JengaTowerEnv


@register_env("JengaTower-ZigZag-v1", max_episode_steps=100)
class ZigZagJengaEnv(JengaTowerEnv):

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        far_pose = torch.tensor([[999.0, 999.0, 999.0]], device=self.device)
        zero3 = torch.zeros(1, 3, device=self.device)

        for block in self.blocks:
            block._bodies[0].kinematic = True
            block.set_linear_velocity(zero3)
            block.set_angular_velocity(zero3)

        for level in range(4, 11):
            remove = [1]
            if level % 2 == 0:
                remove.append(2)
            else:
                remove.append(0)

            for i in remove:
                idx = level * 3 + i
                self.blocks[idx].set_pose(Pose.create_from_pq(far_pose.expand(b, -1)))


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "JengaTower-ZigZag-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend="cpu",
    )
    obs, _ = env.reset(seed=42)

    print("ZigZag Z字形脊柱: 第 4~10 层交替单边支撑")
    for level in range(4, 11):
        kept = 0 if level % 2 == 0 else 2
        print(f"  L{level:2d}: 保留 i={kept}")

    viewer = env.render()
    if hasattr(viewer, "paused"):
        viewer.paused = True

    zero_action = np.zeros(env.action_space.shape)
    while True:
        env.step(zero_action)
        try:
            env.render()
        except (TypeError, AttributeError):
            break

    env.close()
