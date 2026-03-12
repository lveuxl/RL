"""
RealisticJengaEnvB: 逼真残局 B — "比萨斜塔"单侧连续抽取 + 反向配重。
"""
import torch
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from jenga_tower import JengaTowerEnv

KEEP_MASK = [
    [1, 1, 1],  # L0:  底座
    [1, 1, 1],  # L1
    [1, 1, 1],  # L2
    [1, 1, 0],  # L3:  抽右侧
    [1, 1, 1],  # L4
    [1, 1, 0],  # L5:  抽右侧
    [1, 1, 1],  # L6
    [1, 1, 0],  # L7:  抽右侧
    [0, 1, 0],  # L8:  单点支撑
    [0, 1, 1],  # L9:  反向配重 (偏右)
    [0, 1, 1],  # L10: 反向配重 (偏右)
    [1, 0, 1],  # L11: 掏空中心
    [1, 1, 1],  # L12: 稳固顶部
    [1, 1, 1],  # L13
    [0, 0, 0],  # L14: 移除
    [0, 0, 0],  # L15: 移除
    [0, 0, 0],  # L16: 移除
    [0, 0, 0],  # L17: 移除
]

# KEEP_MASK = [
#     [1, 1, 1],  # L0:  底座
#     [1, 1, 1],  # L1
#     [1, 1, 1],  # L2
#     [1, 1, 0],  # L3:  抽右侧
#     [1, 1, 1],  # L4
#     [1, 1, 0],  # L5:  抽右侧
#     [1, 1, 1],  # L6
#     [1, 1, 0],  # L7:  抽右侧
#     [0, 1, 0],  # L8:  单点支撑
#     [0, 1, 1],  # L9:  反向配重 (偏右)
#     [0, 1, 1],  # L10: 反向配重 (偏右)
#     [1, 0, 1],  # L11: 掏空中心
#     [1, 1, 1],  # L12: 稳固顶部
#     [1, 1, 1],  # L13
#     [0, 0, 0],  # L14: 移除
#     [0, 0, 0],  # L15: 移除
#     [0, 0, 0],  # L16: 移除
#     [0, 0, 0],  # L17: 移除
# ]


@register_env("JengaTower-RealisticB-v1", max_episode_steps=100)
class RealisticJengaEnvB(JengaTowerEnv):

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        far_pose = torch.tensor([[999.0, 999.0, 999.0]], device=self.device)
        # zero3 = torch.zeros(1, 3, device=self.device)

        # for block in self.blocks:
        #     block._bodies[0].kinematic = True
        #     block.set_linear_velocity(zero3)
        #     block.set_angular_velocity(zero3)

        for level in range(18):
            for i in range(3):
                if KEEP_MASK[level][i] == 0:
                    idx = level * 3 + i
                    self.blocks[idx].set_pose(Pose.create_from_pq(far_pose.expand(b, -1)))


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "JengaTower-RealisticB-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend="cpu",
    )
    obs, _ = env.reset(seed=42)

    kept = sum(sum(row) for row in KEEP_MASK)
    print(f"RealisticB 比萨斜塔残局: 保留 {kept} 块, 移除 {54 - kept} 块")
    for lv, mask in enumerate(KEEP_MASK):
        print(f"  L{lv:2d}: {''.join('■' if k else '□' for k in mask)}")

    viewer = env.render()
    # if hasattr(viewer, "paused"):
    #     viewer.paused = True

    zero_action = np.zeros(env.action_space.shape)
    while True:
        env.step(zero_action)
        try:
            env.render()
        except (TypeError, AttributeError):
            break

    env.close()
