"""
RealisticJengaEnvA: 逼真残局 A — 掏心/独木/不对称交替的中后期残局。
"""
import torch
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from jenga_tower import JengaTowerEnv


KEEP_MASK = [
    [1, 1, 1],  # L0: 坚固底座
    [0, 1, 0],  # L1: 稳固过渡
    [0, 1, 0],  # L2: 单点故障链起点
    [0, 1, 0],  # L3: 极限单柱
    [0, 1, 0],  # L4: 极限单柱
    [0, 1, 0],  # L5: 极限单柱
    [0, 1, 0],  # L6: 极限单柱
    [0, 1, 0],  # L7: 悬挑承台
    [1, 0, 1],  # L8: 掏空中心
    [0, 1, 0],  # L9: 第二段单柱起点
    [0, 1, 0],  # L10: 极限单柱
    [0, 1, 0],  # L11: 极限单柱
    [0, 1, 0],  # L12: 悬挑承台
    [0, 1, 0],  # L13: 第三段单柱起点
    [0, 1, 0],  # L14: 极限单柱
    [0, 1, 0],  # L15: 极限单柱
    [0, 1, 0],  # L16: 稳固过渡
    [1, 1, 1],  # L17: 顶部保留
]

#paper
# KEEP_MASK = [
#     [1, 1, 1],  # L0:  坚固底座
#     [1, 1, 1],  # L1
#     [1, 1, 1],  # L2
#     [1, 0, 1],  # L3:  掏空中心
#     [0, 1, 0],  # L4:  独木支撑 (十字交叉)
#     [1, 0, 1],  # L5:  掏空中心
#     [0, 1, 0],  # L6:  独木支撑
#     [1, 1, 0],  # L7:  不对称 (偏左)
#     [1, 1, 0],  # L8:  不对称 (偏左)
#     [1, 1, 1],  # L9:  稳固过渡层
#     [0, 1, 1],  # L10: 不对称 (偏右)
#     [0, 1, 1],  # L11: 不对称 (偏右)
#     [1, 0, 1],  # L12: 掏空中心
#     [0, 1, 0],  # L13: 独木支撑
#     [1, 1, 1],  # L14: 顶部保留
#     [1, 1, 1],  # L15
#     [0, 0, 0],  # L16: 减重移除
#     [0, 0, 0],  # L17: 减重移除
# ]

#general1
# KEEP_MASK = [
#     [1, 1, 1],  # L0:  绝对坚固底座
#     [1, 1, 1],  # L1
#     [1, 1, 1],  # L2
#     [1, 1, 1],  # L3
#     [1, 1, 1],  # L4
#     [1, 1, 1],  # L5
#     [1, 1, 1],  # L6
#     [1, 1, 1],  # L7
#     [1, 1, 1],  # L8
#     [1, 1, 1],  # L9
#     [1, 1, 1],  # L10
#     [1, 1, 1],  # L11
#     [1, 1, 1],  # L12
#     [1, 1, 1],  # L13
#     [1, 1, 1],  # L14
#     [1, 1, 1],  # L15
#     [1, 1, 1],  # L16
#     [1, 1, 1],  # L17: 绝对完整顶层
# ]

# general2
# KEEP_MASK = [
#     [1, 1, 1],  # L0: 坚固底座
#     [1, 0, 1],  # L1: 掏空中心
#     [0, 1, 0],  # L2: 独木支撑
#     [1, 0, 1],  # L3: 掏空中心
#     [1, 1, 1],  # L4: 结构重置层
#     [1, 1, 0],  # L5: 轻微偏左
#     [0, 1, 1],  # L6: 轻微偏右
#     [1, 1, 1],  # L7: 结构重置层
#     [1, 0, 1],  # L8: 掏空中心
#     [1, 1, 1],  # L9: 结构重置层
#     [0, 1, 0],  # L10: 独木支撑
#     [1, 0, 1],  # L11: 掏空中心
#     [1, 1, 1],  # L12: 结构重置层
#     [1, 1, 0],  # L13: 轻微偏左
#     [1, 0, 1],  # L14: 掏空中心
#     [0, 1, 0],  # L15: 独木支撑
#     [1, 1, 1],  # L16: 稳固过渡
#     [1, 1, 1],  # L17: 顶部保留
# ]

# OOD1
# KEEP_MASK = [
#     [1, 1, 1],  # L0: 坚固底座
#     [1, 1, 1],  # L1: 稳固过渡
#     [0, 1, 0],  # L2: 单点故障链起点
#     [0, 1, 0],  # L3: 极限单柱
#     [0, 1, 0],  # L4: 极限单柱
#     [0, 1, 0],  # L5: 极限单柱
#     [0, 1, 0],  # L6: 极限单柱
#     [1, 1, 1],  # L7: 悬挑承台
#     [1, 0, 1],  # L8: 掏空中心
#     [0, 1, 0],  # L9: 第二段单柱起点
#     [0, 1, 0],  # L10: 极限单柱
#     [0, 1, 0],  # L11: 极限单柱
#     [1, 1, 1],  # L12: 悬挑承台
#     [0, 1, 0],  # L13: 第三段单柱起点
#     [0, 1, 0],  # L14: 极限单柱
#     [0, 1, 0],  # L15: 极限单柱
#     [1, 1, 1],  # L16: 稳固过渡
#     [1, 1, 1],  # L17: 顶部保留
# ]

# OOD2
# KEEP_MASK = [
#     [1, 1, 1],  # L0:  绝对坚固底座
#     [1, 1, 1],  # L1
#     [1, 0, 1],  # L2:  掏空中心
#     [0, 1, 0],  # L3:  十字交叉单点
#     [1, 0, 1],  # L4:  两翼支撑，无中心
#     [0, 1, 0],  # L5:  十字交叉单点
#     [0, 1, 0],  # L6:  连续十字交叉单点 (极度脆弱)
#     [1, 0, 1],  # L7:  两翼支撑，无中心
#     [0, 1, 0],  # L8:  全局致命瓶颈层 (The Bottleneck)
#     [1, 0, 1],  # L9:  两翼支撑，无中心
#     [0, 1, 0],  # L10: 连续十字交叉单点
#     [0, 1, 0],  # L11: 连续十字交叉单点 (极度脆弱)
#     [1, 0, 1],  # L12: 两翼支撑，无中心
#     [0, 1, 0],  # L13: 十字交叉单点
#     [1, 0, 1],  # L14: 掏空中心
#     [1, 1, 1],  # L15
#     [1, 1, 1],  # L16
#     [1, 1, 1],  # L17: 绝对完整顶层
# ]

@register_env("JengaTower-RealisticA-v1", max_episode_steps=100)
class RealisticJengaEnvA(JengaTowerEnv):

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
        "JengaTower-RealisticA-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
        sim_backend="cpu",
    )
    obs, _ = env.reset(seed=42)

    kept = sum(sum(row) for row in KEEP_MASK)
    print(f"RealisticA 残局: 保留 {kept} 块, 移除 {54 - kept} 块")
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
