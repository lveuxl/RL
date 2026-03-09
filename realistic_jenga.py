"""
RealisticJengaEnvA: 逼真残局 A — 掏心/独木/不对称交替的中后期残局。
"""
import torch
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from jenga_tower import JengaTowerEnv

KEEP_MASK = [
    [1, 1, 1],  # L0:  坚固底座
    [1, 1, 0],  # L1
    [0, 1, 1],  # L2
    [1, 0, 1],  # L3:  掏空中心
    [1, 1, 1],  # L4:  独木支撑 (十字交叉)
    [0, 0, 0],  # L5:  掏空中心
    [0, 0, 0],  # L6:  独木支撑
    [0, 0, 0],  # L7:  不对称 (偏左)
    [0, 0, 0],  # L8:  不对称 (偏左)
    [0, 0, 0],  # L9:  稳固过渡层
    [0, 0, 0],  # L10: 不对称 (偏右)
    [0, 0, 0],  # L11: 不对称 (偏右)
    [0, 0, 0],  # L12: 掏空中心
    [0, 0, 0],  # L13: 独木支撑
    [0, 0, 0],  # L14: 顶部保留
    [0, 0, 0],  # L15
    [0, 0, 0],  # L16: 减重移除
    [0, 0, 0],  # L17: 减重移除
]

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
@register_env("JengaTower-RealisticA-v1", max_episode_steps=100)
class RealisticJengaEnvA(JengaTowerEnv):

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        far_pose = torch.tensor([[999.0, 999.0, 999.0]], device=self.device)

        for level in range(18):
            for i in range(3):
                if KEEP_MASK[level][i] == 0:
                    idx = level * 3 + i
                    self.blocks[idx].set_pose(Pose.create_from_pq(far_pose.expand(b, -1)))


if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from jenga_tower import render_point_cloud, NUM_LEVELS, NUM_BLOCKS

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

    zero_action = np.zeros(env.action_space.shape)
    for _ in range(10):
        env.step(zero_action)

    # ─── 提取多视角点云 ───
    uw = env.unwrapped
    surround_cams = {uid: s for uid, s in uw.scene.sensors.items() if uid.startswith("surround")}
    pcd_data = render_point_cloud(uw.scene, surround_cams, uw.blocks)

    global_pcd = pcd_data["global_pcd"]
    per_block = pcd_data["per_block_pcd"]
    cam_rgbs = pcd_data["camera_rgbs"]

    counts = [len(p) for p in per_block]
    visible = sum(1 for c in counts if c > 0)
    print(f"\n  全局点数: {len(global_pcd)}, 可见木块: {visible}/{NUM_BLOCKS}")

    # ─── 可视化: 4 个相机 RGB 图 ───
    fig_cams, axes_cams = plt.subplots(1, len(cam_rgbs), figsize=(4 * len(cam_rgbs), 4))
    if len(cam_rgbs) == 1:
        axes_cams = [axes_cams]
    for ax_c, (uid, img) in zip(axes_cams, cam_rgbs.items()):
        ax_c.imshow(img)
        ax_c.set_title(uid, fontsize=10)
        ax_c.axis("off")
    plt.tight_layout()
    plt.savefig("realistic_a_cameras_rgb.png", dpi=150, bbox_inches="tight")
    print(f"  相机 RGB 图已保存到: realistic_a_cameras_rgb.png")
    plt.show(block=False)

    # ─── 可视化: 3D 点云 (仅木块, 下采样) ───
    block_sid_set = {blk._objs[0].per_scene_id for blk in uw.blocks}
    block_sid_to_level = {blk._objs[0].per_scene_id: bi // 3 for bi, blk in enumerate(uw.blocks)}
    block_mask = np.isin(pcd_data["global_seg_ids"], list(block_sid_set))
    block_pcd = global_pcd[block_mask]
    block_seg = pcd_data["global_seg_ids"][block_mask]

    max_vis_pts = 30000
    if len(block_pcd) > max_vis_pts:
        idx_sample = np.random.choice(len(block_pcd), max_vis_pts, replace=False)
        vis_pcd, vis_seg = block_pcd[idx_sample], block_seg[idx_sample]
    else:
        vis_pcd, vis_seg = block_pcd, block_seg

    fig3d = plt.figure(figsize=(16, 7))

    ax3 = fig3d.add_subplot(121, projection="3d")
    ax3.scatter(vis_pcd[:, 0], vis_pcd[:, 1], vis_pcd[:, 2],
                c=vis_pcd[:, 3:6].clip(0, 1), s=0.3, alpha=0.6)
    ax3.set_title("RealisticA — Point Cloud (RGB)", fontsize=12, fontweight="bold")
    ax3.set_box_aspect([1, 1, 2])
    
    # ======== 关键修改点 1 ========
    # ax3.set_axis_off()  # 注释掉完全关闭坐标轴的代码，以保留网格和轴线
    ax3.set_xticklabels([])  # 隐藏 X 轴数值
    ax3.set_yticklabels([])  # 隐藏 Y 轴数值
    ax3.set_zticklabels([])  # 隐藏 Z 轴数值
    # ===============================

    ax3b = fig3d.add_subplot(122, projection="3d")
    level_colors = np.array([block_sid_to_level[s] for s in vis_seg], dtype=float)
    sc = ax3b.scatter(vis_pcd[:, 0], vis_pcd[:, 1], vis_pcd[:, 2],
                      c=level_colors, cmap="viridis", s=0.3, alpha=0.6,
                      vmin=0, vmax=NUM_LEVELS - 1)
    ax3b.set_title("RealisticA — Point Cloud (by Layer)", fontsize=12, fontweight="bold")
    ax3b.set_box_aspect([1, 1, 2])
    ax3b.locator_params(axis="x", nbins=4)
    ax3b.locator_params(axis="y", nbins=4)
    ax3b.locator_params(axis="z", nbins=5)
    ax3b.tick_params(axis="both", labelsize=7, pad=1)
    
    # ======== 关键修改点 2 ========
    ax3b.set_xticklabels([])  # 隐藏 X 轴数值
    ax3b.set_yticklabels([])  # 隐藏 Y 轴数值
    ax3b.set_zticklabels([])  # 隐藏 Z 轴数值
    # ===============================
    
    fig3d.colorbar(sc, ax=ax3b, shrink=0.5, label="Layer")

    plt.tight_layout()
    plt.savefig("realistic_a_point_cloud_3d.png", dpi=150, bbox_inches="tight")
    print(f"  3D 点云图已保存到: realistic_a_point_cloud_3d.png")
    plt.show(block=False)

    viewer = env.render()
    if hasattr(viewer, "paused"):
        viewer.paused = True

    while True:
        env.step(zero_action)
        try:
            env.render()
        except (TypeError, AttributeError):
            break

    env.close()