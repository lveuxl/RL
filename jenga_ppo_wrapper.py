"""
JengaPPOWrapper — 在线 PPO 训练用 Gym Wrapper

功能:
  1. 离散动作空间 Discrete(N): 选择哪个积木抽取
  2. 先验引导奖励塑形: reward = r_ext + λ·p_pot_i
  3. 每步快速 GT (support_matrix + stability) → info
  4. 选中积木的 gt_potentiality_i 从实际抽取中免费获得
  5. Episode 结束后 compute_distill_gt() 批量回溯完整 potentiality [N]

数据流:
  PPO Loop:
    1. obs = env.reset()
    2. V-P3E(obs) → Z, stab, pot        (frozen)
    3. env.set_prior_scores(pot)
    4. ActorCritic(Z, stab, pot, mask) → action_i
    5. obs, reward, done, _, info = env.step(action_i)
    6. PPO Buffer 收集 info 中的 GT 数据
  Episode End:
    distill_data = env.compute_distill_gt()   → V-P3E 在线蒸馏
"""
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from mani_skill.utils.structs.pose import Pose
from jenga_tower import get_support_graph, get_gt_stability


class JengaPPOWrapper(gym.Wrapper):
    """
    高层 PPO Wrapper: 离散积木选择 + 先验奖励塑形 + GT 蒸馏数据收集。

    动作: Discrete(N_blocks) — 选择抽取的积木索引
    奖励: reward_ext + λ·p_pot_i
    观测: block_poses [N*3] ∥ mask [N]  (flat float32)
    """

    def __init__(self, env, lambda_int=0.1, settle_steps=80,
                 collapse_threshold=0.03, collapse_ratio=0.9,
                 reward_ok=1.0, reward_collapse=-10.0, reward_invalid=-5.0):
        super().__init__(env)
        uw = self.unwrapped
        self.n = len(uw.blocks)

        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n * 4,), dtype=np.float32
        )

        self.lambda_int = lambda_int
        self.settle_steps = settle_steps
        self.collapse_threshold = collapse_threshold
        self.collapse_ratio = collapse_ratio
        self.r_ok, self.r_col, self.r_inv = reward_ok, reward_collapse, reward_invalid

        self._removed = None
        self._step_count = 0
        self._p_pot = None
        self._buf = []

    # ──────────────── 外部接口 ────────────────

    def set_prior_scores(self, p_pot):
        """step 前注入 V-P3E 预测的潜能分数 [N]"""
        self._p_pot = p_pot

    def reset(self, target_c=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._removed = np.zeros(self.n, dtype=bool)
        self._step_count = 0
        self._p_pot = None
        self._buf = []

        # 物理预热: 让接触力收敛
        uw = self.unwrapped
        zero_act = np.zeros(self.env.action_space.shape)
        for _ in range(10):
            self.env.step(zero_act)

        # 课程学习: 根据 target_c 配置塔难度
        if target_c is not None:
            self._configure_difficulty(target_c)

        info.update(self._quick_gt())
        info["mask"] = ~self._removed
        return self._obs(), info

    def _configure_difficulty(self, target_c: float):
        """
        根据目标复杂度 c ∈ [0, 1] 配置塔难度。

        c → num_layers:  3 + c*15  (3层 → 18层)
        c → p_drop:      高c少空洞 (更难), 低c多空洞 (更简单)
        """
        uw = self.unwrapped
        device = uw.blocks[0].device

        num_layers = int(np.clip(3 + target_c * 15, 3, 18))
        p_drop = 0.2 * (1 - target_c * 0.8)

        far = torch.tensor([[999., 999., 999.]], device=device)
        zero3 = torch.zeros(1, 3, device=device)

        # 移除超出层数的积木
        for idx in range(num_layers * 3, self.n):
            uw.blocks[idx].set_pose(Pose.create_from_pq(far))
            uw.blocks[idx].set_linear_velocity(zero3)
            uw.blocks[idx].set_angular_velocity(zero3)
            self._removed[idx] = True

        # 随机移除部分中间积木 (保留底层)
        removable = [i for i in range(3, num_layers * 3)]
        n_drop = int(len(removable) * p_drop)
        if n_drop > 0:
            for idx in np.random.choice(removable, n_drop, replace=False):
                uw.blocks[idx].set_pose(Pose.create_from_pq(far))
                uw.blocks[idx].set_linear_velocity(zero3)
                uw.blocks[idx].set_angular_velocity(zero3)
                self._removed[idx] = True

        for _ in range(30):
            uw.scene.step()

    def step(self, action):
        i = int(action)
        uw = self.unwrapped
        device = uw.blocks[0].device
        self._step_count += 1

        # ① Pre-extraction GT (fast: 接触力 + 解析稳定性)
        gt = self._quick_gt()

        # ② 非法动作: 已移除的积木
        if self._removed[i]:
            info = {**gt, "mask": ~self._removed.copy(),
                    "gt_potentiality_i": 0.0, "action": i, "collapsed": False}
            return self._obs(), self.r_inv, False, False, info

        # ③ 保存抽取前快照
        removed_snap = self._removed.copy()
        init_pos = torch.stack([a.pose.p[0] for a in uw.blocks])
        sim_state = uw.scene.get_sim_state()

        # ④ 抽取 block_i → 物理稳定
        far = torch.tensor([[999., 999., 999.]], device=device)
        zero3 = torch.zeros(1, 3, device=device)
        uw.blocks[i].set_pose(Pose.create_from_pq(far))
        uw.blocks[i].set_linear_velocity(zero3)
        uw.blocks[i].set_angular_velocity(zero3)

        for _ in range(self.settle_steps):
            uw.scene.step()

        # ⑤ 位移检测 → 坍塌判定 + gt_potentiality_i (免费获得)
        n_rem = int((~self._removed).sum()) - 1
        stable = sum(
            1 for j in range(self.n)
            if j != i and not self._removed[j]
            and torch.norm(uw.blocks[j].pose.p[0] - init_pos[j]).item()
            < self.collapse_threshold
        )
        gt_pot_i = stable / max(n_rem, 1)
        collapsed = gt_pot_i < self.collapse_ratio

        # ⑥ 状态更新
        self._removed[i] = True
        done = collapsed or self._removed.all()

        # ⑦ 奖励 = 外部 + λ·先验
        r_ext = self.r_col if collapsed else self.r_ok
        p_pot_i = float(self._p_pot[i]) if self._p_pot is not None else 0.0
        reward = r_ext + self.lambda_int * p_pot_i

        # ⑧ Episode Buffer (存 sim_state 供批量回溯)
        self._buf.append({
            "step": self._step_count, "action": i,
            "sim_state": sim_state, "removed_before": removed_snap,
            "gt_support_matrix": gt["gt_support_matrix"],
            "gt_stability": gt["gt_stability"],
            "gt_potentiality_i": gt_pot_i,
        })

        # ⑨ info
        info = {
            **gt,
            "mask": ~self._removed.copy(),
            "gt_potentiality_i": gt_pot_i,
            "action": i, "collapsed": collapsed,
            "reward_ext": r_ext, "p_pot_i": p_pot_i,
        }
        return self._obs(), reward, done, False, info

    # ──────────────── GT 蒸馏 ────────────────

    def compute_distill_gt(self):
        """
        Episode 结束后批量回溯: 恢复每步 sim_state,
        计算完整 gt_potentiality [N] 向量。

        Returns:
            list[dict]: 每步含完整 GT 数据 (N 维 potentiality)
        """
        uw = self.unwrapped
        current_state = uw.scene.get_sim_state()

        results = []
        for entry in self._buf:
            uw.scene.set_sim_state(entry["sim_state"])
            removed = entry["removed_before"]
            active = [j for j in range(self.n) if not removed[j]]

            pot = np.zeros(self.n, dtype=np.float32)
            for j in active:
                pot[j] = self._single_pot(j, removed)

            results.append({
                "step": entry["step"],
                "action": entry["action"],
                "gt_support_matrix": entry["gt_support_matrix"],
                "gt_stability": entry["gt_stability"],
                "gt_potentiality": pot,
                "mask": ~removed,
            })

        uw.scene.set_sim_state(current_state)
        return results

    # ──────────────── 内部工具 ────────────────

    def _quick_gt(self):
        """每步快速 GT: 接触力 → support_matrix, 解析公式 → stability"""
        uw = self.unwrapped
        graph = get_support_graph(uw.scene, uw.blocks)
        poses = np.array([b.pose.p[0].cpu().numpy() for b in uw.blocks])
        s_load, s_balance = get_gt_stability(
            graph["support_matrix"], graph["volumes"], poses
        )
        return {
            "gt_support_matrix": np.array(graph["support_matrix"], dtype=np.float32),
            "gt_stability": (s_load * s_balance).astype(np.float32),
        }

    def _single_pot(self, idx, removed):
        """单块反事实模拟: 移除 idx → 剩余稳定比例"""
        uw = self.unwrapped
        device = uw.blocks[0].device

        state = uw.scene.get_sim_state()
        init_pos = torch.stack([a.pose.p[0] for a in uw.blocks])

        uw.blocks[idx].set_pose(
            Pose.create_from_pq(torch.tensor([[999., 999., 999.]], device=device))
        )
        uw.blocks[idx].set_linear_velocity(torch.zeros(1, 3, device=device))
        uw.blocks[idx].set_angular_velocity(torch.zeros(1, 3, device=device))

        for _ in range(self.settle_steps):
            uw.scene.step()

        n_check = sum(1 for j in range(self.n) if j != idx and not removed[j])
        stable = sum(
            1 for j in range(self.n)
            if j != idx and not removed[j]
            and torch.norm(uw.blocks[j].pose.p[0] - init_pos[j]).item()
            < self.collapse_threshold
        )

        uw.scene.set_sim_state(state)
        return stable / max(n_check, 1)

    def _obs(self):
        """block_poses [N×3] ∥ mask [N] → flat float32"""
        uw = self.unwrapped
        poses = torch.stack([b.pose.p[0] for b in uw.blocks]).cpu().numpy()
        return np.concatenate([
            poses.flatten(), (~self._removed).astype(np.float32)
        ]).astype(np.float32)
