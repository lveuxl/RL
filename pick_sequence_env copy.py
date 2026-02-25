"""
PickSequence-v1: 高层 RL 环境 —— 抓取顺序规划 (Pick Sequence Planning)

5 个立方体按列堆叠，agent 逐步选择移除哪个物体。
目标：学习从上到下的合理移除顺序，最小化剩余物体扰动。

观测 (40 维)：positions 5×3 + sizes 5×3 + removed 5×1 + blocked 5×1
动作：Discrete(5)
奖励：被遮挡 → −10 | 最上层 → +5 | 位移惩罚 → −α·Σdisp | 已移除 → −5
"""

import platform
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class PickSequenceEnv(gym.Env):
    """
    高层任务级 RL 环境：学习最优抓取顺序

    场景：5 个立方体随机分布在 2-3 个堆叠列中
    Agent 每步选择一个物体 index (0-4) 进行移除
    奖励信号引导 agent 学会"从上到下"的移除策略
    """

    metadata = {"render_modes": ["human"]}

    NUM_OBJECTS = 5
    MAX_STEPS = 5
    ALPHA = 1.0  # 位移惩罚系数

    def __init__(self, render_mode=None):
        super().__init__()
        self.device = (
            torch.device("cpu")
            if platform.system() == "Darwin"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.NUM_OBJECTS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
        )

        self.positions = None   # [5, 3] 物体中心坐标
        self.sizes = None       # [5, 3] 物体半尺寸 (half-length, half-width, half-height)
        self.removed = None     # [5]    是否已被移除
        self.step_count = 0

    # ─── Gymnasium 核心接口 ──────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.removed = torch.zeros(self.NUM_OBJECTS, dtype=torch.bool, device=self.device)
        self._generate_scene()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        self.step_count += 1

        if self.removed[action]:
            reward = -5.0
        else:
            reward = -10.0 if self._is_blocked(action) else 5.0
            old_pos = self.positions.clone()
            self.removed[action] = True
            self._simulate_gravity()
            reward -= self.ALPHA * self._compute_displacement(old_pos, action)
            self._simulate_pick(action)

        terminated = self.removed.all().item()
        truncated = self.step_count >= self.MAX_STEPS and not terminated
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode != "human":
            return
        labels = {True: "REMOVED", False: None}
        print(f"\n{'='*55}")
        print(f"  Step {self.step_count}/{self.MAX_STEPS}  "
              f"Removed: {int(self.removed.sum())}/{self.NUM_OBJECTS}")
        print(f"{'-'*55}")
        for i in range(self.NUM_OBJECTS):
            if self.removed[i]:
                tag = "REMOVED"
            elif self._is_blocked(i):
                tag = "BLOCKED"
            else:
                tag = "FREE   "
            p = self.positions[i].cpu().numpy()
            s = self.sizes[i].cpu().numpy()
            print(f"  Obj {i} [{tag}]  "
                  f"pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  "
                  f"size=({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})")
        print(f"{'='*55}")

    # ─── 场景生成 ────────────────────────────────────────────────

    def _generate_scene(self):
        """随机生成 5 个立方体的堆叠场景 (2-3 列)"""
        self.sizes = torch.tensor(
            [[self.np_random.uniform(0.03, 0.06),
              self.np_random.uniform(0.03, 0.06),
              self.np_random.uniform(0.02, 0.05)]
             for _ in range(self.NUM_OBJECTS)],
            device=self.device, dtype=torch.float32,
        )

        num_cols = int(self.np_random.integers(2, 4))
        col_xs = np.linspace(-0.12, 0.12, num_cols)
        col_ys = self.np_random.uniform(-0.05, 0.05, size=num_cols)

        # 分配物体到列（保证每列至少一个物体）
        col_assign = self.np_random.integers(0, num_cols, size=self.NUM_OBJECTS)
        for c in range(num_cols):
            if not np.any(col_assign == c):
                col_assign[self.np_random.integers(0, self.NUM_OBJECTS)] = c

        self.positions = torch.zeros(self.NUM_OBJECTS, 3, device=self.device)
        col_h = np.zeros(num_cols)
        for i in range(self.NUM_OBJECTS):
            c = col_assign[i]
            h = self.sizes[i, 2].item()
            self.positions[i] = torch.tensor(
                [col_xs[c], col_ys[c], col_h[c] + h], device=self.device
            )
            col_h[c] += 2 * h

    # ─── 观测构建 ────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        blocked = torch.tensor(
            [float(self._is_blocked(i)) for i in range(self.NUM_OBJECTS)],
            device=self.device,
        )
        return torch.cat([
            self.positions.flatten(),   # 15
            self.sizes.flatten(),       # 15
            self.removed.float(),       # 5
            blocked,                    # 5
        ]).cpu().numpy().astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "num_removed": int(self.removed.sum().item()),
            "is_success": self.removed.all().item(),
        }

    # ─── 物理/逻辑辅助 ──────────────────────────────────────────

    def _is_blocked(self, obj_id: int) -> bool:
        """判断 obj_id 上方是否存在未移除且 xy 投影重叠的物体"""
        if self.removed[obj_id]:
            return False
        oz = self.positions[obj_id, 2]
        for j in range(self.NUM_OBJECTS):
            if j == obj_id or self.removed[j]:
                continue
            if self.positions[j, 2] > oz and self._xy_overlap(j, obj_id):
                return True
        return False

    def _xy_overlap(self, a: int, b: int) -> bool:
        """判断两个物体在 xy 平面的包围盒是否重叠"""
        dx = (self.positions[a, 0] - self.positions[b, 0]).abs().item()
        dy = (self.positions[a, 1] - self.positions[b, 1]).abs().item()
        return (dx < (self.sizes[a, 0] + self.sizes[b, 0]).item()
                and dy < (self.sizes[a, 1] + self.sizes[b, 1]).item())

    def _simulate_gravity(self):
        """移除物体后，上方物体在重力作用下沉降到最近支撑面"""
        for _ in range(self.NUM_OBJECTS):
            settled = True
            for i in range(self.NUM_OBJECTS):
                if self.removed[i]:
                    continue
                support_z = 0.0
                for j in range(self.NUM_OBJECTS):
                    if j == i or self.removed[j]:
                        continue
                    if self.positions[j, 2] < self.positions[i, 2] and self._xy_overlap(i, j):
                        support_z = max(support_z, (self.positions[j, 2] + self.sizes[j, 2]).item())
                new_z = support_z + self.sizes[i, 2].item()
                if abs(new_z - self.positions[i, 2].item()) > 1e-6:
                    self.positions[i, 2] = new_z
                    settled = False
            if settled:
                break

    def _compute_displacement(self, old_pos: torch.Tensor, removed_id: int) -> float:
        """计算移除后其余物体的总 L2 位移"""
        total = 0.0
        for i in range(self.NUM_OBJECTS):
            if i == removed_id or self.removed[i]:
                continue
            total += torch.linalg.norm(self.positions[i] - old_pos[i]).item()
        return total

    # ─── 未来接口（预留） ────────────────────────────────────────

    def _simulate_pick(self, object_id: int):
        """Placeholder：未来对接 motion planner + 连续控制器"""
        pass


# ─── 环境注册 ────────────────────────────────────────────────────
gym.register(id="PickSequence-v1", entry_point="pick_sequence_env:PickSequenceEnv")


# ─── 快速验证 ────────────────────────────────────────────────────
if __name__ == "__main__":
    env = PickSequenceEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"观测维度: {obs.shape}, 设备: {env.device}")
    env.render()

    total_reward = 0.0
    for t in range(env.MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"\nStep {t+1}: action={action}, reward={reward:+.2f}")
        env.render()
        if terminated or truncated:
            break

    print(f"\n总奖励: {total_reward:+.2f}, 成功: {info['is_success']}")
