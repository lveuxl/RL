"""
PickSequence-v1 PPO 训练脚本
验证强化学习能否学会 5 个物体的合理抓取顺序

注意：PickSequenceEnv 是纯 Gymnasium 环境，不依赖 ManiSkill。
因此不需要 CPUGymWrapper / ManiSkillVectorEnv 等包装器。
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from collections import deque

import pick_sequence_env  # noqa: F401  触发 gym.register
import gymnasium as gym


# ─── 网络 ────────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.backbone(state)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, state, mask=None):
        logits, value = self.forward(state)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e8)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions, masks=None):
        logits, values = self.forward(states)
        if masks is not None:
            logits = logits.masked_fill(~masks, -1e8)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)


# ─── GAE ─────────────────────────────────────────────────────────


def compute_gae(rewards, values, next_val, dones, gamma=0.99, lam=0.95):
    n = len(rewards)
    adv = torch.zeros(n)
    gae = 0.0
    for t in reversed(range(n)):
        nv = next_val if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv


# ─── 动作掩码提取 ─────────────────────────────────────────────────


def get_action_mask(obs):
    """从观测中提取合法动作掩码：obs[30:35] 为 removed 标志，0=可选"""
    return obs[30:35] < 0.5


# ─── 训练 ────────────────────────────────────────────────────────


def train(args):
    env = gym.make("PickSequence-v1")
    state_dim = env.observation_space.shape[0]  # 40
    action_dim = env.action_space.n  # 5

    model = ActorCritic(state_dim, action_dim, args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ep_rewards_log = []
    ep_success_log = []
    recent_rewards = deque(maxlen=50)

    obs, _ = env.reset()
    ep_reward = 0.0

    for epoch in range(args.epochs):
        # ── 采集轨迹 ──
        buf_s, buf_a, buf_r, buf_d, buf_lp, buf_v, buf_m = (
            [], [], [], [], [], [], [],
        )

        for _ in range(args.steps_per_epoch):
            s_t = torch.FloatTensor(obs).unsqueeze(0)
            m_t = torch.BoolTensor(get_action_mask(obs)).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = model.get_action(s_t, m_t)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            buf_s.append(obs)
            buf_a.append(action.item())
            buf_r.append(reward)
            buf_d.append(float(done))
            buf_lp.append(log_prob.item())
            buf_v.append(value.item())
            buf_m.append(get_action_mask(obs))

            ep_reward += reward
            obs = next_obs

            if done:
                ep_rewards_log.append(ep_reward)
                ep_success_log.append(float(info.get("is_success", False)))
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        # ── PPO 更新 ──
        states_t = torch.FloatTensor(np.array(buf_s))
        actions_t = torch.LongTensor(buf_a)
        rewards_t = torch.FloatTensor(buf_r)
        dones_t = torch.FloatTensor(buf_d)
        old_lp_t = torch.FloatTensor(buf_lp)
        masks_t = torch.BoolTensor(np.array(buf_m))
        values_t = torch.FloatTensor(buf_v)

        with torch.no_grad():
            _, _, nv = model.get_action(torch.FloatTensor(obs).unsqueeze(0))

        advantages = compute_gae(
            rewards_t, values_t, nv.item(), dones_t, args.gamma, args.gae_lambda
        )
        returns = advantages + values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            new_lp, entropy, new_v = model.evaluate(states_t, actions_t, masks_t)
            ratio = (new_lp - old_lp_t).exp()
            s1 = ratio * advantages
            s2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantages
            loss = (
                -torch.min(s1, s2).mean()
                + 0.5 * F.mse_loss(new_v, returns)
                - args.ent_coef * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # ── 日志 ──
        if epoch % args.log_interval == 0 and recent_rewards:
            avg_r = np.mean(recent_rewards)
            sr = np.mean(ep_success_log[-50:]) if ep_success_log else 0
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"AvgReward {avg_r:+7.2f} | "
                f"SuccRate {sr:.0%} | "
                f"Episodes {len(ep_rewards_log)}"
            )

    env.close()

    # ── 保存模型 ──
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "pick_sequence_ppo.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存: {model_path}")

    return model, ep_rewards_log, ep_success_log


# ─── 可视化 ──────────────────────────────────────────────────────


def plot_curves(ep_rewards, ep_successes, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 奖励曲线
    ax1.plot(ep_rewards, alpha=0.25, color="steelblue", label="Episode Reward")
    w = min(50, max(len(ep_rewards) // 5, 1))
    if len(ep_rewards) >= w:
        ma = np.convolve(ep_rewards, np.ones(w) / w, mode="valid")
        ax1.plot(range(w - 1, len(ep_rewards)), ma, color="tomato", lw=2, label=f"MA-{w}")
    ax1.axhline(25, color="limegreen", ls="--", alpha=0.6, label="Optimal (+25)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("训练奖励收敛曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 成功率曲线
    if len(ep_successes) >= w:
        sr = np.convolve(ep_successes, np.ones(w) / w, mode="valid")
        ax2.plot(range(w - 1, len(ep_successes)), sr, color="seagreen", lw=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("成功率曲线 (滑动平均)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "reward_curve.png")
    plt.savefig(path, dpi=150)
    print(f"收敛曲线已保存: {path}")
    plt.close()


# ─── 评估 + Render 可视化 ────────────────────────────────────────


def evaluate(model, num_episodes=5):
    env = gym.make("PickSequence-v1", render_mode="human")
    model.eval()

    total_success = 0
    total_reward_sum = 0.0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        print(f"\n{'#'*55}")
        print(f"  评估 Episode {ep + 1}/{num_episodes}")
        print(f"{'#'*55}")
        env.unwrapped.render()

        ep_r = 0.0
        for step in range(5):
            s_t = torch.FloatTensor(obs).unsqueeze(0)
            m_t = torch.BoolTensor(get_action_mask(obs)).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(s_t)
                logits = logits.masked_fill(~m_t, -1e8)
                action = logits.argmax(-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward

            tag = "BLOCKED→惩罚" if reward <= -10 else "FREE→奖励"
            print(f"  Step {step+1}: 选择物体 {action}  [{tag}]  reward={reward:+.2f}")
            env.unwrapped.render()

            if terminated or truncated:
                break

        total_reward_sum += ep_r
        total_success += int(info.get("is_success", False))
        print(f"  Episode总奖励: {ep_r:+.2f}  成功: {info['is_success']}")

    env.close()
    print(f"\n{'='*55}")
    print(f"  评估汇总: {num_episodes} episodes")
    print(f"  平均奖励: {total_reward_sum / num_episodes:+.2f}")
    print(f"  成功率:   {total_success}/{num_episodes} = {total_success / num_episodes:.0%}")
    print(f"{'='*55}")


# ─── 入口 ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PickSequence-v1 PPO 训练")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./models/pick_sequence")
    parser.add_argument("--eval_only", type=str, default=None,
                        help="跳过训练，直接加载模型评估 (传入模型路径)")
    parser.add_argument("--eval_episodes", type=int, default=5)
    args = parser.parse_args()

    if args.eval_only:
        model = ActorCritic(40, 5, args.hidden_dim)
        model.load_state_dict(torch.load(args.eval_only, map_location="cpu"))
        print(f"已加载模型: {args.eval_only}")
        evaluate(model, args.eval_episodes)
        return

    model, ep_rewards, ep_successes = train(args)
    plot_curves(ep_rewards, ep_successes, args.save_dir)
    evaluate(model, args.eval_episodes)


if __name__ == "__main__":
    main()
