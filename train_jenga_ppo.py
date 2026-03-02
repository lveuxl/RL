"""
Jenga PPO 主训练循环 — 在线课程学习 + 双网络梯度隔离 + 蒸馏系数衰减

全局调度:
  ┌─ 课程控制: target_c 从 0.2 起, success_rate > 0.85 时 +0.05
  ├─ 蒸馏衰减: λ_t = λ_0 · (1 − t/T), 先验引导逐渐退火
  ├─ Rollout:  T 步交互 → Buffer (obs_pcd, action, reward, GT labels)
  ├─ GT 回填:  episode 结束 → compute_distill_gt() → 填充 gt_pot [N]
  ├─ GAE:      per-episode 计算 advantage & returns
  └─ Update:   ppo_vision_update() × K epochs × minibatches

Usage:
    conda activate /opt/anaconda3/envs/skill
    python train_jenga_ppo.py --vp3e_ckpt checkpoints/best.pt --total_steps 200000
"""
import argparse
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from jenga_tower import NUM_BLOCKS, render_point_cloud
from jenga_ppo_wrapper import JengaPPOWrapper
from vp3e_modules import VP3ENetwork, PriorGuidedActorCritic
from ppo_jenga_update import ppo_vision_update


# ════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════

def render_vp3e_input(env, n_pts=256):
    """多视角点云 → V-P3E 输入 [1, N, K, 3]"""
    uw = env.unwrapped
    cams = {k: v for k, v in uw.scene.sensors.items() if k.startswith("surround")}
    pcd_data = render_point_cloud(uw.scene, cams, uw.blocks)

    N = len(uw.blocks)
    obs = np.zeros((1, N, n_pts, 3), dtype=np.float32)
    for i, pc in enumerate(pcd_data["per_block_pcd"]):
        if len(pc) == 0:
            continue
        xyz = pc[:, :3]
        if len(xyz) >= n_pts:
            obs[0, i] = xyz[np.random.choice(len(xyz), n_pts, replace=False)]
        else:
            obs[0, i, :len(xyz)] = xyz
    return torch.from_numpy(obs)


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95,
                next_value=0.0):
    """单 episode GAE → (advantages, returns)，均为 [T] tensor."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        non_term = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * non_term - values[t]
        last_gae = delta + gamma * gae_lambda * non_term * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values



# ════════════════════════════════════════════════════
#  主训练循环
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    # 模型
    p.add_argument("--vp3e_ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--n_pts", type=int, default=256)
    # 训练
    p.add_argument("--total_steps", type=int, default=200_000)
    p.add_argument("--rollout_steps", type=int, default=64,
                   help="每次 update 前收集的最少步数")
    p.add_argument("--max_ep_steps", type=int, default=20)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=16)
    p.add_argument("--lr_rl", type=float, default=3e-4)
    p.add_argument("--lr_vision", type=float, default=1e-4)
    # PPO
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--target_kl", type=float, default=0.05)
    # 课程
    p.add_argument("--init_complexity", type=float, default=0.2)
    p.add_argument("--complexity_step", type=float, default=0.05)
    p.add_argument("--success_threshold", type=float, default=0.85)
    p.add_argument("--ema_decay", type=float, default=0.9)
    # 蒸馏
    p.add_argument("--lambda_init", type=float, default=0.1)
    p.add_argument("--w_stab", type=float, default=1.0)
    p.add_argument("--w_pot", type=float, default=1.0)
    p.add_argument("--w_graph", type=float, default=1.0)
    # 环境
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="runs/jenga_ppo")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 环境 ──
    base_env = gym.make(
        "JengaTower-v1", obs_mode="state", render_mode="rgb_array",
        num_envs=1, sim_backend="cpu",
    )
    env = JengaPPOWrapper(base_env, lambda_int=args.lambda_init)

    # ── 网络 ──
    vision_net = VP3ENetwork(
        feat_dim=args.feat_dim, gnn_layers=args.gnn_layers, max_blocks=NUM_BLOCKS,
    ).to(device)
    if os.path.isfile(args.vp3e_ckpt):
        vision_net.load_state_dict(torch.load(args.vp3e_ckpt, map_location=device))
        print(f"V-P3E loaded from {args.vp3e_ckpt}")

    rl_net = PriorGuidedActorCritic(feat_dim=args.feat_dim).to(device)

    # ── 优化器 ──
    opt_vision = torch.optim.AdamW(vision_net.parameters(), lr=args.lr_vision, weight_decay=1e-4)
    opt_rl = torch.optim.Adam(rl_net.parameters(), lr=args.lr_rl, eps=1e-5)

    # ── 课程状态 ──
    target_c = args.init_complexity
    success_ema = 0.0
    success_history = deque(maxlen=100)

    # ── 日志 ──
    n_params_vis = sum(p.numel() for p in vision_net.parameters())
    n_params_rl = sum(p.numel() for p in rl_net.parameters())
    print(f"Device: {device}")
    print(f"V-P3E: {n_params_vis:,} params  |  ActorCritic: {n_params_rl:,} params")
    print(f"Curriculum: c={target_c:.2f}  |  λ_init={args.lambda_init}")
    print(f"Rollout: {args.rollout_steps} steps  |  Update: {args.update_epochs} epochs\n")

    global_step = 0
    iteration = 0
    t_start = time.time()

    # ════════════════════════════════════════════
    #  主循环
    # ════════════════════════════════════════════
    while global_step < args.total_steps:
        iteration += 1

        # ── ① 蒸馏系数衰减 ──
        progress = min(global_step / args.total_steps, 1.0)
        lambda_t = args.lambda_init * (1.0 - progress)
        env.lambda_int = lambda_t

        # ── ② Rollout: 收集 ≥ rollout_steps 步 ──
        vision_net.eval()
        rl_net.eval()

        all_pcd, all_mask = [], []
        all_act, all_lp, all_val = [], [], []
        all_rew, all_done = [], []
        all_gt_stab, all_gt_pot, all_gt_graph = [], [], []
        all_adv, all_ret = [], []

        steps_collected = 0
        ep_in_rollout = 0

        while steps_collected < args.rollout_steps:
            # Episode reset (带课程难度)
            obs, info = env.reset(target_c=target_c)
            mask_np = info["mask"]
            ep_rewards = []
            ep_collapsed = False

            ep_pcd, ep_mask = [], []
            ep_act, ep_lp, ep_val = [], [], []
            ep_rew, ep_done = [], []
            ep_gt_stab, ep_gt_graph = [], []

            for step in range(args.max_ep_steps):
                pcd = render_vp3e_input(env, args.n_pts).to(device)
                mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)

                with torch.no_grad():
                    vp3e_out = vision_net(pcd, mask_t, use_causal=True)
                    action, log_prob, _, value = rl_net.get_action_and_value(
                        vp3e_out["Z"], vp3e_out["stab_hat"],
                        vp3e_out["pot_hat"], mask_t,
                    )

                env.set_prior_scores(vp3e_out["pot_hat"][0].cpu().numpy())
                obs_new, reward, done, truncated, info = env.step(action.item())

                ep_pcd.append(pcd.cpu())
                ep_mask.append(mask_t.cpu())
                ep_act.append(action.cpu())
                ep_lp.append(log_prob.cpu())
                ep_val.append(value.view(-1).cpu())
                ep_rew.append(reward)
                ep_done.append(float(done))
                ep_gt_stab.append(torch.from_numpy(info["gt_stability"]).unsqueeze(0))
                ep_gt_graph.append(torch.from_numpy(info["gt_support_matrix"]).unsqueeze(0))

                mask_np = info["mask"]

                if done:
                    ep_collapsed = info.get("collapsed", False)
                    break

            ep_len = len(ep_rew)

            # Episode 结束 → gt_potentiality 批量回溯
            distill_results = env.compute_distill_gt()
            ep_gt_pot = []
            for i in range(ep_len):
                if i < len(distill_results):
                    ep_gt_pot.append(
                        torch.from_numpy(distill_results[i]["gt_potentiality"]).unsqueeze(0)
                    )
                else:
                    ep_gt_pot.append(torch.zeros(1, NUM_BLOCKS))

            # GAE (per-episode)
            rew_t = torch.tensor(ep_rew, dtype=torch.float32)
            val_t = torch.cat(ep_val)
            done_t = torch.tensor(ep_done, dtype=torch.float32)
            adv, ret = compute_gae(
                rew_t, val_t, done_t,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
            )

            # 汇入 Rollout Buffer
            all_pcd.extend(ep_pcd)
            all_mask.extend(ep_mask)
            all_act.extend(ep_act)
            all_lp.extend(ep_lp)
            all_val.extend(ep_val)
            all_rew.extend([torch.tensor([r]) for r in ep_rew])
            all_done.extend([torch.tensor([d]) for d in ep_done])
            all_gt_stab.extend(ep_gt_stab)
            all_gt_pot.extend(ep_gt_pot)
            all_gt_graph.extend(ep_gt_graph)
            all_adv.append(adv)
            all_ret.append(ret)

            steps_collected += ep_len
            ep_in_rollout += 1

            # 课程: 记录成功率
            success = not ep_collapsed
            success_history.append(float(success))
            success_ema = args.ema_decay * success_ema + (1 - args.ema_decay) * float(success)

        global_step += steps_collected

        # ── ③ Stack → Tensor ──
        B = steps_collected
        batch_pcd    = torch.cat(all_pcd, dim=0).to(device)       # [B,N,K,3]
        batch_mask   = torch.cat(all_mask, dim=0).to(device)      # [B,N]
        batch_act    = torch.cat(all_act).to(device)              # [B]
        batch_lp     = torch.cat(all_lp).to(device)               # [B]
        batch_val    = torch.cat(all_val).to(device)              # [B]
        batch_adv    = torch.cat(all_adv).to(device)              # [B]
        batch_ret    = torch.cat(all_ret).to(device)              # [B]
        batch_stab   = torch.cat(all_gt_stab, dim=0).to(device)  # [B,N]
        batch_pot    = torch.cat(all_gt_pot, dim=0).to(device)    # [B,N]
        batch_graph  = torch.cat(all_gt_graph, dim=0).to(device)  # [B,N,N]

        # ── ④ PPO + 蒸馏 联合更新 ──
        vision_net.train()
        rl_net.train()

        update_metrics = {}
        early_stop = False

        for epoch in range(args.update_epochs):
            if early_stop:
                break

            perm = torch.randperm(B, device=device)
            for mb_start in range(0, B, args.minibatch_size):
                mb = perm[mb_start: mb_start + args.minibatch_size]
                if len(mb) < 2:
                    continue

                metrics = ppo_vision_update(
                    vision_net, rl_net, opt_vision, opt_rl,
                    obs_pcd=batch_pcd[mb],
                    mask=batch_mask[mb],
                    actions=batch_act[mb],
                    old_log_probs=batch_lp[mb],
                    returns=batch_ret[mb],
                    advantages=batch_adv[mb],
                    old_values=batch_val[mb],
                    gt_stab=batch_stab[mb],
                    gt_pot=batch_pot[mb],
                    gt_graph=batch_graph[mb],
                    clip_coef=args.clip_coef,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm,
                    w_stab=args.w_stab,
                    w_pot=args.w_pot,
                    w_graph=args.w_graph,
                )
                update_metrics = metrics

                if metrics["approx_kl"] > args.target_kl:
                    early_stop = True
                    break

        # ── ⑤ 课程更新 ──
        if success_ema > args.success_threshold and len(success_history) >= 10:
            old_c = target_c
            target_c = min(1.0, target_c + args.complexity_step)
            if target_c > old_c:
                print(f"  ★ 课程升级: c {old_c:.2f} → {target_c:.2f}  "
                      f"(success_ema={success_ema:.3f})")

        # ── ⑥ 保存 ──
        if iteration % args.save_every == 0:
            torch.save({
                "vision_net": vision_net.state_dict(),
                "rl_net": rl_net.state_dict(),
                "opt_vision": opt_vision.state_dict(),
                "opt_rl": opt_rl.state_dict(),
                "target_c": target_c,
                "success_ema": success_ema,
                "global_step": global_step,
                "iteration": iteration,
            }, os.path.join(args.save_dir, f"ckpt_{iteration:04d}.pt"))

        # ── ⑦ 日志 ──
        elapsed = time.time() - t_start
        sps = global_step / max(elapsed, 1)
        avg_success = np.mean(list(success_history)) if success_history else 0
        avg_rew = sum(r.item() for r in all_rew) / max(len(all_rew), 1)

        if iteration <= 3 or iteration % 5 == 0:
            print(
                f"[Iter {iteration:4d}]  steps={global_step:7d}/{args.total_steps}  "
                f"c={target_c:.2f}  λ={lambda_t:.4f}  "
                f"suc={avg_success:.2f}({success_ema:.3f})  "
                f"rew={avg_rew:+.2f}  "
                f"rl={update_metrics.get('loss_rl', 0):.4f}  "
                f"dist={update_metrics.get('loss_distill', 0):.4f}  "
                f"kl={update_metrics.get('approx_kl', 0):.4f}  "
                f"ep/roll={ep_in_rollout}  "
                f"SPS={sps:.0f}  {elapsed:.0f}s"
            )

    # ── 保存最终模型 ──
    torch.save(vision_net.state_dict(), os.path.join(args.save_dir, "vision_final.pt"))
    torch.save(rl_net.state_dict(), os.path.join(args.save_dir, "rl_final.pt"))
    print(f"\n训练完成 → {args.save_dir}/")
    print(f"  vision_final.pt + rl_final.pt")
    print(f"  总步数: {global_step}  |  最终 c={target_c:.2f}  success_ema={success_ema:.3f}")

    base_env.close()


if __name__ == "__main__":
    main()
