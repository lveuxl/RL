"""
Jenga PPO Update — 双网络梯度隔离联合训练

架构:
  vision_net (V-P3E):  pcd → Z, stab_hat, pot_hat, A_hat   [有梯度, 蒸馏更新]
                         ↓ .detach()                        [梯度截断]
  rl_net (ActorCritic): z_det, stab_det, pot_det → dist, V  [有梯度, PPO 更新]

两条独立梯度流:
  loss_rl      = L_clip + c_v·L_value − c_e·L_entropy  →  opt_rl   → rl_net
  loss_distill = w_s·L_stab + w_p·L_pot + w_g·L_graph  →  opt_vis  → vision_net

关键: .detach() 使两条 loss 的计算图完全不相交,
      不需要 retain_graph=True, 两次 backward 互不干扰。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vp3e_modules import VP3ENetwork, PriorGuidedActorCritic


def ppo_vision_update(
    vision_net: VP3ENetwork,
    rl_net: PriorGuidedActorCritic,
    opt_vision: torch.optim.Optimizer,
    opt_rl: torch.optim.Optimizer,
    # ── Buffer 数据 ──
    obs_pcd: torch.Tensor,        # [B, N, K, 3]
    mask: torch.Tensor,           # [B, N] bool
    actions: torch.Tensor,        # [B] long
    old_log_probs: torch.Tensor,  # [B]
    returns: torch.Tensor,        # [B]
    advantages: torch.Tensor,     # [B]
    old_values: torch.Tensor,     # [B]
    # ── GT 蒸馏标签 ──
    gt_stab: torch.Tensor,        # [B, N]
    gt_pot: torch.Tensor,         # [B, N]
    gt_graph: torch.Tensor,       # [B, N, N]
    # ── PPO 超参 ──
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    norm_adv: bool = True,
    clip_vloss: bool = False,
    # ── 蒸馏权重 ──
    w_stab: float = 1.0,
    w_pot: float = 1.0,
    w_graph: float = 1.0,
) -> dict:
    """
    单次 PPO minibatch 更新。

    Returns:
        dict: 所有标量指标 (pg_loss, v_loss, entropy, loss_rl,
              loss_distill, L_stab, L_pot, L_graph, approx_kl, clipfrac)
    """

    # ═══════════════════════════════════════════
    #  ① Vision-net 前向 (保留计算图, 供蒸馏)
    # ═══════════════════════════════════════════
    vp3e_out = vision_net(obs_pcd, mask, use_causal=True)

    # ═══════════════════════════════════════════
    #  ② Detach barrier — 梯度截断
    # ═══════════════════════════════════════════
    z_det    = vp3e_out["Z"].detach()
    stab_det = vp3e_out["stab_hat"].detach()
    pot_det  = vp3e_out["pot_hat"].detach()

    # ═══════════════════════════════════════════
    #  ③ RL-net 前向 (截断后的特征输入)
    # ═══════════════════════════════════════════
    _, new_log_prob, entropy, value = rl_net.get_action_and_value(
        z_det, stab_det, pot_det, mask, action=actions,
    )

    # ═══════════════════════════════════════════
    #  ④ PPO Loss
    # ═══════════════════════════════════════════
    logratio = new_log_prob - old_log_probs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8) if norm_adv else advantages

    # Clipped surrogate
    pg_loss = torch.max(
        -adv * ratio,
        -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
    ).mean()

    # Value loss
    val_flat = value.view(-1)
    if clip_vloss:
        v_unclip = (val_flat - returns) ** 2
        v_clip = old_values + torch.clamp(val_flat - old_values, -clip_coef, clip_coef)
        v_loss = 0.5 * torch.max(v_unclip, (v_clip - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((val_flat - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss_rl = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

    # ═══════════════════════════════════════════
    #  ⑤ Distillation Loss (未截断的 vision_net 输出)
    # ═══════════════════════════════════════════
    mask_f = mask.float()
    n_valid = mask_f.sum().clamp(min=1)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    n_pairs = pair_mask.float().sum().clamp(min=1)

    L_stab = ((vp3e_out["stab_hat"] - gt_stab) ** 2 * mask_f).sum() / n_valid
    L_pot  = ((vp3e_out["pot_hat"]  - gt_pot)  ** 2 * mask_f).sum() / n_valid

    # BCE: 强制 float32 + clamp 保证数值安全
    a_pred = (vp3e_out["A_hat"] * pair_mask.float()).clamp(1e-7, 1 - 1e-7)
    L_graph = F.binary_cross_entropy(
        a_pred.float(),
        (gt_graph * pair_mask.float()).float(),
        reduction="sum",
    ) / n_pairs

    loss_distill = w_stab * L_stab + w_pot * L_pot + w_graph * L_graph

    # ═══════════════════════════════════════════
    #  ⑥ 双路 Backward (计算图不相交, 无需 retain_graph)
    # ═══════════════════════════════════════════
    opt_rl.zero_grad()
    loss_rl.backward()
    gn_rl = nn.utils.clip_grad_norm_(rl_net.parameters(), max_grad_norm)
    opt_rl.step()

    opt_vision.zero_grad()
    loss_distill.backward()
    gn_vis = nn.utils.clip_grad_norm_(vision_net.parameters(), max_grad_norm)
    opt_vision.step()

    return {
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "loss_rl": loss_rl.item(),
        "loss_distill": loss_distill.item(),
        "L_stab": L_stab.item(),
        "L_pot": L_pot.item(),
        "L_graph": L_graph.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
        "grad_norm_rl": gn_rl.item(),
        "grad_norm_vis": gn_vis.item(),
    }
