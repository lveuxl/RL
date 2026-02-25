"""
VP3E 离线训练 — 特权蒸馏 + LP 课程采样

Usage:
    conda activate /opt/anaconda3/envs/skill
    python train_vp3e.py --h5 dataset_jenga_3000.h5 --epochs 100 --batch_size 8
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jenga_dataloader import JengaDataset, LPCurriculumSampler
from vp3e_modules import VP3ENetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=str, default="dataset_jenga_3000.h5")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Loss 权重
    p.add_argument("--w_amodal", type=float, default=1.0)
    p.add_argument("--w_graph", type=float, default=1.0)
    p.add_argument("--w_stab", type=float, default=1.0)
    p.add_argument("--w_pot", type=float, default=1.0)
    # 课程采样
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=0.05)
    return p.parse_args()


def compute_loss(pred, batch, args):
    """
    特权蒸馏损失:
      L = w1·L_amodal + w2·L_graph + w3·L_stab + w4·L_pot

    所有 loss 均只在 mask=True 的有效节点上计算。
    """
    mask = batch["mask"].float()            # [B, N]
    n_valid = mask.sum().clamp(min=1)
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, N, N]
    n_pairs = pair_mask.sum().clamp(min=1)

    # L_amodal: MSE(bbox_pred, [gt_center, gt_volume])
    L_amodal = ((pred["bbox"] - batch["gt_bbox"]) ** 2 * mask.unsqueeze(-1)).sum() / n_valid

    # L_graph: BCE(A_hat, GT_support_matrix)
    L_graph = F.binary_cross_entropy(
        pred["A_hat"] * pair_mask, batch["support_matrix"] * pair_mask,
        reduction="sum",
    ) / n_pairs

    # L_stability: MSE
    L_stab = ((pred["stab_hat"] - batch["gt_stability"]) ** 2 * mask).sum() / n_valid

    # L_potential: MSE
    L_pot = ((pred["pot_hat"] - batch["gt_potentiality"]) ** 2 * mask).sum() / n_valid

    total = (args.w_amodal * L_amodal + args.w_graph * L_graph
             + args.w_stab * L_stab + args.w_pot * L_pot)

    return total, {
        "amodal": L_amodal.item(), "graph": L_graph.item(),
        "stab": L_stab.item(), "pot": L_pot.item(), "total": total.item(),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = JengaDataset(args.h5, max_blocks=54)
    sampler = LPCurriculumSampler(dataset, tau=args.tau, eta=args.eta)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    model = VP3ENetwork(
        feat_dim=args.feat_dim, gnn_layers=args.gnn_layers, max_blocks=54,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VP3ENetwork: {n_params:,} params  device={device}")
    print(f"Dataset: {len(dataset)} samples  batch={args.batch_size}")
    print(f"Loss weights: amodal={args.w_amodal} graph={args.w_graph} "
          f"stab={args.w_stab} pot={args.w_pot}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"amodal": 0, "graph": 0, "stab": 0, "pot": 0, "total": 0}
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            pred = model(batch["obs_point_clouds"], batch["mask"])
            loss, loss_dict = compute_loss(pred, batch, args)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 更新 LP Sampler
            bucket_ids = batch["bucket_idx"]
            for b_idx in bucket_ids.unique():
                sampler.update_loss(b_idx.item(), loss_dict["total"])

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1

        scheduler.step()
        sampler.snapshot()

        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        w = sampler.get_sampling_weights()
        dt = time.time() - t0

        if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:3d}/{args.epochs}]  "
                  f"loss={avg['total']:.4f}  "
                  f"amodal={avg['amodal']:.4f}  graph={avg['graph']:.4f}  "
                  f"stab={avg['stab']:.4f}  pot={avg['pot']:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]  "
                  f"{dt:.1f}s")

    torch.save(model.state_dict(), "vp3e_model.pt")
    print(f"\n模型已保存到 vp3e_model.pt")


if __name__ == "__main__":
    main()
