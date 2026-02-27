"""
VP3E GPU 训练 — MPS/CUDA 加速 + 渐进式端到端 + LP 课程 + 训练曲线

CUDA: float16 AMP + GradScaler
MPS:  float32 (原生加速)
CPU:  float32 (fallback)

训练曲线: Matplotlib 实时 PNG + TensorBoard (若可用)

Usage:
    conda activate /opt/anaconda3/envs/skill
    python train_vp3e_gpu.py --h5 dataset_jenga_3000.h5 --epochs 100 --batch_size 16
    # 训练曲线保存在 checkpoints/curves.png, 每个 epoch 自动更新
    # 若已安装 tensorboard: tensorboard --logdir checkpoints/tb_logs
"""
import argparse
import math
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jenga_dataloader import JengaDataset, LPCurriculumSampler
from vp3e_modules import VP3ENetwork

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class TrainingLogger:
    """Matplotlib 实时绘图 + 可选 TensorBoard 双后端."""

    def __init__(self, save_dir: str, warmup_epochs: int):
        self.save_dir = save_dir
        self.warmup_epochs = warmup_epochs
        self.history = defaultdict(list)
        self.tb = None
        if HAS_TB:
            self.tb = SummaryWriter(os.path.join(save_dir, "tb_logs"))

    def log_epoch(self, epoch: int, avg_losses: dict, lr: float, weights: list):
        for k, v in avg_losses.items():
            self.history[k].append(v)
        self.history["lr"].append(lr)
        for i, w in enumerate(weights):
            self.history[f"w{i}"].append(w)

        if self.tb:
            for k, v in avg_losses.items():
                self.tb.add_scalar(f"loss/{k}", v, epoch)
            self.tb.add_scalar("lr", lr, epoch)
            for i, w in enumerate(weights):
                self.tb.add_scalar(f"curriculum/w{i}", w, epoch)
            self.tb.flush()

        self._plot(epoch)

    def _plot(self, epoch: int):
        epochs = list(range(1, epoch + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"VP3E Training — Epoch {epoch}", fontsize=14, fontweight="bold")

        # (0,0) Total loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history["total"], "k-", linewidth=2, label="total")
        if self.warmup_epochs < epoch:
            ax.axvline(self.warmup_epochs, color="gray", ls="--", alpha=0.6,
                       label=f"Phase 2 @ {self.warmup_epochs}")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (0,1) Component losses
        ax = axes[0, 1]
        for key, color, label in [
            ("amodal", "#2196F3", "L_amodal"),
            ("graph",  "#4CAF50", "L_graph"),
            ("stab",   "#FF9800", "L_stab"),
            ("pot",    "#F44336", "L_pot"),
        ]:
            ax.plot(epochs, self.history[key], color=color, linewidth=1.5, label=label)
        if self.warmup_epochs < epoch:
            ax.axvline(self.warmup_epochs, color="gray", ls="--", alpha=0.6)
        ax.set_ylabel("Loss")
        ax.set_title("Component Losses")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,0) Learning rate
        ax = axes[1, 0]
        ax.plot(epochs, self.history["lr"], "#9C27B0", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate (Cosine)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)

        # (1,1) Curriculum weights
        ax = axes[1, 1]
        labels_cmap = [("#4CAF50", "Easy"), ("#FF9800", "Medium"), ("#F44336", "Hard")]
        for i, (c, lbl) in enumerate(labels_cmap):
            key = f"w{i}"
            if key in self.history:
                ax.plot(epochs, self.history[key], color=c, linewidth=1.5, label=lbl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sampling Prob")
        ax.set_title("LP Curriculum Weights")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(self.save_dir, "curves.png"), dpi=150)
        plt.close(fig)

    def close(self):
        if self.tb:
            self.tb.close()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=str, default="dataset_jenga_3000.h5")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / mps / cpu (默认自动检测)")
    p.add_argument("--w_amodal", type=float, default=1.0)
    p.add_argument("--w_graph", type=float, default=1.0)
    p.add_argument("--w_stab", type=float, default=1.0)
    p.add_argument("--w_pot", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint 路径, 断点续训")
    return p.parse_args()


def compute_loss(pred, batch, args, use_causal: bool):
    mask = batch["mask"].float()
    n_valid = mask.sum().clamp(min=1)
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    n_pairs = pair_mask.sum().clamp(min=1)

    L_amodal = ((pred["bbox"] - batch["gt_bbox"]) ** 2 * mask.unsqueeze(-1)).sum() / n_valid

    a_pred = (pred["A_hat"] * pair_mask).clamp(1e-7, 1 - 1e-7)
    L_graph = F.binary_cross_entropy(
        a_pred, batch["support_matrix"] * pair_mask, reduction="sum",
    ) / n_pairs

    L_stab = ((pred["stab_hat"] - batch["gt_stability"]) ** 2 * mask).sum() / n_valid

    total = args.w_amodal * L_amodal + args.w_graph * L_graph + args.w_stab * L_stab

    if use_causal:
        L_pot = ((pred["pot_hat"] - batch["gt_potentiality"]) ** 2 * mask).sum() / n_valid
        total = total + args.w_pot * L_pot
    else:
        L_pot = torch.tensor(0.0, device=mask.device)

    return total, {
        "amodal": L_amodal.item(), "graph": L_graph.item(),
        "stab": L_stab.item(), "pot": L_pot.item(), "total": total.item(),
    }


def to_device(batch, device):
    """将 batch 中的 tensor 移至 device, bucket_idx 保留在 CPU."""
    out = {}
    for k, v in batch.items():
        if k == "bucket_idx":
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    is_cuda = device.type == "cuda"
    is_mps = device.type == "mps"

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- data ----------
    dataset = JengaDataset(args.h5, max_blocks=54)
    sampler = LPCurriculumSampler(dataset, tau=args.tau, eta=args.eta)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=is_cuda and args.num_workers > 0,
    )

    # ---------- model ----------
    model = VP3ENetwork(
        feat_dim=args.feat_dim, gnn_layers=args.gnn_layers, max_blocks=54,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP only on CUDA
    use_amp = is_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if is_cuda else None

    start_epoch = 1
    best_loss = float("inf")

    # ---------- resume ----------
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from {args.resume}  epoch={start_epoch}")

    # ---------- logger ----------
    logger = TrainingLogger(args.save_dir, args.warmup_epochs)

    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    print(f"VP3ENetwork: {n_params:,} params")
    print(f"Device: {device}  AMP: {use_amp}  TensorBoard: {HAS_TB}")
    print(f"Dataset: {len(dataset)} samples  batch={args.batch_size}  "
          f"steps/epoch={steps_per_epoch}")
    print(f"Loss weights: amodal={args.w_amodal} graph={args.w_graph} "
          f"stab={args.w_stab} pot={args.w_pot}")
    print(f"Progressive: warmup={args.warmup_epochs} → joint  "
          f"(start_epoch={start_epoch})\n")

    for epoch in range(start_epoch, args.epochs + 1):
        use_causal = epoch > args.warmup_epochs
        phase = "Phase 2: Joint" if use_causal else "Phase 1: Warmup"

        model.train()
        epoch_losses = {"amodal": 0., "graph": 0., "stab": 0., "pot": 0., "total": 0.}
        n_batches = 0
        t0 = time.time()

        for raw_batch in loader:
            bucket_ids = raw_batch["bucket_idx"]
            batch = to_device(raw_batch, device)

            # --- forward ---
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(batch["obs_point_clouds"], batch["mask"],
                                 use_causal=use_causal)
                    loss, loss_dict = compute_loss(pred, batch, args,
                                                   use_causal=use_causal)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch["obs_point_clouds"], batch["mask"],
                             use_causal=use_causal)
                loss, loss_dict = compute_loss(pred, batch, args,
                                               use_causal=use_causal)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # NaN guard
            if math.isnan(loss_dict["total"]):
                print(f"  [WARN] NaN loss at epoch {epoch}, skipping sampler update")
                continue

            for b_idx in bucket_ids.unique():
                sampler.update_loss(b_idx.item(), loss_dict["total"])

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1

        scheduler.step()
        sampler.snapshot()

        if n_batches == 0:
            print(f"[{phase}] E{epoch}: all batches NaN, skipping")
            continue

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        w = sampler.get_sampling_weights()
        dt = time.time() - t0

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(args.save_dir, f"ckpt_epoch{epoch:03d}.pt"))

        # 训练曲线
        cur_lr = scheduler.get_last_lr()[0]
        logger.log_epoch(epoch, avg, cur_lr, [w[i].item() for i in range(len(w))])

        should_print = (epoch <= 3 or epoch % 5 == 0
                        or epoch == args.epochs
                        or epoch == args.warmup_epochs + 1)
        if should_print:
            eta_min = dt * (args.epochs - epoch) / 60
            print(f"[{phase:15s}] E{epoch:3d}/{args.epochs}  "
                  f"loss={avg['total']:.4f}  "
                  f"am={avg['amodal']:.4f} gr={avg['graph']:.4f} "
                  f"st={avg['stab']:.4f} po={avg['pot']:.4f}  "
                  f"lr={cur_lr:.1e}  "
                  f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]  "
                  f"{dt:.1f}s  ETA≈{eta_min:.0f}min")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final.pt"))
    logger.close()
    print(f"\n训练完成 → {args.save_dir}/")
    print(f"  best.pt  (loss={best_loss:.4f})")
    print(f"  final.pt (epoch {args.epochs})")
    print(f"  curves.png (训练曲线)")


if __name__ == "__main__":
    main()
