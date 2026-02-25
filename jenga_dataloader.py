"""
JengaDataset + LPCurriculumSampler — 离线训练数据加载模块

Usage:
    from jenga_dataloader import JengaDataset, LPCurriculumSampler
    ds = JengaDataset("dataset_jenga_3000.h5", max_blocks=54)
    sampler = LPCurriculumSampler(ds)
    loader = DataLoader(ds, batch_size=32, sampler=sampler)
"""
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# ════════════════════════════════════════════════════
#  JengaDataset
# ════════════════════════════════════════════════════

class JengaDataset(Dataset):
    """
    从 HDF5 加载 Jenga 数据集。

    每个样本的 block 数 N 不同, 统一 pad 到 max_blocks,
    并返回 mask 标记有效节点。

    Returns dict:
        obs_point_clouds: (max_blocks, K, 3)   — xyz 坐标
        support_matrix:   (max_blocks, max_blocks) — 邻接矩阵
        gt_stability:     (max_blocks,)         — s_load * s_balance
        gt_potentiality:  (max_blocks,)         — 反事实稳定度
        complexity:       scalar                — 连续拓扑复杂度 c
        mask:             (max_blocks,)         — bool, True=有效节点
        bucket_idx:       int                   — 0=easy, 1=medium, 2=hard
    """

    BUCKET_BOUNDS = [0.33, 0.66]  # easy < 0.33, medium < 0.66, hard >= 0.66

    def __init__(self, h5_path: str, max_blocks: int = 54, pts_dim: int = 3):
        self.h5_path = h5_path
        self.max_blocks = max_blocks
        self.pts_dim = pts_dim

        with h5py.File(h5_path, "r") as f:
            self.sample_keys = sorted(f.keys())
            self.complexities = np.array(
                [f[k].attrs["complexity"] for k in self.sample_keys], dtype=np.float32
            )

        self.bucket_ids = np.digitize(self.complexities, self.BUCKET_BOUNDS).astype(int)
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._open()
        g = self._h5[self.sample_keys[idx]]
        M, K = self.max_blocks, g["obs_point_clouds"].shape[1]
        N = len(g["active_ids"])

        pcd = np.zeros((M, K, self.pts_dim), dtype=np.float32)
        pcd[:N] = g["obs_point_clouds"][:, :, :self.pts_dim]

        sup = np.zeros((M, M), dtype=np.float32)
        sup[:N, :N] = g["support_matrix"][:]

        stab = np.zeros(M, dtype=np.float32)
        stab[:N] = g["s_load"][:] * g["s_balance"][:]

        pot = np.zeros(M, dtype=np.float32)
        pot[:N] = g["potentiality"][:]

        mask = np.zeros(M, dtype=bool)
        mask[:N] = True

        gt_bbox = np.zeros((M, 4), dtype=np.float32)
        gt_bbox[:N, :3] = g["poses"][:]
        gt_bbox[:N, 3]  = g["volumes"][:]

        return {
            "obs_point_clouds": torch.from_numpy(pcd),
            "support_matrix":   torch.from_numpy(sup),
            "gt_stability":     torch.from_numpy(stab),
            "gt_potentiality":  torch.from_numpy(pot),
            "gt_bbox":          torch.from_numpy(gt_bbox),
            "complexity":       torch.tensor(self.complexities[idx]),
            "mask":             torch.from_numpy(mask),
            "bucket_idx":       int(self.bucket_ids[idx]),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

    def get_bucket_indices(self) -> List[List[int]]:
        """返回每个桶包含的样本索引列表。"""
        n_buckets = len(self.BUCKET_BOUNDS) + 1
        return [np.where(self.bucket_ids == b)[0].tolist() for b in range(n_buckets)]


# ════════════════════════════════════════════════════
#  LP Curriculum Sampler
# ════════════════════════════════════════════════════

class LPCurriculumSampler(Sampler):
    """
    基于 Learning Progress (LP) 的动态课程采样器。

    LP(c) = |ē_{t-Δt}(c) - ē_t(c)| / (ē_{t-Δt}(c) + ε)

    采样概率:
      P(c) ∝ exp(LP(c) / τ) + η · U(c)

    Args:
        dataset:     JengaDataset 实例
        epoch_size:  每个 epoch 采样数 (默认等于数据集大小)
        tau:         Boltzmann 温度
        eta:         均匀探索噪声系数
        ema_alpha:   EMA 平滑系数
        epsilon:     LP 分母中的数值稳定项
    """

    def __init__(
        self,
        dataset: JengaDataset,
        epoch_size: Optional[int] = None,
        tau: float = 0.1,
        eta: float = 0.05,
        ema_alpha: float = 0.1,
        epsilon: float = 1e-6,
    ):
        self.dataset = dataset
        self.epoch_size = epoch_size or len(dataset)
        self.tau = tau
        self.eta = eta
        self.alpha = ema_alpha
        self.eps = epsilon

        self.n_buckets = len(dataset.BUCKET_BOUNDS) + 1
        self.bucket_indices = dataset.get_bucket_indices()

        # EMA loss: 当前 ē_t 与前一快照 ē_{t-Δt}
        self.ema_loss = torch.ones(self.n_buckets)
        self.ema_loss_prev = torch.ones(self.n_buckets)

        self._weights = torch.ones(self.n_buckets) / self.n_buckets

    def update_loss(self, bucket_idx: int, current_loss: float):
        """用一个 batch 的 loss 更新对应桶的 EMA。"""
        self.ema_loss[bucket_idx] = (
            (1 - self.alpha) * self.ema_loss[bucket_idx]
            + self.alpha * current_loss
        )

    def snapshot(self):
        """在每个 epoch 结束时调用, 保存当前 EMA 为 prev, 并重新计算采样权重。"""
        self.ema_loss_prev = self.ema_loss.clone()
        self._weights = self.get_sampling_weights()

    def get_sampling_weights(self) -> torch.Tensor:
        """返回三个桶的采样概率向量 (归一化)。"""
        lp = torch.abs(self.ema_loss_prev - self.ema_loss) / (self.ema_loss_prev + self.eps)
        boltzmann = torch.exp(lp / self.tau)
        uniform = torch.ones(self.n_buckets)
        raw = boltzmann + self.eta * uniform
        return raw / raw.sum()

    def __iter__(self):
        weights = self._weights
        indices = []
        for _ in range(self.epoch_size):
            bucket = torch.multinomial(weights, 1).item()
            pool = self.bucket_indices[bucket]
            idx = pool[torch.randint(len(pool), (1,)).item()]
            indices.append(idx)
        return iter(indices)

    def __len__(self):
        return self.epoch_size
