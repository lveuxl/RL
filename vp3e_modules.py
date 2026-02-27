"""
V-P3E 模块

    InstancePerception   — 点云 → 节点特征 + Amodal BBox
    ImplicitTopology     — 节点特征 → 支撑概率矩阵
    PhysicalMessagePassing — 物理图卷积 (载荷传播)
    StabilityHead        — 即刻稳定性预测
    CausalSTIT           — 反事实 Transformer → Potentiality
    VP3ENetwork          — 端到端封装
"""
import torch
import torch.nn as nn


class InstancePerception(nn.Module):
    """
    局部点云 → 节点特征 F^0 + Amodal BBox 辅助头。

    PointNet: 逐点 MLP + max-pool → per-instance feature。
    Amodal head: 预测 3D BBox 中心 (x,y,z) 和体积 v, 用于对抗遮挡。

    Input:  pcd  [B, N, K, 3]   — N 个木块, 每块 K 个点 (xyz)
            mask [B, N]          — bool, True=有效节点
    Output: F0   [B, N, D]      — 节点特征
            bbox [B, N, 4]      — (cx, cy, cz, volume)
    """

    def __init__(self, feat_dim: int = 128, K: int = 256):
        super().__init__()
        self.feat_dim = feat_dim

        self.pointnet = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

        self.amodal_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, pcd: torch.Tensor, mask: torch.Tensor):
        B, N, K, _ = pcd.shape

        # [B, N, K, 3] → [B*N, K, 3]
        x = pcd.reshape(B * N, K, 3)

        # 逐点 MLP: [B*N, K, 3] → [B*N, K, D]
        x = self.pointnet(x)

        # max-pool over K: [B*N, D]
        x = x.max(dim=1).values

        # [B, N, D]
        F0 = x.reshape(B, N, self.feat_dim)

        # mask 掉 padding 节点
        F0 = F0 * mask.unsqueeze(-1).float()

        # Amodal 辅助头: [B, N, 4]
        bbox = self.amodal_head(F0)

        return F0, bbox


class ImplicitTopology(nn.Module):
    """
    节点特征 → 预测支撑概率矩阵 Â。

    对任意对 (i,j): 拼接 [f_i, f_j, f_i-f_j] 送入 MLP → sigmoid。

    Input:  F0   [B, N, D]  — 节点特征
            mask [B, N]      — bool, True=有效节点
    Output: A_hat [B, N, N]  — 支撑概率 (经 Sigmoid)
    """

    def __init__(self, feat_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, F0: torch.Tensor, mask: torch.Tensor):
        B, N, D = F0.shape

        # [B, N, 1, D] × [B, 1, N, D] → 广播得到所有 (i,j) 对
        fi = F0.unsqueeze(2).expand(B, N, N, D)
        fj = F0.unsqueeze(1).expand(B, N, N, D)

        # 拼接: [f_i, f_j, f_i - f_j]  → [B, N, N, 3D]
        pair = torch.cat([fi, fj, fi - fj], dim=-1)

        # MLP → logit → sigmoid: [B, N, N]
        logit = self.edge_mlp(pair).squeeze(-1)

        # mask: 只有 (有效i, 有效j) 对才有意义
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        fill_value = -1e9 if logit.dtype == torch.float32 else -65000
        logit = logit.masked_fill(~pair_mask, fill_value)

        return torch.sigmoid(logit)


# ════════════════════════════════════════════════════
#  Physical GNN Layer (single layer)
# ════════════════════════════════════════════════════

class _PhysGNNLayer(nn.Module):
    """
    单层物理图卷积。

    载荷累加语义: A_hat[i,j] = "i 支撑 j" → 信息从 j 流向 i。
    更新: f_i^{l+1} = ReLU( W·f_i^l + Σ_j A_hat[i,j]·MLP_msg(f_j^l) )

    这里 A_hat 的第 i 行汇聚了节点 i 所支撑的所有节点 j 的消息,
    正好模拟 "被支撑者的载荷向下传递给支撑者" 的物理过程。
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(inplace=True),
            nn.Linear(d_out, d_out),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, F: torch.Tensor, A: torch.Tensor, mask: torch.Tensor):
        """
        F: [B, N, d_in], A: [B, N, N], mask: [B, N]
        """
        # 自更新
        h_self = self.W_self(F)                   # [B, N, d_out]

        # 消息: MLP_msg(f_j) for all j
        msg = self.msg_mlp(F)                     # [B, N, d_out]

        # 聚合: Σ_j A[i,j] · msg_j  →  A @ msg
        # A[i,j] 大 = i 支撑 j → j 的载荷信息流入 i
        h_agg = torch.bmm(A, msg)                 # [B, N, d_out]

        out = torch.relu(self.norm(h_self + h_agg))
        return out * mask.unsqueeze(-1).float()


# ════════════════════════════════════════════════════
#  PhysicalMessagePassing (multi-layer)
# ════════════════════════════════════════════════════

class PhysicalMessagePassing(nn.Module):
    """
    堆叠 L 层物理图卷积, 输出物理融合特征 Z。

    Input:  F0    [B, N, D]    — 初始节点特征
            A_hat [B, N, N]    — 预测支撑概率矩阵
            mask  [B, N]       — 有效节点
    Output: Z     [B, N, D]    — 物理融合后的节点特征
    """

    def __init__(self, feat_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [_PhysGNNLayer(feat_dim, feat_dim) for _ in range(n_layers)]
        )

    def forward(self, F0: torch.Tensor, A_hat: torch.Tensor, mask: torch.Tensor):
        Z = F0
        for layer in self.layers:
            Z = Z + layer(Z, A_hat, mask)   # 残差连接
        return Z


# ════════════════════════════════════════════════════
#  Stability Head
# ════════════════════════════════════════════════════

class StabilityHead(nn.Module):
    """
    物理融合特征 → 即刻稳定性预测。

    Input:  Z    [B, N, D]  — 物理融合特征
            mask [B, N]      — 有效节点
    Output: p    [B, N]      — 稳定性概率 ∈(0,1)
    """

    def __init__(self, feat_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z: torch.Tensor, mask: torch.Tensor):
        logit = self.mlp(Z).squeeze(-1)                    # [B, N]
        return torch.sigmoid(logit) * mask.float()


# ════════════════════════════════════════════════════
#  CausalSTIT (Counterfactual Transformer)
# ════════════════════════════════════════════════════

class CausalSTIT(nn.Module):
    """
    反事实结构干预 Transformer。

    对每个节点 i: 替换为 [MASK] token + attention 屏蔽 →
    Transformer Encoder → 池化剩余节点 → MLP → p_potential(i)。

    将 N 个干预展开为 B*N 个序列, 一次 forward 完成全部推理。

    Input:  Z    [B, N, D]  — 物理融合特征
            mask [B, N]      — 有效节点
    Output: pot  [B, N]      — 移除节点 i 后的剩余稳定度
    """

    def __init__(self, feat_dim: int = 128, n_heads: int = 4, max_blocks: int = 54):
        super().__init__()
        self.max_blocks = max_blocks

        self.pos_embed = nn.Parameter(torch.randn(1, max_blocks, feat_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=n_heads,
                dim_feedforward=feat_dim * 2, batch_first=True,
                dropout=0.1,
            ),
            num_layers=1,
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, Z: torch.Tensor, mask: torch.Tensor):
        B, N, D = Z.shape
        device = Z.device

        # 加入位置编码
        Z_pe = Z + self.pos_embed[:, :N, :]

        # 展开: 每个样本 × N 个干预 → [B*N, N, D]
        # Z_exp[b*N+i] = Z_pe[b], 但位置 i 被替换为 mask_token
        Z_exp = Z_pe.unsqueeze(1).expand(B, N, N, D).reshape(B * N, N, D).clone()
        diag = torch.arange(N, device=device)
        block_offset = torch.arange(B, device=device).unsqueeze(1) * N + diag.unsqueeze(0)
        Z_exp[block_offset.reshape(-1), diag.repeat(B), :] = self.mask_token

        # key_padding_mask: [B*N, N]  True = 忽略
        # 每个干预 i 要屏蔽: padding 位置 + 干预目标 i
        pad_mask = ~mask                                          # [B, N]
        pad_exp = pad_mask.unsqueeze(1).expand(B, N, N).reshape(B * N, N)
        interv_mask = torch.zeros(B * N, N, dtype=torch.bool, device=device)
        interv_mask[block_offset.reshape(-1), diag.repeat(B)] = True
        kp_mask = pad_exp | interv_mask

        # Transformer
        out = self.transformer(Z_exp, src_key_padding_mask=kp_mask)  # [B*N, N, D]

        # Mean-pool over 有效且非干预目标的节点
        pool_valid = (~kp_mask).float().unsqueeze(-1)              # [B*N, N, 1]
        pooled = (out * pool_valid).sum(dim=1) / pool_valid.sum(dim=1).clamp(min=1)

        pot = torch.sigmoid(self.head(pooled).squeeze(-1))         # [B*N]
        pot = pot.reshape(B, N) * mask.float()
        return pot


# ════════════════════════════════════════════════════
#  VP3ENetwork (端到端封装)
# ════════════════════════════════════════════════════

class VP3ENetwork(nn.Module):
    """
    Vision-Physics Predictive Perception Engine.

    Pipeline:
        pcd → InstancePerception → F0, bbox
        F0  → ImplicitTopology   → A_hat
        F0 + A_hat → PhysicalMessagePassing → Z
        Z   → StabilityHead     → stab_hat
        Z   → CausalSTIT        → pot_hat

    Input:
        pcd  [B, N, K, 3], mask [B, N]

    Output dict:
        F0, bbox, A_hat, Z, stab_hat, pot_hat
    """

    def __init__(self, feat_dim: int = 128, gnn_layers: int = 3,
                 n_heads: int = 4, max_blocks: int = 54):
        super().__init__()
        self.perception = InstancePerception(feat_dim)
        self.topology   = ImplicitTopology(feat_dim)
        self.gnn        = PhysicalMessagePassing(feat_dim, gnn_layers)
        self.stab_head  = StabilityHead(feat_dim)
        self.causal     = CausalSTIT(feat_dim, n_heads, max_blocks)

    def forward(self, pcd: torch.Tensor, mask: torch.Tensor, use_causal: bool = True):
        F0, bbox = self.perception(pcd, mask)
        A_hat    = self.topology(F0, mask)
        Z        = self.gnn(F0, A_hat, mask)
        stab_hat = self.stab_head(Z, mask)
        pot_hat  = self.causal(Z, mask) if use_causal else torch.zeros_like(stab_hat)

        return {
            "F0": F0, "bbox": bbox, "A_hat": A_hat,
            "Z": Z, "stab_hat": stab_hat, "pot_hat": pot_hat,
        }
