"""多标签损失。

- BCEWithLogitsLoss：基线。
- MultiLabelFocalLoss：标准 focal+pos_weight。
- AsymmetricLoss (ASL)：长尾多标签的强基线，对负样本独立加 γ⁻=4，正样本 γ⁺=0，
  并允许给负样本 logit 做"概率裁切（probability shifting, m）"。比 focal+BCE 在 BC 这种
  极稀疏标签上通常 +1~2pt（Ben-Baruch et al., 2020）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma: float, pos_weight: torch.Tensor):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        return ((1 - p_t) ** self.gamma * bce).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification (Ben-Baruch et al., 2020).

    Args:
        gamma_pos: focal 因子对正样本的指数（默认 0，不下权正样本）
        gamma_neg: focal 因子对负样本的指数（默认 4）
        clip: 概率移位（m），把 P(negative) 高于 clip 的部分置 0 → 强力抑制 easy negatives
        eps: 数值稳定项
        pos_weight: 每个标签的正样本加权（可选）
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Basic CE
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Probability shifting on negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Asymmetric Focusing
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        loss_pos = targets * log_pos
        loss_neg = (1 - targets) * log_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            # Detach focal weight per Ben-Baruch et al. (avoid gradient flow through it)
            with torch.no_grad():
                pt0 = xs_pos * targets
                pt1 = xs_neg * (1 - targets)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                focal_weight = torch.pow(1 - pt, one_sided_gamma)
            loss_pos = loss_pos * focal_weight
            loss_neg = loss_neg * focal_weight

        loss = loss_pos + loss_neg
        if self.pos_weight is not None:
            # 在 ASL 里 pos_weight 只乘正样本项
            loss = loss + (self.pos_weight - 1.0) * loss_pos
        return -loss.mean()


def build_criterion(cfg_train: dict, pos_weight: torch.Tensor) -> nn.Module:
    """Build training criterion from cfg['train']. Backward compatible:

    - cfg.train.loss = "bce" | "focal" | "asl"  (default "focal" if focal_gamma>0 else "bce")
    - For "focal": cfg.train.focal_gamma
    - For "asl":   cfg.train.asl_gamma_pos / asl_gamma_neg / asl_clip
    """
    loss_type = str(cfg_train.get("loss", "")).lower()
    if not loss_type:
        loss_type = "focal" if float(cfg_train.get("focal_gamma", 0.0)) > 0 else "bce"

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_type == "focal":
        return MultiLabelFocalLoss(float(cfg_train.get("focal_gamma", 1.0)), pos_weight)
    if loss_type == "asl":
        return AsymmetricLoss(
            gamma_pos=float(cfg_train.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(cfg_train.get("asl_gamma_neg", 4.0)),
            clip=float(cfg_train.get("asl_clip", 0.05)),
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unknown loss type: {loss_type}")
