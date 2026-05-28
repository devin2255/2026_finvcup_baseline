"""WeightedRandomSampler 工厂：按标签稀疏度给样本权重。

设计思路：
- 全负样本（label_vec 全 0）权重为 base_neg_weight（默认 1.0）。
- 多标签样本权重 = 1 + sum_j (label_j * inv_freq_j ** alpha)，其中 inv_freq_j = N / pos_j。
- 然后整体 clip 到 [1.0, max_weight]，防止极端稀疏标签的单条样本权重过大导致过拟合。
- 默认 alpha=0.5（开根号软化）；alpha=0 退化为均匀采样；alpha=1 完全按 inv_freq 加权。

在 lmf_8g 数据集（130w 训练样本）上的预期效果（label_vec 顺序 [C,NA,I,BC,T]）：
  C: pos_rate~80%, NA~40%, I~14%, BC~3.6%, T~30%
  各标签 inv_freq ≈ [1.25, 2.5, 7, 27, 3.3]
  开根号后 weight 加项约 [1.12, 1.58, 2.65, 5.2, 1.82]
  → BC 正样本被采到的频率从 3.6% 提到 ~15~18%（取决于共现）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from torch.utils.data import WeightedRandomSampler


def build_label_weighted_sampler(
    samples: Sequence,
    *,
    alpha: float = 0.5,
    max_weight: float = 8.0,
    base_neg_weight: float = 1.0,
    num_samples: int | None = None,
    seed: int | None = None,
) -> WeightedRandomSampler:
    """构造按标签稀疏度加权的 sampler。samples 必须有 .label_vec 属性。"""
    y_mat = np.asarray([s.label_vec for s in samples], dtype=np.float32)  # [N, C]
    n = y_mat.shape[0]
    pos = y_mat.sum(axis=0)
    inv_freq = n / np.maximum(1.0, pos)  # [C]
    inv_freq = inv_freq ** float(alpha)

    w = base_neg_weight + (y_mat * inv_freq[None, :]).sum(axis=1)  # [N]
    w = np.clip(w, base_neg_weight, float(max_weight)).astype(np.float64)

    generator = None
    if seed is not None:
        import torch

        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return WeightedRandomSampler(
        weights=w.tolist(),
        num_samples=int(num_samples) if num_samples is not None else n,
        replacement=True,
        generator=generator,
    )


def summarize_weights(samples: Sequence, *, alpha: float = 0.5, max_weight: float = 8.0) -> dict:
    """返回权重分布摘要，便于配置时确认。"""
    y_mat = np.asarray([s.label_vec for s in samples], dtype=np.float32)
    n = y_mat.shape[0]
    pos = y_mat.sum(axis=0)
    inv_freq = n / np.maximum(1.0, pos)
    inv_freq = inv_freq ** float(alpha)
    w = 1.0 + (y_mat * inv_freq[None, :]).sum(axis=1)
    w = np.clip(w, 1.0, float(max_weight))
    return {
        "n_samples": int(n),
        "pos_per_label": pos.tolist(),
        "pos_rate_per_label": (pos / max(1, n)).tolist(),
        "inv_freq_pow_alpha": inv_freq.tolist(),
        "weight_mean": float(w.mean()),
        "weight_max": float(w.max()),
        "weight_p95": float(np.percentile(w, 95)),
        "weight_p50": float(np.percentile(w, 50)),
        "alpha": float(alpha),
        "max_weight_cap": float(max_weight),
    }
