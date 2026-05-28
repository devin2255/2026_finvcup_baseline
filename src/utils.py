import json
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: Dict) -> None:
    Path(cfg["paths"]["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["cache_root"]).mkdir(parents=True, exist_ok=True)


def set_env_paths(cfg: Dict) -> None:
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = str(v)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # 验证/测试若只在 rank0 跑全量，其它 rank 会在 barrier 等待；默认 NCCL 约 600s 会超时
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(hours=2),
    )
    return local_rank, world_size, rank


def cleanup_distributed() -> None:
    if is_distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def compute_binary_metrics(labels, probs) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def find_best_f1_threshold(probs, labels, n_steps: int = 200) -> tuple[float, float]:
    probs = np.asarray(probs).astype(float)
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.linspace(0.01, 0.99, n_steps):
        preds = (probs >= threshold).astype(int)
        tp = float((preds * labels).sum())
        fp = float((preds * (1 - labels)).sum())
        fn = float(((1 - preds) * labels).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def _micro_f1_at_thresholds(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> float:
    preds = (probs >= thresholds[None, :]).astype(np.int8)
    tp = float((preds * labels).sum())
    fp = float((preds * (1 - labels)).sum())
    fn = float(((1 - preds) * labels).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return float(2 * precision * recall / max(1e-8, precision + recall))


def _find_best_micro_f1_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    rounds: int = 2,
    n_steps: int = 100,
) -> tuple[np.ndarray, float]:
    """坐标轮转：先各自取 per-label F1 最优阈值作为起点，再做 1~2 轮逐维微调最大化 micro F1。

    复杂度 O(rounds * C * n_steps * N)；N 为样本数。比较稳，但对 N>=2w 推荐 n_steps=80。
    """
    n_labels = probs.shape[1]
    cur_thr = np.full(n_labels, 0.5, dtype=np.float64)
    for j in range(n_labels):
        t, _ = find_best_f1_threshold(probs[:, j], labels[:, j])
        cur_thr[j] = t

    best_micro = _micro_f1_at_thresholds(probs, labels, cur_thr)
    grid = np.linspace(0.01, 0.99, n_steps)
    for _ in range(rounds):
        improved = False
        for j in range(n_labels):
            best_t = cur_thr[j]
            for t in grid:
                cur_thr[j] = t
                m = _micro_f1_at_thresholds(probs, labels, cur_thr)
                if m > best_micro + 1e-6:
                    best_micro = m
                    best_t = t
                    improved = True
            cur_thr[j] = best_t
        if not improved:
            break
    return cur_thr, best_micro


def _sample_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Sample-wise (per-row) F1，对每行取 F1 后求平均。"""
    tp = (preds * labels).sum(axis=1)
    fp = (preds * (1 - labels)).sum(axis=1)
    fn = ((1 - preds) * labels).sum(axis=1)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / np.maximum(1, tp + fn)
    f1 = 2 * precision * recall / np.maximum(1e-8, precision + recall)
    return float(f1.mean())


def compute_multilabel_metrics(labels, probs, label_names=None) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if labels.ndim != 2 or probs.ndim != 2:
        raise ValueError(f"Expected 2D labels/probs, got {labels.shape} and {probs.shape}")
    if labels.shape != probs.shape:
        raise ValueError(f"Shape mismatch: labels {labels.shape} vs probs {probs.shape}")

    n_labels = labels.shape[1]
    if label_names is None:
        label_names = [f"label{i}" for i in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError(f"label_names length {len(label_names)} != n_labels {n_labels}")

    out: Dict[str, float] = {}
    per_acc, per_f1, per_auc, per_best_f1 = [], [], [], []
    per_best_thr = np.zeros(n_labels, dtype=np.float64)
    for i, name in enumerate(label_names):
        y = labels[:, i]
        p = probs[:, i]
        pred = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        best_threshold, best_f1 = find_best_f1_threshold(p, y)
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.5
        out[f"{name}_accuracy"] = acc
        out[f"{name}_f1"] = f1
        out[f"{name}_best_threshold"] = best_threshold
        out[f"{name}_best_f1"] = best_f1
        out[f"{name}_roc_auc"] = auc
        per_acc.append(acc)
        per_f1.append(f1)
        per_auc.append(auc)
        per_best_f1.append(best_f1)
        per_best_thr[i] = best_threshold

    # === micro / sample-wise metrics（与线上 metric 更接近的口径）===
    thr_05 = np.full(n_labels, 0.5, dtype=np.float64)
    preds_05 = (probs >= thr_05[None, :]).astype(np.int8)
    out["micro_f1"] = _micro_f1_at_thresholds(probs, labels, thr_05)
    out["sample_f1"] = _sample_f1(preds_05, labels.astype(np.int8))

    # 用 per-label-best 阈值再算一次 micro / sample（这是 thresholds 落盘后真实的指标）
    preds_pl = (probs >= per_best_thr[None, :]).astype(np.int8)
    out["micro_f1_at_perlabel_thr"] = _micro_f1_at_thresholds(probs, labels, per_best_thr)
    out["sample_f1_at_perlabel_thr"] = _sample_f1(preds_pl, labels.astype(np.int8))

    # 联合搜最优 micro F1 阈值（坐标轮转）
    best_micro_thr, best_micro = _find_best_micro_f1_thresholds(
        probs.astype(np.float32), labels.astype(np.int8), rounds=2, n_steps=80
    )
    out["best_micro_f1"] = float(best_micro)
    preds_bm = (probs >= best_micro_thr[None, :]).astype(np.int8)
    out["best_micro_sample_f1"] = _sample_f1(preds_bm, labels.astype(np.int8))
    for i, name in enumerate(label_names):
        out[f"{name}_best_micro_threshold"] = float(best_micro_thr[i])

    out["macro_accuracy"] = float(np.mean(per_acc))
    out["macro_f1"] = float(np.mean(per_f1))
    out["macro_best_f1"] = float(np.mean(per_best_f1))
    out["macro_roc_auc"] = float(np.mean(per_auc))
    # Alias for backward-compatible save_metric/print flow
    out["accuracy"] = out["macro_accuracy"]
    out["f1"] = out["macro_f1"]
    out["best_f1"] = out["macro_best_f1"]
    out["roc_auc"] = out["macro_roc_auc"]
    return out


@torch.no_grad()
def compute_gaussian_soft_f1_sequence(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5,
    sigma: float = 2.0,
    avg_class_indices: tuple[int, ...] = (1, 2, 3),
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    高斯平滑时序 soft-f1（按类别做 TP/FP/FN 的 soft 版本）。

    probs: [B, C, T]，每个时间步每类的概率（例如 softmax 后）。
    targets: [B, T]，每个时间步的类别id（0..C-1）。
    """
    if probs.ndim != 3:
        raise ValueError(f"Expected probs shape [B,C,T], got {tuple(probs.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets shape [B,T], got {tuple(targets.shape)}")
    b, c, t = probs.shape
    if c != num_classes:
        raise ValueError(f"probs C={c} != num_classes={num_classes}")
    if targets.shape[0] != b or targets.shape[1] != t:
        raise ValueError(f"targets shape {tuple(targets.shape)} not match probs {tuple(probs.shape)}")

    targets_onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 2, 1).float()  # [B,C,T]

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, device=probs.device).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.max()
    kernel = kernel.view(1, 1, -1)  # [1,1,K]

    padding = kernel_size // 2

    # 平滑 targets：对每个 (B,C) 位置做 conv1d
    targets_flat = targets_onehot.reshape(b * c, 1, t)  # [B*C,1,T]
    targets_smooth = F.conv1d(targets_flat, kernel, padding=padding).view(b, c, t)  # [B,C,T]

    # soft TP/FP/FN 形式的一种等价推导
    tp = (probs * targets_smooth).sum(dim=(0, 2))  # [C]
    sum_p = probs.sum(dim=(0, 2))  # [C]
    sum_t_true = targets_onehot.sum(dim=(0, 2))  # [C]

    f1 = (2 * tp + epsilon) / (sum_p + sum_t_true + epsilon)  # [C]
    avg_class_indices = tuple(avg_class_indices)
    score = f1[list(avg_class_indices)].mean()

    return {
        "soft_macro_f1": float(score.item()),
        "soft_f1_per_class_mean": float(f1.mean().item()),
    }


def apply_label_constraints(
    preds: list[int],
    probs: list[float],
    label_cols: list[str],
) -> list[int]:
    """Apply domain mutual-exclusion rules on multi-label predictions."""
    out = list(preds)
    name_to_idx = {name: idx for idx, name in enumerate(label_cols)}

    na_idx = name_to_idx.get("na")
    if na_idx is not None and out[na_idx] == 1:
        for name in ("c", "i", "bc", "t"):
            idx = name_to_idx.get(name)
            if idx is not None:
                out[idx] = 0

    c_idx = name_to_idx.get("c")
    t_idx = name_to_idx.get("t")
    if c_idx is not None and t_idx is not None and out[c_idx] == 1 and out[t_idx] == 1:
        if probs[c_idx] >= probs[t_idx]:
            out[t_idx] = 0
        else:
            out[c_idx] = 0

    return out


def apply_label_constraints_v2(
    preds: list[int],
    probs: list[float],
    label_cols: list[str],
    *,
    na_strong_thr: float = 0.6,
    na_max_other_prob: float = 0.5,
    ct_margin: float = 0.0,
    allzero_fallback: str = "na",
) -> list[int]:
    """Softer constraints + all-zero fallback.

    - NA 只在 (P(NA) >= na_strong_thr) 且其它 4 类的最大 prob < na_max_other_prob 时
      才硬压制其它标签；否则保留 NA + 其它标签共存（线上 micro-F1 友好）。
    - C/T 互斥仅在 |P(C) - P(T)| > ct_margin 时裁更弱的一方；否则两者都保留。
    - 全 0 兜底：若 5 类预测都为 0，按 allzero_fallback 行为打开：
        * "na"           → 强制 NA=1（最安全）
        * "argmax"       → 把概率最大的那一类置 1
        * "na_or_argmax" → 若 P(NA) 最大置 NA=1，否则置 argmax=1
        * "none"         → 不做处理
    """
    out = list(preds)
    name_to_idx = {name: idx for idx, name in enumerate(label_cols)}

    na_idx = name_to_idx.get("na")
    other_idxs = [name_to_idx[n] for n in ("c", "i", "bc", "t") if n in name_to_idx]

    if na_idx is not None and out[na_idx] == 1 and other_idxs:
        other_max_prob = max(float(probs[i]) for i in other_idxs)
        if float(probs[na_idx]) >= na_strong_thr and other_max_prob < na_max_other_prob:
            for i in other_idxs:
                out[i] = 0

    c_idx = name_to_idx.get("c")
    t_idx = name_to_idx.get("t")
    if c_idx is not None and t_idx is not None and out[c_idx] == 1 and out[t_idx] == 1:
        diff = float(probs[c_idx]) - float(probs[t_idx])
        if abs(diff) > ct_margin:
            if diff >= 0:
                out[t_idx] = 0
            else:
                out[c_idx] = 0

    if sum(out) == 0 and allzero_fallback != "none":
        if allzero_fallback == "na" and na_idx is not None:
            out[na_idx] = 1
        elif allzero_fallback == "argmax":
            j = int(max(range(len(probs)), key=lambda k: float(probs[k])))
            out[j] = 1
        elif allzero_fallback == "na_or_argmax":
            if na_idx is not None and na_idx == int(
                max(range(len(probs)), key=lambda k: float(probs[k]))
            ):
                out[na_idx] = 1
            else:
                j = int(max(range(len(probs)), key=lambda k: float(probs[k])))
                out[j] = 1

    return out


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
