"""V3 应急救援：复用 test_probs.npz，尝试多种阈值策略。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

V3_DIR = Path("outputs/v3_8g/infer_topk_final")
V3_TEST_PROBS = V3_DIR / "test_probs.npz"
V3_VALID_PROBS = V3_DIR / "valid_probs.npz"

LABEL_COLS = ["c", "na", "i", "bc", "t"]

# === 训练集真实正率（来自 weighted_sampler_summary.json）===
TRAIN_POS_RATE = {
    "c": 0.9424, "na": 0.6583, "i": 0.1400, "bc": 0.0363, "t": 0.2619,
}
# === lmf_8g 上线版预测分布（线上 0.7285 时各标签预测正率）===
LMF8G_PRED_RATE = {
    "c": 0.975, "na": 0.928, "i": 0.052, "bc": 0.017, "t": 0.510,
}

zt = np.load(V3_TEST_PROBS, allow_pickle=True)
probs_t = zt["probs"]  # [1000, 5]
seg_ids = list(zt["segment_id"])
n_test = probs_t.shape[0]

zv = np.load(V3_VALID_PROBS, allow_pickle=True)
probs_v = zv["probs"]
labels_v = zv["labels"]

# Load chosen thresholds (failed)
chosen_thr = json.load(open(V3_DIR / "thresholds_micro.json", "r", encoding="utf-8"))["thresholds"]

print("=== Test prob distribution ===")
for j, c in enumerate(LABEL_COLS):
    print(f"  {c}: mean={probs_t[:, j].mean():.4f}  median={np.median(probs_t[:, j]):.4f}")


def thr_from_quantile(probs_col: np.ndarray, target_rate: float) -> float:
    """让 (probs >= thr) 的比例 ≈ target_rate。"""
    q = max(0.0, min(1.0, 1.0 - target_rate))
    return float(np.quantile(probs_col, q))


def build_thresholds(strategy: str) -> dict:
    if strategy == "v3_chosen":
        return dict(chosen_thr)
    if strategy == "quantile_train_pos":
        # 让 test 预测正率匹配训练集 pos rate
        return {c: thr_from_quantile(probs_t[:, j], TRAIN_POS_RATE[c]) for j, c in enumerate(LABEL_COLS)}
    if strategy == "quantile_lmf8g_rate":
        # 让 test 预测正率匹配 lmf_8g 上线版分布
        return {c: thr_from_quantile(probs_t[:, j], LMF8G_PRED_RATE[c]) for j, c in enumerate(LABEL_COLS)}
    if strategy == "v3_minus_005":
        return {c: max(0.01, chosen_thr[c] - 0.05) for c in LABEL_COLS}
    if strategy == "v3_minus_010":
        return {c: max(0.01, chosen_thr[c] - 0.10) for c in LABEL_COLS}
    if strategy == "v3_minus_015":
        return {c: max(0.01, chosen_thr[c] - 0.15) for c in LABEL_COLS}
    if strategy == "valid_perlabel_best":
        # 重新用 valid 上的 per-label best F1 阈值
        thr = {}
        for j, c in enumerate(LABEL_COLS):
            p = probs_v[:, j]
            y = labels_v[:, j]
            best_t, best_f1 = 0.5, 0.0
            for t in np.linspace(0.01, 0.99, 200):
                pred = (p >= t).astype(int)
                tp = (pred * y).sum(); fp = (pred * (1 - y)).sum(); fn = ((1 - pred) * y).sum()
                prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
                f1 = 2 * prec * rec / max(1e-8, prec + rec)
                if f1 > best_f1:
                    best_f1 = f1; best_t = t
            thr[c] = float(best_t)
        return thr
    raise ValueError(strategy)


def micro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = (preds * labels).sum(); fp = (preds * (1 - labels)).sum(); fn = ((1 - preds) * labels).sum()
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return float(2 * p * r / max(1e-8, p + r))


def sample_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = (preds * labels).sum(axis=1)
    fp = (preds * (1 - labels)).sum(axis=1)
    fn = ((1 - preds) * labels).sum(axis=1)
    p = tp / np.maximum(1, tp + fp); r = tp / np.maximum(1, tp + fn)
    return float((2 * p * r / np.maximum(1e-8, p + r)).mean())


def apply_thr(probs: np.ndarray, thr: dict) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.int8)
    for j, c in enumerate(LABEL_COLS):
        out[:, j] = (probs[:, j] >= thr[c]).astype(np.int8)
    return out


strategies = [
    "v3_chosen",
    "valid_perlabel_best",
    "v3_minus_005",
    "v3_minus_010",
    "v3_minus_015",
    "quantile_train_pos",
    "quantile_lmf8g_rate",
]

results = []
print("\n=== Strategy comparison ===")
print(f"{'strategy':<25}  {'thr_c':>6}  {'thr_na':>6}  {'thr_i':>6}  {'thr_bc':>6}  {'thr_t':>6}  ||  "
      f"{'valid_micro':>10}  {'valid_sample':>12}  ||  "
      f"{'test_c%':>7}  {'test_na%':>8}  {'test_i%':>7}  {'test_bc%':>7}  {'test_t%':>7}")
print("-" * 160)
for name in strategies:
    thr = build_thresholds(name)
    pv = apply_thr(probs_v, thr)
    mv = micro_f1(pv, labels_v.astype(np.int8))
    sv = sample_f1(pv, labels_v.astype(np.int8))
    pt = apply_thr(probs_t, thr)
    rates = [pt[:, j].mean() for j in range(5)]
    print(
        f"{name:<25}  {thr['c']:>6.3f}  {thr['na']:>6.3f}  {thr['i']:>6.3f}  {thr['bc']:>6.3f}  {thr['t']:>6.3f}  ||  "
        f"{mv:>10.4f}  {sv:>12.4f}  ||  "
        f"{rates[0]*100:>6.1f}%  {rates[1]*100:>7.1f}%  {rates[2]*100:>6.1f}%  {rates[3]*100:>6.1f}%  {rates[4]*100:>6.1f}%"
    )
    results.append((name, thr, mv, sv, pt))

# 选择最好的 valid_sample_f1 + 与 lmf_8g 分布最接近的，导出 csv
print("\n=== Recommended candidates ===")
# 排序：先看 valid_sample_f1
results.sort(key=lambda r: -r[3])
for rank, (name, thr, mv, sv, pt) in enumerate(results[:5], 1):
    out_csv = V3_DIR / f"pred_rescue_{rank}_{name}.csv"
    rows = []
    for i, sid in enumerate(seg_ids):
        row = {"segment_id": str(sid)}
        for j, c in enumerate(LABEL_COLS):
            row[c] = int(pt[i, j])
        rows.append(row)
    rows.sort(key=lambda r: r["segment_id"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id"] + LABEL_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"  [{rank}] {name}  valid_sample_f1={sv:.4f}  valid_micro={mv:.4f}  -> {out_csv}")
