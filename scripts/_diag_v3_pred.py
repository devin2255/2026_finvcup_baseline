"""Diagnose v3 prediction vs lmf_8g & day1."""
import csv
import json
from pathlib import Path

import numpy as np

paths = {
    "lmf_8g_old (0.7285)": "outputs/lmf_8g/pred.csv",
    "day1_rescue2 (0.7061)": "outputs/lmf_8g/day1/pred_rescue_2_new_thr_none.csv",
    "v3_topk (0.6970)": "outputs/v3_8g/infer_topk_final/pred_micro_none_none.csv",
}


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(rows):
    cols = [c for c in rows[0].keys() if c != "segment_id"]
    n = len(rows)
    s = {c: sum(int(r[c]) for r in rows) for c in cols}
    allzero = sum(1 for r in rows if all(int(r[c]) == 0 for c in cols))
    only_na = sum(1 for r in rows if int(r["na"]) == 1 and all(int(r[c]) == 0 for c in cols if c != "na"))
    no_na = sum(1 for r in rows if int(r["na"]) == 0)
    return n, s, allzero, only_na, no_na, cols


all_rows = {name: load_rows(p) for name, p in paths.items()}
print(f"{'name':<28} {'n':>5}  {'c':>6}  {'na':>6}  {'i':>6}  {'bc':>6}  {'t':>6}  {'allzero':>7}  {'only_na':>7}  {'no_na':>6}")
for name, rows in all_rows.items():
    n, s, az, ona, nna, cols = stats(rows)
    print(
        f"{name:<28} {n:>5}  "
        f"{s['c']:>6}  {s['na']:>6}  {s['i']:>6}  {s['bc']:>6}  {s['t']:>6}  "
        f"{az:>7}  {ona:>7}  {nna:>6}"
    )

# Pairwise diff matrix
print("\n--- Pairwise flip counts (rows: new ckpt; cols: label) ---")
ref = all_rows["lmf_8g_old (0.7285)"]
ref_d = {r["segment_id"]: r for r in ref}
cols = ["c", "na", "i", "bc", "t"]
for name in ["day1_rescue2 (0.7061)", "v3_topk (0.6970)"]:
    cur = {r["segment_id"]: r for r in all_rows[name]}
    keys = sorted(set(ref_d) & set(cur))
    print(f"\n  vs lmf_8g_old | {name}")
    for c in cols:
        f01 = sum(1 for k in keys if int(ref_d[k][c]) == 0 and int(cur[k][c]) == 1)
        f10 = sum(1 for k in keys if int(ref_d[k][c]) == 1 and int(cur[k][c]) == 0)
        print(f"    {c}: old0->new1={f01:>4}  old1->new0={f10:>4}  net={f01 - f10:+d}")

# Load v3 probs (we saved them) and see what threshold would have given best
probs_path = Path("outputs/v3_8g/infer_topk_final/test_probs.npz")
thr_path = Path("outputs/v3_8g/infer_topk_final/thresholds_micro.json")
print(f"\n--- v3 thresholds used ---")
thr = json.load(open(thr_path, "r", encoding="utf-8"))
for k, v in thr.get("per_label", {}).items():
    print(f"  {k}: chosen_thr={v.get('chosen_threshold', v.get('threshold')):.4f}  "
          f"pl_f1={v['f1']:.4f}  pos_rate={v['pos_rate']:.4f}  "
          f"chosen_f1={v.get('chosen_f1', 'n/a')}")
print(f"  valid_micro_f1_chosen={thr.get('micro_f1_chosen')}")
print(f"  valid_macro_best_f1={thr.get('macro_best_f1')}")

z = np.load(probs_path, allow_pickle=True)
probs = z["probs"]  # [1000, 5]
seg_ids = list(z["segment_id"])
label_cols = list(z["label_cols"])
print(f"\n--- v3 test probability statistics (test n={probs.shape[0]}) ---")
for j, c in enumerate(label_cols):
    print(f"  {c}: mean={probs[:, j].mean():.4f}  std={probs[:, j].std():.4f}  "
          f"p10={np.percentile(probs[:, j], 10):.3f}  p50={np.percentile(probs[:, j], 50):.3f}  "
          f"p90={np.percentile(probs[:, j], 90):.3f}  >=0.5={(probs[:, j] >= 0.5).mean():.3f}")

# Probability ECDF cross thresholds 0.3/0.5/0.7/0.9
print(f"\n--- v3 # positive under various thresholds ---")
for thr_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    counts = [(probs[:, j] >= thr_val).sum() for j in range(len(label_cols))]
    print(f"  thr={thr_val:.2f}: c={counts[0]:>4} na={counts[1]:>4} i={counts[2]:>4} bc={counts[3]:>4} t={counts[4]:>4}")
