#!/usr/bin/env bash
# Day 1 提分脚本：全量 valid 调阈值 + 双声道 TTA + v2 软约束推理
set -euo pipefail

CONFIG="${CONFIG:-configs/whisper_qwen0_6b_lmf_8g.yaml}"
CHECKPOINTS="${CHECKPOINTS:-outputs/lmf_8g/checkpoints/best_lmf.pt}"
TEST_ROOT="${TEST_ROOT:-test}"
OUT_DIR="${OUT_DIR:-outputs/lmf_8g/day1}"
MAX_VALID="${MAX_VALID:-20000}"
TTA="${TTA:-dual_channel}"
# 默认 none：线上 metric 是 micro/sample-wise F1，v1/v2 后处理会砍 c=1 召回导致 -9pt
CONSTRAINTS="${CONSTRAINTS:-none}"
ALLZERO="${ALLZERO:-na}"
SKIP_THRESHOLD="${SKIP_THRESHOLD:-0}"

mkdir -p "$OUT_DIR"
THR_FILE="$OUT_DIR/thresholds_full.json"
PRED_CSV="$OUT_DIR/pred_ensemble.csv"
VALID_PROBS="$OUT_DIR/valid_probs.npz"
TEST_PROBS="$OUT_DIR/test_probs.npz"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

if [[ "$SKIP_THRESHOLD" != "1" ]]; then
  echo "[day1] Step 1/2: tune thresholds on full valid (TTA=$TTA, max=$MAX_VALID)"
  python -m src.tune_threshold_full \
    --config "$CONFIG" \
    --checkpoints $CHECKPOINTS \
    --tta "$TTA" \
    --max_valid_samples "$MAX_VALID" \
    --output "$THR_FILE" \
    --save_probs "$VALID_PROBS"
else
  echo "[day1] Step 1/2 skipped: reusing $THR_FILE"
fi

echo "[day1] Step 2/2: ensemble inference (TTA=$TTA, constraints=$CONSTRAINTS, fallback=$ALLZERO)"
python -m src.infer_ensemble \
  --config "$CONFIG" \
  --checkpoints $CHECKPOINTS \
  --test_root "$TEST_ROOT" \
  --threshold_file "$THR_FILE" \
  --tta "$TTA" \
  --constraints "$CONSTRAINTS" \
  --allzero_fallback "$ALLZERO" \
  --output_csv "$PRED_CSV" \
  --save_probs "$TEST_PROBS"

echo "[day1] Done. Submit: $PRED_CSV"
