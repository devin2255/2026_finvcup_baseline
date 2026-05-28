#!/bin/bash
set -e

CONFIG="configs/whisper_qwen_lmf_96g_kfold.yaml"
OUT_DIR="outputs/lmf_96g_kfold"
CONSTRAINTS="none"

CKPT="$OUT_DIR/checkpoints/fold_0/best_lmf.pt"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    exit 1
fi

echo "Found fold_0 checkpoint. Running fast inference..."

python -m src.infer_ensemble \
    --config $CONFIG \
    --checkpoints $CKPT \
    --test_root test \
    --output_csv $OUT_DIR/submission_fold0_epoch4.csv \
    --save_probs $OUT_DIR/test_probs_fold0.npz \
    --constraints $CONSTRAINTS

echo "Done! Final submission saved to $OUT_DIR/submission_fold0_epoch4.csv"
