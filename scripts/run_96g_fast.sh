#!/bin/bash
set -e

# 在租用的服务器上修改为正确的配置文件路径
CONFIG="configs/whisper_qwen_lmf_96g_fast.yaml"
OUT_DIR="outputs/lmf_96g_fast"
CONSTRAINTS="none"

echo "==============================================="
echo "  Starting High-Speed Single Model Training"
echo "==============================================="

# 不使用 K-Fold，直接启动普通训练
python -m src.train --config $CONFIG

echo "Training completed."

# 获取最新训练的 Checkpoint
CKPT="$OUT_DIR/checkpoints/best_lmf_fast.pt"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    exit 1
fi

echo "Running full-validation threshold tuning..."
# 使用整个验证集来精调最佳阈值
python -m src.tune_threshold_full \
    --config $CONFIG \
    --checkpoints $CKPT \
    --valid_root train \
    --metric macro \
    --output_thresholds $OUT_DIR/logs/best_thresholds_full.json

echo "Running fast inference..."
python -m src.infer_ensemble \
    --config $CONFIG \
    --checkpoints $CKPT \
    --test_root test \
    --threshold_file $OUT_DIR/logs/best_thresholds_full.json \
    --output_csv $OUT_DIR/submission_96g_fast.csv \
    --save_probs $OUT_DIR/test_probs_96g_fast.npz \
    --constraints $CONSTRAINTS

echo "Done! Final submission saved to $OUT_DIR/submission_96g_fast.csv"
