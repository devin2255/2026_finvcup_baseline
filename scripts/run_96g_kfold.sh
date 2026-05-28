#!/bin/bash
set -e

# 如果在租用的服务器上，你可能需要激活环境
# source /path/to/your/miniconda3/bin/activate finvcup

CONFIG="configs/whisper_qwen_lmf_96g_kfold.yaml"
NUM_FOLDS=5

echo "==============================================="
echo "  Starting 5-Fold Training on 96G Server"
echo "==============================================="

for (( i=0; i<$NUM_FOLDS; i++ ))
do
    echo ""
    echo ">>> Starting Fold $i / $((NUM_FOLDS-1)) <<<"
    echo ""
    
    python -m src.train --config $CONFIG --num_folds $NUM_FOLDS --fold_idx $i
    
    if [ $? -ne 0 ]; then
        echo "Fold $i failed!"
        exit 1
    fi
done

echo "All $NUM_FOLDS folds completed successfully."

# 推理集成
echo ">>> Running K-Fold Inference Ensemble <<<"
python -m src.infer_ensemble \
    --config $CONFIG \
    --checkpoints outputs/lmf_96g_kfold/checkpoints/fold_*/best_lmf.pt \
    --test_root test \
    --output_csv outputs/lmf_96g_kfold/submission_kfold_96g.csv \
    --save_probs outputs/lmf_96g_kfold/test_probs_kfold_96g.npz \
    --constraints none

echo "Done! Final submission saved to outputs/lmf_96g_kfold/submission_kfold_96g.csv"
