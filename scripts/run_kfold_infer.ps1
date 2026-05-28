param(
    [string]$Config = "configs/whisper_qwen_lmf_8g_kfold.yaml",
    [string]$OutDir = "outputs/lmf_8g_kfold",
    [string]$Python = "D:/anaconda/envs/finvcup/python.exe",
    [string]$Constraints = "none"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

# Find all fold checkpoints
$Checkpoints = @()
for ($i = 0; $i -lt 5; $i++) {
    $ckpt = "$OutDir/checkpoints/fold_$i/best_lmf.pt"
    if (Test-Path $ckpt) {
        $Checkpoints += $ckpt
    }
}

if ($Checkpoints.Count -eq 0) {
    throw "No checkpoints found in $OutDir/checkpoints/fold_*"
}

Write-Host "Found $($Checkpoints.Count) fold checkpoints for ensembling." -ForegroundColor Cyan

# 我们可以先收集所有 validation 的 test_probs 吗？不过如果是纯 inference：
# 先用 infer_ensemble.py 直接跑，因为 lmf_8g 本身的阈值估计比较准，或者我们简单 average 他们的 logits/probs 和 thresholds。
# 这里我们用第一个 fold 的阈值文件（因为 infer_ensemble 现在直接接受一个 threshold file 或者从 ckpt 里取）
# 为了更严谨，其实可以在代码里修改 infer_ensemble.py，如果提供多个 ckpt，直接取它们的 average threshold。
# 已经在 src/infer_ensemble.py 里实现了这个逻辑吗？
# 我们来看看。

$cmd = @(
    $Python, "src/infer_ensemble.py",
    "--config", $Config,
    "--checkpoints"
) + $Checkpoints + @(
    "--output_csv", "$OutDir/submission_kfold_ensemble.csv",
    "--save_probs", "$OutDir/test_probs_kfold_ensemble.npz",
    "--constraints", $Constraints
)

Write-Host "Running inference ensemble..." -ForegroundColor Cyan
& $cmd

Write-Host "Done! Saved to $OutDir/submission_kfold_ensemble.csv" -ForegroundColor Green
