param(
    [string]$Config = "configs/whisper_qwen0_6b_lmf_8g.yaml",
    [string[]]$Checkpoints = @("outputs/lmf_8g/checkpoints/best_lmf.pt"),
    [string]$TestRoot = "test",
    [string]$OutDir = "outputs/lmf_8g/day1",
    [int]$MaxValidSamples = 20000,
    [string]$Tta = "dual_channel",
    # Default 'none' for post-processing: online metric is micro/sample-wise F1;
    # v1/v2 constraints both cut down c=1 high recall -> ~9pt regression.
    [string]$Constraints = "none",
    [string]$AllZeroFallback = "na",
    [string]$Python = "D:/anaconda/envs/finvcup/python.exe",
    [switch]$SkipThreshold
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if (-not $Tta)             { $Tta = "dual_channel" }
if (-not $Constraints)     { $Constraints = "none" }
if (-not $AllZeroFallback) { $AllZeroFallback = "na" }

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python. Pass -Python <path-to-python.exe> explicitly."
}
$pyVer = (& $Python -c "import sys; print(sys.version_info[0]*100+sys.version_info[1])").Trim()
if ([int]$pyVer -lt 310) {
    throw "Python $Python is too old ($pyVer < 310). This project requires Python 3.10+."
}
Write-Host "[day1] using python: $Python (version code = $pyVer)" -ForegroundColor DarkGray

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ThresholdFile = Join-Path $OutDir "thresholds_full.json"
$PredCsv = Join-Path $OutDir "pred_ensemble.csv"
$ValidProbs = Join-Path $OutDir "valid_probs.npz"
$TestProbs = Join-Path $OutDir "test_probs.npz"

if (-not $SkipThreshold) {
    Write-Host "[day1] Step 1/2: tune thresholds on full valid (TTA=$Tta, max=$MaxValidSamples)" -ForegroundColor Cyan
    & $Python -m src.tune_threshold_full `
        --config $Config `
        --checkpoints @Checkpoints `
        --tta $Tta `
        --max_valid_samples $MaxValidSamples `
        --output $ThresholdFile `
        --save_probs $ValidProbs
    if ($LASTEXITCODE -ne 0) { throw "tune_threshold_full failed" }
} else {
    Write-Host "[day1] Step 1/2 skipped: reusing $ThresholdFile" -ForegroundColor Yellow
}

Write-Host "[day1] Step 2/2: ensemble inference on test (TTA=$Tta, constraints=$Constraints, fallback=$AllZeroFallback)" -ForegroundColor Cyan
& $Python -m src.infer_ensemble `
    --config $Config `
    --checkpoints @Checkpoints `
    --test_root $TestRoot `
    --threshold_file $ThresholdFile `
    --tta $Tta `
    --constraints $Constraints `
    --allzero_fallback $AllZeroFallback `
    --output_csv $PredCsv `
    --save_probs $TestProbs
if ($LASTEXITCODE -ne 0) { throw "infer_ensemble failed" }

Write-Host "[day1] Done. Submit: $PredCsv" -ForegroundColor Green
