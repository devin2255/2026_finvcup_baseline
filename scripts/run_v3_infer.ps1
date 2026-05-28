param(
    [string]$Config = "configs/whisper_qwen_v3_8g.yaml",
    [string]$BestCkpt = "outputs/v3_8g/checkpoints/best_v3.pt",
    [string]$TopKDir = "outputs/v3_8g/checkpoints/topk",
    [string]$TestRoot = "test",
    [string]$OutDir = "outputs/v3_8g/infer",
    [int]$MaxValidSamples = 20000,
    [string]$Tta = "none",
    [string]$Constraints = "none",
    [string]$Metric = "micro",
    [switch]$UseTopK,
    [switch]$SkipThreshold,
    [string]$Python = "D:/anaconda/envs/finvcup/python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

# Defensive defaults (in case ps file encoding corrupted defaults on Chinese Windows).
if (-not $Metric)      { $Metric = "micro" }
if (-not $Tta)         { $Tta = "none" }
if (-not $Constraints) { $Constraints = "none" }

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python."
}
$pyVer = (& $Python -c "import sys; print(sys.version_info[0]*100+sys.version_info[1])").Trim()
if ([int]$pyVer -lt 310) { throw "Python too old: $pyVer" }

# Build checkpoint list: best + topk (excluding duplicate of best).
$ckpts = @($BestCkpt)
if ($UseTopK -and (Test-Path $TopKDir)) {
    $topk_files = Get-ChildItem -Path $TopKDir -Filter "epoch_*.pt" | Sort-Object Name
    foreach ($f in $topk_files) {
        if ($f.FullName -ne (Resolve-Path $BestCkpt).Path) {
            $ckpts += $f.FullName
        }
    }
}
Write-Host "[v3_infer] python=$Python ver=$pyVer  config=$Config" -ForegroundColor DarkGray
Write-Host "[v3_infer] metric=$Metric  tta=$Tta  constraints=$Constraints  max_valid=$MaxValidSamples" -ForegroundColor DarkGray
Write-Host "[v3_infer] checkpoints (n=$($ckpts.Count)):" -ForegroundColor DarkGray
$ckpts | ForEach-Object { Write-Host "    $_" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ThrFile = Join-Path $OutDir "thresholds_$Metric.json"
$ValidProbs = Join-Path $OutDir "valid_probs.npz"
$TestProbs = Join-Path $OutDir "test_probs.npz"
$PredCsv = Join-Path $OutDir ("pred_" + $Metric + "_" + $Tta + "_" + $Constraints + ".csv")

if (-not $SkipThreshold) {
    Write-Host "[v3_infer] Step 1/2: tune thresholds (metric=$Metric, TTA=$Tta, n=$MaxValidSamples)" -ForegroundColor Cyan
    & $Python -m src.tune_threshold_full `
        --config $Config `
        --checkpoints @ckpts `
        --tta $Tta `
        --metric $Metric `
        --max_valid_samples $MaxValidSamples `
        --output $ThrFile `
        --save_probs $ValidProbs
    if ($LASTEXITCODE -ne 0) { throw "tune_threshold_full failed" }
} else {
    Write-Host "[v3_infer] Step 1/2 skipped: reusing $ThrFile" -ForegroundColor Yellow
}

Write-Host "[v3_infer] Step 2/2: ensemble inference (TTA=$Tta, constraints=$Constraints)" -ForegroundColor Cyan
& $Python -m src.infer_ensemble `
    --config $Config `
    --checkpoints @ckpts `
    --test_root $TestRoot `
    --threshold_file $ThrFile `
    --tta $Tta `
    --constraints $Constraints `
    --output_csv $PredCsv `
    --save_probs $TestProbs
if ($LASTEXITCODE -ne 0) { throw "infer_ensemble failed" }

Write-Host "[v3_infer] Done. Submit: $PredCsv" -ForegroundColor Green
