param(
    [string]$Config = "configs/whisper_qwen_v3_8g.yaml",
    [string]$Resume = "",
    [int]$Epochs = 0,
    [int]$MaxStepsPerEpoch = 0,
    [int]$EvalValidSampleCount = 0,
    [int]$EvalValidMaxBatches = 0,
    [string]$Python = "D:/anaconda/envs/finvcup/python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python. Pass -Python <path-to-python.exe> explicitly."
}
$pyVer = (& $Python -c "import sys; print(sys.version_info[0]*100+sys.version_info[1])").Trim()
if ([int]$pyVer -lt 310) {
    throw "Python $Python is too old ($pyVer < 310). This project requires Python 3.10+."
}
Write-Host "[train_v3] python=$Python ver=$pyVer  config=$Config" -ForegroundColor DarkGray

$cmd = @("-m", "src.train", "--config", $Config)
if ($Resume) { $cmd += @("--resume", $Resume) }
if ($Epochs -gt 0) { $cmd += @("--epochs", "$Epochs") }
if ($MaxStepsPerEpoch -gt 0) { $cmd += @("--max_steps_per_epoch", "$MaxStepsPerEpoch") }
if ($EvalValidSampleCount -gt 0) { $cmd += @("--eval_valid_sample_count", "$EvalValidSampleCount") }
if ($EvalValidMaxBatches -gt 0) { $cmd += @("--eval_valid_max_batches", "$EvalValidMaxBatches") }

Write-Host "[train_v3] launching: $Python $($cmd -join ' ')" -ForegroundColor Cyan
& $Python @cmd
if ($LASTEXITCODE -ne 0) { throw "training failed with exit $LASTEXITCODE" }
Write-Host "[train_v3] done." -ForegroundColor Green
