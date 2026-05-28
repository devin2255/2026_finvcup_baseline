param(
    [string]$Config = "configs/whisper_qwen_lmf_8g_kfold.yaml",
    [int]$NumFolds = 5,
    [string]$Python = "D:/anaconda/envs/finvcup/python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if (-not (Test-Path $Python)) { throw "Python not found: $Python" }

for ($i = 0; $i -lt $NumFolds; $i++) {
    Write-Host "`n===============================================" -ForegroundColor Cyan
    Write-Host ">>> Starting Fold $i / $($NumFolds-1) <<<" -ForegroundColor Cyan
    Write-Host "===============================================`n" -ForegroundColor Cyan

    $cmd = @("-m", "src.train", "--config", $Config, "--num_folds", $NumFolds, "--fold_idx", $i)
    & $Python @cmd
    if ($LASTEXITCODE -ne 0) { throw "Fold $i failed with exit $LASTEXITCODE" }
}

Write-Host "All $NumFolds folds completed." -ForegroundColor Green
