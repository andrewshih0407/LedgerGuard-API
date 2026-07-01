# LedgerGuard — GPU training launcher
# Runs the training pipeline with live, unbuffered output in PowerShell.
#
# Usage:
#   .\run_training.ps1                 # trains on real credit-card dataset (best metrics)
#   .\run_training.ps1 demo            # trains on synthetic demo data (fast)
#
# The -u flag forces Python to stream output live (no buffering).

param(
    [string]$Dataset = "creditcard"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  LedgerGuard Training Pipeline" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

if ($Dataset -eq "demo") {
    Write-Host "Training on synthetic demo data..." -ForegroundColor Yellow
    python -u scripts/train.py --csv sample_data/demo_transactions.csv --save-dir models_saved/demo
}
else {
    Write-Host "Training on REAL credit-card fraud dataset (downloads ~143MB first time)..." -ForegroundColor Yellow
    python -u scripts/train.py --dataset creditcard --save-dir models_saved/creditcard
}

Write-Host ""
Write-Host "Done. Model saved under models_saved/" -ForegroundColor Green
