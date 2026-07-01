param([int]$Port = 8000)
Set-Location $PSScriptRoot
Write-Host "Starting LedgerGuard API server on http://localhost:$Port ..."
python -m uvicorn src.ledgerguard.api.server:app --reload --port $Port
