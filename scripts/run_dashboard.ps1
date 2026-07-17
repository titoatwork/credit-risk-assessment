# Launch the Credit Risk interactive dashboard (Streamlit)
# Usage:  right-click → Run with PowerShell
#    or:  powershell -ExecutionPolicy Bypass -File scripts\run_dashboard.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "ERROR: .venv not found. Create it and install requirements first." -ForegroundColor Red
    exit 1
}

# Prefer full-data academic model when present
$full = Join-Path (Get-Location) "models\full_data"
if (Test-Path (Join-Path $full "credit_risk_bundle.joblib")) {
    $env:CREDIT_RISK_MODELS_DIR = $full
    Write-Host "Using full-data model pack: $full" -ForegroundColor Cyan
}

& $py -m pip install streamlit plotly -q
& $py -m streamlit run dashboard/app.py --server.headless false
