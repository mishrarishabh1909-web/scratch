# Activation script for Windows PowerShell
# Run this to activate the complete environment

Write-Host "=" * 80
Write-Host "COMPLETE ENVIRONMENT ACTIVATION" -ForegroundColor Cyan
Write-Host "=" * 80
Write-Host ""

$venv_path = "$PSScriptRoot\venv_complete"

if (-not (Test-Path $venv_path)) {
    Write-Host "ERROR: venv_complete not found!" -ForegroundColor Red
    Write-Host "Please create it first with: python -m venv venv_complete" -ForegroundColor Yellow
    exit 1
}

Write-Host "Activating environment..." -ForegroundColor Green
& "$venv_path\Scripts\Activate.ps1"

Write-Host ""
Write-Host "=" * 80
Write-Host "✓ ENVIRONMENT ACTIVE" -ForegroundColor Green
Write-Host "=" * 80
Write-Host ""
Write-Host "Environment location: $venv_path" -ForegroundColor Cyan
Write-Host "Python version:" -ForegroundColor Cyan
python --version
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  python main_complete.py                    (Run main training)"
Write-Host "  python comprehensive_analysis.py           (Analysis with custom env)"
Write-Host "  python comprehensive_analysis.py --use-yafs (Analysis with YAFS)"
Write-Host ""
Write-Host "Tip: Use 'deactivate' to exit environment" -ForegroundColor Yellow
Write-Host ""
