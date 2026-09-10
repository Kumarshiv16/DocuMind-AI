# DocuMind AI PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "              Launching DocuMind AI" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".\venv\Scripts\streamlit.exe")) {
    Write-Host "[ERROR] Virtual environment not found. Please set up the environment first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
& ".\venv\Scripts\streamlit.exe" run app.py
