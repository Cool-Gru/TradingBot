$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Run this from C:\FunStuff\TradingBot. The .venv folder was not found."
}

.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard.py
