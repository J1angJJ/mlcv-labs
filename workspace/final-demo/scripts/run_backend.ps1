$ErrorActionPreference = "Stop"

Set-Location "R:\mlcv-labs\workspace\final-demo"
. .\scripts\setup_powershell_utf8.ps1

conda run -n cv-train python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
