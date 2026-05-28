@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"
if "%~1"=="" (
  echo Usage: run_retrieve_by_image_id.cmd IMAGE_ID
  exit /b 1
)

"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_retrieval_playground.py ^
  --query-id "%~1" ^
  --top-k 12

endlocal
