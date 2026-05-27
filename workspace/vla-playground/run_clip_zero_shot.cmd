@echo off
setlocal

chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set PROJECT_ROOT=%~dp0
set HF_HOME=%PROJECT_ROOT%models\hf_home
set TORCH_HOME=%PROJECT_ROOT%models\torch_home

cd /d "%PROJECT_ROOT%"

echo Running CLIP zero-shot experiment...
echo HF_HOME=%HF_HOME%
echo TORCH_HOME=%TORCH_HOME%
echo.

"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_zero_shot.py

echo.
echo Done. Outputs should be in outputs\clip_zero_shot\
endlocal
