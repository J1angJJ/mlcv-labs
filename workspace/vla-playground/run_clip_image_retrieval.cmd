@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%PROJECT_ROOT%\models\hf_home"
set "TORCH_HOME=%PROJECT_ROOT%\models\torch_home"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_image_retrieval.py ^
  --manifest experiments\manifests\final_demo_failure_cases.csv ^
  --output-dir outputs\clip_image_retrieval ^
  --top-k 3

endlocal
