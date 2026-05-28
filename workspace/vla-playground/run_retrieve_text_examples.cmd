@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%PROJECT_ROOT%\models\hf_home"
set "TORCH_HOME=%PROJECT_ROOT%\models\torch_home"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_retrieval_playground.py ^
  --text "a black and white image of seabirds" ^
  --name text_black_white_seabirds ^
  --top-k 12

"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_retrieval_playground.py ^
  --text "a close photo of puffins on grass" ^
  --name text_close_puffins_grass ^
  --top-k 12

"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_retrieval_playground.py ^
  --text "a distant hard image with tiny birds on a rocky cliff" ^
  --name text_distant_hard_tiny_birds ^
  --top-k 12

endlocal
