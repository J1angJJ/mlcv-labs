@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%PROJECT_ROOT%\models\hf_home"
set "TORCH_HOME=%PROJECT_ROOT%\models\torch_home"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\extract_clip_embeddings.py ^
  --manifest experiments\manifests\seabirds_annotation_300_labeled.csv ^
  --output-dir outputs\clip_labeled_300_embeddings ^
  --batch-size 32

endlocal
