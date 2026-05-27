@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\visualize_clip_embeddings.py ^
  --input-dir outputs\clip_image_retrieval ^
  --output-dir outputs\clip_embedding_plot ^
  --method pca

endlocal
