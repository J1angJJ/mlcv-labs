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
  --text "seabirds" ^
  --filter color_mode_auto=grayscale ^
  --name filtered_grayscale_seabirds ^
  --top-k 12 ^
  --index-dir outputs\clip_labeled_300_embeddings_with_color

"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_retrieval_playground.py ^
  --text "puffins on grass" ^
  --filter scene=grass ^
  --name filtered_grass_puffins ^
  --top-k 12 ^
  --index-dir outputs\clip_labeled_300_embeddings_with_color

"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\clip_retrieval_playground.py ^
  --text "tiny birds on a rocky cliff" ^
  --filter distance=far ^
  --filter difficulty=hard ^
  --name filtered_far_hard_rocky_birds ^
  --top-k 12 ^
  --index-dir outputs\clip_labeled_300_embeddings_with_color

endlocal
