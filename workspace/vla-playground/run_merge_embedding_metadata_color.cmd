@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\merge_embedding_metadata_color.py ^
  --embedding-dir outputs\clip_labeled_300_embeddings ^
  --color-manifest experiments\manifests\seabirds_annotation_300_labeled_color.csv ^
  --output-dir outputs\clip_labeled_300_embeddings_with_color

endlocal
