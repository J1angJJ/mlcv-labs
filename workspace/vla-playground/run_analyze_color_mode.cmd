@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\analyze_color_mode.py ^
  --manifest experiments\manifests\seabirds_annotation_300_labeled.csv ^
  --output-manifest experiments\manifests\seabirds_annotation_300_labeled_color.csv ^
  --probe-dir outputs\clip_labeled_300_probe ^
  --output-dir outputs\color_mode_analysis ^
  --method linear

endlocal
