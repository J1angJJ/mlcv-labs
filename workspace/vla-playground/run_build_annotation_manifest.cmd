@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\build_annotation_manifest.py ^
  --dataset-root R:\mlcv-labs\workspace\final-demo\data\Seabirds.v6i.yolo26 ^
  --output-csv experiments\manifests\seabirds_annotation_300.csv ^
  --output-html experiments\annotation\annotate_seabirds_300.html ^
  --sample-size 300 ^
  --seed 42

endlocal
