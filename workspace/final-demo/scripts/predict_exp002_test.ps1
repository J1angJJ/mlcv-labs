$ErrorActionPreference = "Stop"

Set-Location "R:\mlcv-labs\workspace\final-demo"
. .\scripts\setup_powershell_utf8.ps1

conda run -n cv-train python scripts\predict_yolo_counts.py `
  --model runs\train\exp002_yolo11n_baseline\weights\best.pt `
  --dataset-root data\Seabirds.v6i.yolo26 `
  --split test `
  --class-name puffin `
  --output-dir outputs\predictions\exp002_yolo11n_baseline\test `
  --save-visuals `
  --draw-all-classes
