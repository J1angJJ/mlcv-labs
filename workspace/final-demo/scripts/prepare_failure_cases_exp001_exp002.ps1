$ErrorActionPreference = "Stop"

Set-Location "R:\mlcv-labs\workspace\final-demo"
. .\scripts\setup_powershell_utf8.ps1

conda run -n cv-train python scripts\prepare_failure_cases.py `
  --dataset-root data\Seabirds.v6i.yolo26 `
  --split test `
  --exp001-errors outputs\evaluation\exp001_yolo26n_baseline\test\per_image_errors.csv `
  --exp002-errors outputs\evaluation\exp002_yolo11n_baseline\test\per_image_errors.csv `
  --exp001-visuals outputs\predictions\exp001_yolo26n_baseline\test\visuals `
  --exp002-visuals outputs\predictions\exp002_yolo11n_baseline\test\visuals `
  --output-dir outputs\failure_cases\exp001_vs_exp002 `
  --top-k 8
