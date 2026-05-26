$ErrorActionPreference = "Stop"

Set-Location "R:\mlcv-labs\workspace\final-demo"
. .\scripts\setup_powershell_utf8.ps1

conda run -n cv-train python scripts\evaluate_counting.py `
  --ground-truth outputs\dataset_audit_after_fix\image_counts.csv `
  --predictions outputs\predictions\exp001_yolo26n_baseline\test\test_predictions.csv `
  --split test `
  --class-name puffin `
  --output-dir outputs\evaluation\exp001_yolo26n_baseline\test
