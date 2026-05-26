$ErrorActionPreference = "Stop"

Set-Location "R:\mlcv-labs\workspace\final-demo"
. .\scripts\setup_powershell_utf8.ps1

conda run -n cv-train python scripts\train_yolo.py `
  --model yolo26n.pt `
  --data configs\seabirds_v6_local.yaml `
  --project runs\train `
  --name exp001_yolo26n_baseline
