# Puffin Counting Final：内部项目指南

这个目录是课程 final 的工作区。README 用于快速说明当前工程状态、数据集、脚本、训练结果和下一步；完整过程记录见 `EXPERIMENT_LOG.md`。README 和实验记录都不是最终提交报告，后续英文报告会单独编写。

任务书：

```text
Jens Rittscher-2026-05-cv-assignment.docx
```

## 当前状态

已经完成：

- 应用场景：上传图片后，在后端完成 puffin 检测与计数。
- 部署边界：暂不考虑手机端部署，前端只负责上传和展示，模型推理全部在后端。
- 本机环境：`cv-train`，Python 3.11，PyTorch CUDA，Ultralytics，FiftyOne。
- 数据集：Roboflow Seabirds v6，YOLO26 格式。
- 数据审计：类别统计、随机画框抽查、零星漏标修复。
- 第一轮训练：YOLO26n baseline 已完成。
- 第一轮 test 推理：已用 `best.pt` 对 test split 推理。
- 第一轮 counting 评估：已得到 MAE、RMSE、bias。
- 第二轮 baseline：YOLO11n 已训练、推理并完成 test counting 评估。
- 工程脚本：训练、推理、计数评估、标注抽查、补标、PowerShell UTF-8 设置。
- 后端原型：FastAPI `/health`、`/model`、`/predict` 已接入 YOLO11n best 权重并通过 smoke test。
- 前端原型：单页上传图片、显示 count、预测图和检测框列表。

尚未完成：

- YOLOv8 baseline 训练。
- 失败案例详细分析与报告素材整理。
- 最终英文报告。

## 数据集

当前主数据集：

```text
Roboflow Universe - SeabirdAI / Seabirds
Version: 6
Format: YOLO26
License: CC BY 4.0
Local path: data/Seabirds.v6i.yolo26
```

数据集页面：

```text
https://universe.roboflow.com/seabirdai-hv30y/seabirds
```

类别：

```text
0: fulmar
1: gannet
2: guillemot
3: kittiwake
4: puffin
5: razorbill
6: shag
```

当前策略：

- 保留 7 类训练。
- 推理和计数时只统计 `puffin`。
- 使用原始 train / valid / test split。
- 使用预训练 YOLO 权重微调，不从零训练。

## 数据审计结果

统计脚本：

```text
scripts/dataset_class_stats.py
```

命令：

```powershell
conda run -n cv-train python scripts/dataset_class_stats.py --dataset-root data/Seabirds.v6i.yolo26 --output-dir outputs/dataset_audit --count-class puffin
```

初次统计：

```text
train: images=1465, labels=1465, boxes=4746
  puffin boxes=912, puffin images=249

valid: images=145, labels=145, boxes=452
  puffin boxes=96, puffin images=20

test: images=84, labels=84, boxes=244
  puffin boxes=38, puffin images=17
```

人工抽查发现 valid 中一张图存在明显漏标，已使用 `scripts/add_yolo_box.py` 补充右下角一只 puffin 的 box。修正后：

```text
valid total boxes: 452 -> 453
valid puffin boxes: 96 -> 97
```

## 工程结构

```text
configs/
  default.yaml
  seabirds_v6_local.yaml        # 本地 YOLO 数据配置

frontend/
  index.html                    # 简单上传/结果展示页面
  styles.css
  app.js

report_assets/
  browser_screenshots/          # 报告可用浏览器端截图

schemas/
  image_counts_template.csv
  points_template.csv
  detections_template.csv

scripts/
  dataset_class_stats.py        # YOLO 数据集类别统计、label 检查、puffin count 导出
  sample_yolo_boxes.py          # 随机抽样画框，用于标注质量抽查
  add_yolo_box.py               # 给 YOLO label 追加一个像素框，并生成修正预览
  train_yolo.py                 # Ultralytics Python API 训练入口
  train_yolo26n_baseline.ps1    # exp001 训练入口
  predict_yolo_counts.py        # YOLO 推理并导出每张图 puffin count
  predict_exp001_test.ps1       # exp001 test 推理入口
  evaluate_counting.py          # count MAE/RMSE/bias 评估
  evaluate_exp001_test.ps1      # exp001 test counting 评估入口
  tensorboard_runs.ps1          # TensorBoard 入口
  setup_powershell_utf8.ps1     # PowerShell UTF-8 控制台设置
  prepare_failure_cases.py      # 合并模型误差并整理失败案例素材
  prepare_failure_cases_exp001_exp002.ps1
  run_backend.ps1               # 启动 FastAPI 后端
  smoke_test_backend.py         # 使用 FastAPI TestClient 测试 /predict
  smoke_test_backend.ps1

backend/
  app.py                        # FastAPI 上传与 YOLO 推理接口

src/puffin_counting/
  annotations.py
  config.py
  dataset.py
  density.py
  detection.py
  evaluation.py
  io_utils.py
  model_interface.py

run_counting.py
EXPERIMENT_LOG.md
```

## 常用命令

训练 YOLO26n baseline：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_yolo26n_baseline.ps1
```

test split 推理：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\predict_exp001_test.ps1
```

test counting 评估：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\evaluate_exp001_test.ps1
```

启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
```

后端 smoke test：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_test_backend.ps1
```

后端接口：

```text
GET  /health
GET  /model
POST /predict
```

启动后打开：

```text
http://127.0.0.1:8000/
```

随机抽查 puffin 标注：

```powershell
conda run -n cv-train python scripts/sample_yolo_boxes.py --dataset-root data/Seabirds.v6i.yolo26 --split valid --count 12 --seed 42 --classes puffin --output-dir outputs/label_audit/puffin_valid
```

追加一个 YOLO 标注框：

```powershell
conda run -n cv-train python scripts/add_yolo_box.py --dataset-root data/Seabirds.v6i.yolo26 --split valid --image image_name.jpg --class-name puffin --xyxy X1 Y1 X2 Y2 --output-preview outputs/label_audit/fixes/image_name_preview.jpg
```

## exp001 训练结果

实验：

```text
exp001_yolo26n_baseline
```

模型：

```text
yolo26n.pt
```

结果目录：

```text
runs/train/exp001_yolo26n_baseline/
```

关键输出：

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png
BoxF1_curve.png
val_batch*_pred.jpg
```

最终 epoch 指标：

```text
epoch: 100
time: 2885.27 s
precision(B): 0.82642
recall(B): 0.81357
mAP50(B): 0.83341
mAP50-95(B): 0.62897
```

说明：

- TensorBoard 未生成 event 文件，训练过程可视化主要看 `results.csv` 和 `results.png`。
- 本机 RTX 4060 Laptop GPU 可以使用默认 `batch=16` 完成 YOLO26n 训练。
- 训练输出 `runs/` 被 Git 忽略。

## exp001 Counting 评估

推理输出：

```text
outputs/predictions/exp001_yolo26n_baseline/test/test_predictions.csv
outputs/predictions/exp001_yolo26n_baseline/test/test_detections.csv
outputs/predictions/exp001_yolo26n_baseline/test/visuals/
```

评估输出：

```text
outputs/evaluation/exp001_yolo26n_baseline/test/count_metrics.csv
outputs/evaluation/exp001_yolo26n_baseline/test/per_image_errors.csv
outputs/evaluation/exp001_yolo26n_baseline/test/prediction_vs_ground_truth.png
```

test counting 指标：

```text
num_images: 84
MAE: 0.1190476190
RMSE: 0.5976143047
mean_relative_error: 0.1191176471
bias: -0.1190476190
```

当前判断：

- 84 张 test 图中，80 张计数完全正确。
- 4 张存在计数误差。
- `bias` 为负，主要问题是漏检而不是误检。

失败案例候选：

```text
DJI_20220726115422_0304_Z_JPG.rf.830e85ff1aeb2fa9c366ad16eaf0caf7.jpg
ground_truth=4, prediction=0, error=-4

DJI_20220726115400_0293_Z_JPG.rf.1a9deb01ac219b8835aa9f89ddf90007.jpg
ground_truth=8, prediction=5, error=-3

1127_maine-puffins-1000x644_jpeg.rf.21367ecf0d8ee509f74ef00728fab9a3.jpg
ground_truth=5, prediction=3, error=-2

Screenshot-2023-04-10-at-8-48-07-PM_png.rf.4a6fca1fab16bb899af2a706ce16de78.jpg
ground_truth=4, prediction=3, error=-1
```

下一步应优先人工查看这些 `*_pred.jpg`，整理至少 3 个失败案例。

## exp002 YOLO11n Baseline

实验：

```text
exp002_yolo11n_baseline
```

模型：

```text
yolo11n.pt
```

结果目录：

```text
runs/train/exp002_yolo11n_baseline/
```

最终 epoch 指标：

```text
epoch: 100
time: 2279.97 s
precision(B): 0.87038
recall(B): 0.81385
mAP50(B): 0.85236
mAP50-95(B): 0.62863
```

test counting 指标：

```text
num_images: 84
MAE: 0.0952380952
RMSE: 0.4629100499
mean_relative_error: 0.1338235294
bias: 0.0
```

与 exp001 的计数结果对比：

```text
exp001 YOLO26n: MAE=0.11905, RMSE=0.59761, bias=-0.11905
exp002 YOLO11n: MAE=0.09524, RMSE=0.46291, bias=0.0
```

当前结论：

- 在 test counting 指标上，YOLO11n baseline 暂时优于 YOLO26n。
- YOLO26n 主要偏漏检；YOLO11n 的总体 bias 为 0，但同时存在过检和漏检。
- 后续报告不能只声称 YOLO26n 更好，需要基于当前结果如实比较。

exp002 误差最大的样本：

```text
DJI_20220726115400_0293_Z_JPG.rf.1a9deb01ac219b8835aa9f89ddf90007.jpg
ground_truth=8, prediction=11, error=3

1127_maine-puffins-1000x644_jpeg.rf.21367ecf0d8ee509f74ef00728fab9a3.jpg
ground_truth=5, prediction=3, error=-2

DJI_20220726115422_0304_Z_JPG.rf.830e85ff1aeb2fa9c366ad16eaf0caf7.jpg
ground_truth=4, prediction=2, error=-2

AtlanticPuffin10_jpeg.rf.6d3aa2d97c83dde15afde529752e5020.jpg
ground_truth=1, prediction=2, error=1
```

## 后续优先级

1. 人工查看 `outputs/failure_cases/exp001_vs_exp002/` 下的 5 个 case。
2. 在每个 `notes.md` 中补充失败类型、场景特征、可能原因和报告结论。
3. 从中选择至少 3 个作为 final 报告失败案例。
4. 决定是否继续训练 YOLOv8n 作为第二个 baseline。
5. 将 `best.pt` 接入后端模型接口。
6. 撰写英文 final report。

失败案例素材目录：

```text
outputs/failure_cases/exp001_vs_exp002/
  comparison_summary.csv
  case_manifest.csv
  case_001_both_under_count/
  case_002_under_to_over/
  case_003_both_under_count/
  case_004_exp002_new_error/
  case_005_exp002_fixed_exp001_error/
```

每个 case 目录包含：

```text
original.jpg
exp001_yolo26n_pred.jpg
exp002_yolo11n_pred.jpg
notes.md
```

报告截图素材：

```text
report_assets/browser_screenshots/
  01_success_complex_23_targets.png
  02_failure_case_001_drone_far_both_under_count.png
  03_failure_case_002_drone_far_under_to_over.png
  04_failure_case_003_near_overlap_both_under_count.png
  05_failure_case_004_single_closeup_yolo11n_over_count.png
  06_case_005_near_overlap_yolo11n_success.png
```

暂不做：

- 手机端部署。
- ONNX / TensorRT。
- 模型压缩。

## 后端原型

默认模型：

```text
runs/train/exp002_yolo11n_baseline/weights/best.pt
```

配置位置：

```text
configs/default.yaml
```

`POST /predict` 接收一张图片，返回：

```text
upload_id
file_name
stored_path
model_path
target_class
count
mean_confidence
all_detections
conf_threshold
iou_threshold
elapsed_ms
figure_path
boxes[]
warnings[]
```

smoke test 结果：

```text
health: ok
model_loaded: True
target_class: puffin
sample count: 3
sample all_detections: 3
```

说明：

- 后端推理全部在服务器/本机端完成，前端只需要上传图片并展示 JSON/可视化结果。
- 上传图片保存在 `data/uploads/`，预测可视化保存在 `outputs/backend_predictions/`。
- `data/` 和 `outputs/` 都被 Git 忽略。

## 前端原型

位置：

```text
frontend/
```

功能：

```text
选择图片
上传到 POST /predict
显示本地图预览
显示后端预测可视化图
显示 count、mean confidence、elapsed time、total detections
显示 puffin boxes 表格
```

前端不做模型推理，只是上传图片并展示后端结果，符合当前 deployment plan。

## Git 与大文件保护

`workspace/final-demo/.gitignore` 已忽略：

```text
data/
outputs/
runs/
weights/
models/
checkpoints/
tmp/
temp/
*.pt
*.pth
*.onnx
*.engine
*.tflite
*.mlpackage
```

当前数据集、审计图片、训练输出、推理输出和模型权重都不会进入 Git。

commit 前建议检查：

```powershell
git status --short --untracked-files=all workspace/final-demo
git status --ignored --short workspace/final-demo/data workspace/final-demo/outputs workspace/final-demo/runs
```

看到以下内容说明数据和输出被正确忽略：

```text
!! workspace/final-demo/data/
!! workspace/final-demo/outputs/
!! workspace/final-demo/runs/
```
