# Puffin Counting Final：项目指南

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
- 工程脚本：训练、推理、计数评估、标注抽查、补标、PowerShell UTF-8 设置。

尚未完成：

- YOLO11 / YOLOv8 baseline 训练。
- 失败案例详细分析与报告素材整理。
- 真实模型推理接入 FastAPI 后端。
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

backend/
  app.py                        # FastAPI 上传接口骨架

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

## 后续优先级

1. 查看 4 张失败案例候选的预测可视化，判断漏检原因。
2. 选择至少 3 张作为 final 报告失败案例。
3. 训练 YOLO11n 或 YOLOv8n baseline，形成模型对比。
4. 将 `best.pt` 接入后端模型接口。
5. 撰写英文 final report。

暂不做：

- 手机端部署。
- ONNX / TensorRT。
- 模型压缩。

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
