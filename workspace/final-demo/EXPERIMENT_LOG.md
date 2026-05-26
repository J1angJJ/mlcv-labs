# Puffin Counting Final 实验记录

本文档用于记录 final 项目的全过程，面向自己复盘和后续写英文报告时查证。它不是最终提交报告。

## 1. 项目目标

课程 final 任务要求围绕 puffin counting 设计一个计算机视觉方案，核心问题是：给定包含 puffin 的图片，自动识别并统计图片中的 puffin 数量。

任务书要求覆盖以下方面：

- 说明应用背景和模型方案选择。
- 描述数据集如何收集、划分和标注。
- 给出至少 3 个失败案例，并解释失败原因。
- 说明部署思路。

本项目当前采用检测计数路线：先检测每只 puffin 的 bounding box，再统计检测框数量作为图片中的 puffin count。

## 2. 仓库与工作区初始化

仓库位置：

```text
R:\mlcv-labs
```

final 工作区：

```text
R:\mlcv-labs\workspace\final-demo
```

已建立的项目文件包括：

```text
workspace/final-demo/
  README.md
  EXPERIMENT_LOG.md
  requirements.txt
  run_counting.py
  configs/
  schemas/
  scripts/
  backend/
  src/puffin_counting/
```

Git 管理策略：

- 所有 commit 需要人工确认。
- push 由本人执行。
- 数据集、输出结果、模型权重不进入 Git。

`workspace/final-demo/.gitignore` 已忽略：

```text
data/
outputs/
runs/
weights/
models/
checkpoints/
*.pt
*.pth
*.onnx
*.engine
```

已检查 `data/` 和 `outputs/` 被 Git 正确忽略，因此下载的数据集和生成的审计图片不会撑大 Git 仓库。

## 3. 应用场景与部署边界

当前应用场景设定：

- 用户上传包含海鸟或 puffin 的图片。
- 后端运行模型推理，输出 puffin 检测框和计数结果。
- 前端只负责图片上传和结果展示。

当前不考虑：

- 手机端模型部署。
- TFLite、CoreML 等移动端链路。
- 训练初期的 ONNX / TensorRT 部署优化。

部署思路保留为：

- 本机先完成实验验证。
- 如果后续训练速度或显存不足，再迁移到高性能服务器。
- 如果模型稳定后需要后端推理加速，再考虑 ONNX 或 TensorRT。

## 4. 环境记录

新建了专门用于视觉训练的 Conda 环境：

```text
cv-train
```

Python 版本：

```text
Python 3.11
```

本机硬件：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
显存: 8 GB
```

已验证 PyTorch CUDA 可用：

```text
torch 2.12.0+cu126
torchvision 0.27.0+cu126
ultralytics 8.4.54
CUDA available: True
```

已验证 Ultralytics 能识别并构建：

```text
YOLOv8
YOLO11
YOLO26
```

后续数据审计安装了 FiftyOne：

```text
fiftyone 1.15.0
```

安装后检查结果：

```text
pip check: No broken requirements found
PyTorch CUDA: available
Ultralytics import: OK
```

说明：FiftyOne 安装过程中将 `protobuf` 调整为 `6.33.5`，目前未发现依赖冲突。

## 5. 课程代码参考

已检查课程代码中的相关 demo：

```text
Code/code_10/puffin_counting_demo/
```

该 demo 有参考价值，但不直接照搬。

可复用思路：

- YOLOv8 bird detection/counting 可作为 baseline。
- NMS IoU 阈值比较可用于解释密集目标场景中的漏检、合并或重复检测问题。
- density map demo 可作为点标注计数的辅助方案或报告对照。
- feature map 可视化可作为报告中的模型解释材料。

当前 final 工程已将这些想法拆分为更清晰的模块，包括数据校验、计数评估、检测入口和密度图工具。

## 6. 模型方案选择

当前主线方案：

```text
YOLO26 detection-based counting
```

选择理由：

- 任务天然适合 object detection：每只 puffin 是一个可见目标，计数可以由检测框数量得到。
- Roboflow 数据集已经提供 box 标注。
- YOLO 系列训练和推理链路成熟，适合课程 final 的完整工程实践。
- YOLO26 支持当前 Ultralytics 环境，且可与 YOLOv8、YOLO11 做对比。

保留 baseline：

```text
YOLO11n
YOLOv8n
```

当前不优先采用从零训练或复杂模型压缩。数据量不算大，应优先使用预训练权重微调。

第一轮训练建议：

```text
model: yolo26n.pt
imgsz: 640
epochs: 50
batch: 8
optimizer: auto
lr0: 0.01
lrf: 0.01
patience: 15
seed: 42
device: 0
```

如果显存不足，优先将 `batch` 降到 `4`。如果 baseline 跑通且显存余量充足，再考虑更大的模型。

## 7. 数据集选择

由于没有条件自行拍摄 puffin 图片，本项目采用开源数据集。

当前数据集：

```text
Roboflow Universe - SeabirdAI / Seabirds
Version: 6
Format: YOLO26
License: CC BY 4.0
Local path: R:\mlcv-labs\workspace\final-demo\data\Seabirds.v6i.yolo26
```

数据集页面：

```text
https://universe.roboflow.com/seabirdai-hv30y/seabirds
```

本地 `data.yaml` 类别：

```text
0: fulmar
1: gannet
2: guillemot
3: kittiwake
4: puffin
5: razorbill
6: shag
```

当前决策：

- 保留多类检测训练。
- 推理和计数时只统计 `puffin` 类。

理由：

- 多类训练更接近真实海鸟场景。
- 可以降低把其他海鸟误判为 puffin 的风险。
- 后续报告可以分析 puffin 与相近类别之间的混淆。

## 8. 数据集统计

编写了类别统计脚本：

```text
scripts/dataset_class_stats.py
```

运行命令：

```powershell
conda run -n cv-train python scripts/dataset_class_stats.py --dataset-root data/Seabirds.v6i.yolo26 --output-dir outputs/dataset_audit --count-class puffin
```

初次统计结果：

```text
train: images=1465, labels=1465, boxes=4746
  puffin boxes=912, puffin images=249

valid: images=145, labels=145, boxes=452
  puffin boxes=96, puffin images=20

test: images=84, labels=84, boxes=244
  puffin boxes=38, puffin images=17
```

输出文件：

```text
outputs/dataset_audit/
  split_stats.csv
  class_stats.csv
  image_counts.csv
  invalid_labels.csv
  summary.json
```

统计脚本同时检查：

- 缺失 label 文件。
- 空 label 文件。
- 孤立 label 文件。
- class id 越界。
- YOLO 坐标异常。

初次统计中没有发现 invalid label 行。

## 9. 标注抽查与修正

编写了随机画框抽查脚本：

```text
scripts/sample_yolo_boxes.py
```

抽查 valid split 中的 puffin 标注：

```powershell
conda run -n cv-train python scripts/sample_yolo_boxes.py --dataset-root data/Seabirds.v6i.yolo26 --split valid --count 12 --seed 42 --classes puffin --output-dir outputs/label_audit/puffin_valid
```

输出：

```text
outputs/label_audit/puffin_valid/
  *_boxed.jpg
  sample_manifest.csv
```

人工检查发现一张图存在明显漏标：

```text
split: valid
image: macareux-groupe-rocher_jpeg.rf.ef2573c6c9fd43c558800a824c34be52.jpg
issue: 右下角一只 puffin 未标注
```

为处理零星漏标，编写了补框脚本：

```text
scripts/add_yolo_box.py
```

补标命令：

```powershell
conda run -n cv-train python scripts/add_yolo_box.py --dataset-root data/Seabirds.v6i.yolo26 --split valid --image macareux-groupe-rocher_jpeg.rf.ef2573c6c9fd43c558800a824c34be52.jpg --class-name puffin --xyxy 499 414 617 611 --output-preview outputs/label_audit/fixes/macareux-groupe-rocher_fixed_preview.jpg
```

追加的 YOLO 标签：

```text
4 0.87187500 0.80078125 0.18437500 0.30781250
```

脚本自动保留原始标签备份：

```text
data/Seabirds.v6i.yolo26/valid/labels/macareux-groupe-rocher_jpeg.rf.ef2573c6c9fd43c558800a824c34be52.txt.bak
```

修正后重新统计：

```powershell
conda run -n cv-train python scripts/dataset_class_stats.py --dataset-root data/Seabirds.v6i.yolo26 --output-dir outputs/dataset_audit_after_fix --count-class puffin
```

修正后变化：

```text
valid total boxes: 452 -> 453
valid puffin boxes: 96 -> 97
```

当前判断：网页端和本地抽查未发现更多明显漏标，可以进入第一轮训练。

## 10. 已实现工程脚本

数据统计：

```text
scripts/dataset_class_stats.py
```

随机抽样画框：

```text
scripts/sample_yolo_boxes.py
```

单框补标：

```text
scripts/add_yolo_box.py
```

通用 CLI：

```text
run_counting.py
```

当前支持：

```text
manifest
split
validate-data
evaluate
density
density-demo
detect
```

后端骨架：

```text
backend/app.py
```

当前后端只完成上传接口和模型接口占位，尚未接入真实 YOLO26 推理。

## 11. 下一步计划

## 11. 实验 exp001_yolo26n_baseline

日期：

```text
2026-05-26
```

模型：

```text
yolo26n.pt
```

数据配置：

```text
configs/seabirds_v6_local.yaml
```

训练入口：

```text
scripts/train_yolo26n_baseline.ps1
```

第一次运行时使用了 Ultralytics CLI：

```powershell
conda run -n cv-train yolo detect train model=yolo26n.pt data=configs/seabirds_v6_local.yaml project=runs/train name=exp001_yolo26n_baseline
```

这次训练成功完成，但输出路径被 Ultralytics CLI 拼成：

```text
runs/detect/runs/train/exp001_yolo26n_baseline
```

训练完成后已整理到：

```text
runs/train/exp001_yolo26n_baseline
```

随后训练脚本已改为调用：

```text
scripts/train_yolo.py
```

这样后续输出路径由 Python API 控制，避免再次出现 `runs/detect/runs/train` 嵌套。

实际训练参数来自 Ultralytics 默认值：

```text
epochs: 100
batch: 16
imgsz: 640
optimizer: auto
seed: 0
pretrained: true
amp: true
```

训练耗时：

```text
2885.27 s，约 48.1 min
```

最终 epoch 指标：

```text
precision(B): 0.82642
recall(B): 0.81357
mAP50(B): 0.83341
mAP50-95(B): 0.62897
```

最高 mAP50 出现在 epoch 50：

```text
mAP50(B): 0.83648
mAP50-95(B): 0.59935
```

关键输出：

```text
runs/train/exp001_yolo26n_baseline/weights/best.pt
runs/train/exp001_yolo26n_baseline/weights/last.pt
runs/train/exp001_yolo26n_baseline/results.csv
runs/train/exp001_yolo26n_baseline/results.png
runs/train/exp001_yolo26n_baseline/confusion_matrix.png
runs/train/exp001_yolo26n_baseline/confusion_matrix_normalized.png
runs/train/exp001_yolo26n_baseline/BoxPR_curve.png
runs/train/exp001_yolo26n_baseline/BoxF1_curve.png
```

观察：

- 本机 RTX 4060 Laptop GPU 可以使用默认 `batch=16` 完成 YOLO26n 训练。
- TensorBoard 未生成 event 文件，因此当前训练过程可视化主要依赖 `results.csv` 和 `results.png`。
- 第一轮检测指标已经可用，但 final 任务还需要进一步计算 puffin counting 的 MAE、RMSE 和 bias。

下一步：

1. 在 valid/test split 上用 `best.pt` 推理并导出 puffin 预测数量。
2. 计算 count MAE、RMSE、bias。
3. 查看 confusion matrix 和预测可视化，整理失败案例。
4. 训练 YOLO11n 或 YOLOv8n baseline。

## 12. 下一步计划

第一阶段训练：

1. 使用 `yolo26n.pt` 进行预训练权重微调。
2. 使用 Roboflow Seabirds v6 原始 split。
3. 先跑 `imgsz=640`、`batch=8`、`epochs=50`。
4. 如果显存不足，降到 `batch=4`。
5. 保存训练命令、日志目录、best.pt、last.pt、metrics 和预测可视化。

第二阶段 baseline：

1. 使用 `yolo11n.pt` 或 `yolov8n.pt` 训练相同数据。
2. 保持数据划分和主要训练参数一致。
3. 对比 mAP 与计数误差。

第三阶段评估：

1. 在 test split 上生成预测。
2. 统计每张图 puffin 预测数量。
3. 与 ground truth count 对比。
4. 计算 MAE、RMSE、bias。
5. 整理至少 3 张失败案例。

第四阶段部署：

1. 将 best model 接入 `model_interface.py`。
2. 让 FastAPI 后端支持图片上传和返回检测计数。
3. 如有必要，再考虑 ONNX 或 TensorRT。

## 13. 待记录内容

后续每次训练都需要追加一节，至少记录：

```text
日期:
实验编号:
模型:
预训练权重:
数据集版本:
训练命令:
关键超参:
输出目录:
best.pt 路径:
valid 指标:
test 指标:
count MAE/RMSE/bias:
主要观察:
失败案例:
下一步决策:
```

建议实验编号格式：

```text
exp001_yolo26n_baseline
exp002_yolo11n_baseline
exp003_yolov8n_baseline
```

## 14. 实验 exp001 test 推理与 counting 评估

日期：

```text
2026-05-26
```

模型权重：

```text
runs/train/exp001_yolo26n_baseline/weights/best.pt
```

推理脚本：

```text
scripts/predict_yolo_counts.py
scripts/predict_exp001_test.ps1
```

评估脚本：

```text
scripts/evaluate_counting.py
scripts/evaluate_exp001_test.ps1
```

推理命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\predict_exp001_test.ps1
```

评估命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\evaluate_exp001_test.ps1
```

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

误差最大的样本：

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

观察：

- 84 张 test 图中，80 张计数完全正确。
- 4 张存在计数误差。
- `bias` 为负，说明主要失败模式是漏检。
- 下一步应查看这 4 张图的 `*_pred.jpg`，从中选择至少 3 张作为报告失败案例。

## 15. 实验 exp002_yolo11n_baseline

日期：

```text
2026-05-26
```

模型：

```text
yolo11n.pt
```

数据配置：

```text
configs/seabirds_v6_local.yaml
```

训练入口：

```text
scripts/train_yolo11n_baseline.ps1
```

训练输出最初仍被 Ultralytics 写入：

```text
runs/detect/runs/train/exp002_yolo11n_baseline
```

原因分析：

```text
Ultralytics detect task 会以 runs/detect 作为默认 save root。
传入相对 project=runs/train 时，内部会在默认 root 下再次拼接该相对路径，
因此得到 runs/detect/runs/train/exp002_yolo11n_baseline。
```

修正：

```text
已将 exp002 输出移动到 runs/train/exp002_yolo11n_baseline。
已修改 scripts/train_yolo.py，使 data 和 project 都解析为项目根目录下的绝对路径。
后续训练应直接输出到 runs/train/<experiment_name>。
```

实际训练参数来自 Ultralytics 默认值：

```text
epochs: 100
batch: 16
imgsz: 640
optimizer: auto
seed: 0
pretrained: true
amp: true
```

训练耗时：

```text
2279.97 s，约 38.0 min
```

最终 epoch 指标：

```text
precision(B): 0.87038
recall(B): 0.81385
mAP50(B): 0.85236
mAP50-95(B): 0.62863
```

最高 mAP50 出现在 epoch 95：

```text
mAP50(B): 0.85929
mAP50-95(B): 0.63022
```

推理输出：

```text
outputs/predictions/exp002_yolo11n_baseline/test/test_predictions.csv
outputs/predictions/exp002_yolo11n_baseline/test/test_detections.csv
outputs/predictions/exp002_yolo11n_baseline/test/visuals/
```

评估输出：

```text
outputs/evaluation/exp002_yolo11n_baseline/test/count_metrics.csv
outputs/evaluation/exp002_yolo11n_baseline/test/per_image_errors.csv
outputs/evaluation/exp002_yolo11n_baseline/test/prediction_vs_ground_truth.png
```

test counting 指标：

```text
num_images: 84
MAE: 0.0952380952
RMSE: 0.4629100499
mean_relative_error: 0.1338235294
bias: 0.0
```

与 exp001 对比：

```text
exp001 YOLO26n: MAE=0.11905, RMSE=0.59761, bias=-0.11905
exp002 YOLO11n: MAE=0.09524, RMSE=0.46291, bias=0.0
```

误差最大的样本：

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

观察：

- YOLO11n 的 detection mAP50 和 counting MAE 都略优于 YOLO26n。
- YOLO11n 的 overall bias 为 0，但存在过检和漏检互相抵消。
- YOLO26n 更偏向漏检。
- 后续失败案例分析应同时查看 exp001 与 exp002 在相同图片上的表现差异。

## 16. 失败案例素材整理

日期：

```text
2026-05-26
```

脚本：

```text
scripts/prepare_failure_cases.py
scripts/prepare_failure_cases_exp001_exp002.ps1
```

命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_failure_cases_exp001_exp002.ps1
```

输出目录：

```text
outputs/failure_cases/exp001_vs_exp002/
```

输出文件：

```text
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

当前整理出的失败案例：

```text
case_001_both_under_count
image: DJI_20220726115422_0304_Z_JPG.rf.830e85ff1aeb2fa9c366ad16eaf0caf7.jpg
ground_truth=4, YOLO26n=0, YOLO11n=2

case_002_under_to_over
image: DJI_20220726115400_0293_Z_JPG.rf.1a9deb01ac219b8835aa9f89ddf90007.jpg
ground_truth=8, YOLO26n=5, YOLO11n=11

case_003_both_under_count
image: 1127_maine-puffins-1000x644_jpeg.rf.21367ecf0d8ee509f74ef00728fab9a3.jpg
ground_truth=5, YOLO26n=3, YOLO11n=3

case_004_exp002_new_error
image: AtlanticPuffin10_jpeg.rf.6d3aa2d97c83dde15afde529752e5020.jpg
ground_truth=1, YOLO26n=1, YOLO11n=2

case_005_exp002_fixed_exp001_error
image: Screenshot-2023-04-10-at-8-48-07-PM_png.rf.4a6fca1fab16bb899af2a706ce16de78.jpg
ground_truth=4, YOLO26n=3, YOLO11n=4
```

下一步：

```text
人工查看每个 case 的 original 和两个模型预测图，在 notes.md 中补充中文观察。
后续英文报告从这些 notes 中选择至少 3 个失败案例。
```

## 17. 人工失败案例观察

日期：

```text
2026-05-26
```

观察摘要：

```text
case_001 和 case_002 都是无人机超远景航拍。
目标非常小，人眼也难以分辨。
在这类图中，能检测到更多疑似目标的模型更接近真实计数，但也更容易过检。

case_003 和 case_005 都是近景重叠场景。
case_003 中两个模型都漏掉了重叠较高的部分目标。
case_005 中 YOLO11n 全部检出，而 YOLO26n 漏检一个目标。

case_004 是单目标近景特写。
YOLO26n 正确检出一个目标，YOLO11n 多检一个目标。
```

初步结论：

```text
1. YOLO26n 在当前实验中整体更保守，主要错误是漏检。
2. YOLO11n 在 test counting 指标上更好，但更敏感，可能在单目标近景中产生过检。
3. 最主要的失败模式有两类：
   - 无人机超远景小目标漏检或计数不稳定。
   - 近景重叠个体被合并或漏检。
4. 报告中可选择 case_001、case_002、case_003 作为主要失败案例，case_004/case_005 用于对比模型灵敏度差异。
```

## 18. 后端推理原型

日期：

```text
2026-05-26
```

目标：

```text
将最终候选模型 YOLO11n best.pt 接入 FastAPI 后端。
前端只上传图片，后端完成模型推理并返回 puffin count 和 boxes。
```

参考课程 demo：

```text
Code/code_10/puffin_counting_demo/main.py
```

可借鉴部分：

```text
按目标类别筛选 YOLO detection boxes。
统计目标类别数量。
保存带框可视化图片。
```

未直接复用部分：

```text
课程 demo 没有后端服务结构，因此 FastAPI 上传、配置加载、响应 JSON 和 predictor 接口均在本项目中重新实现。
```

新增/修改文件：

```text
backend/app.py
src/puffin_counting/model_interface.py
src/puffin_counting/yolo_predictor.py
scripts/run_backend.ps1
scripts/smoke_test_backend.py
scripts/smoke_test_backend.ps1
configs/default.yaml
```

默认后端模型：

```text
runs/train/exp002_yolo11n_baseline/weights/best.pt
```

接口：

```text
GET  /health
GET  /model
POST /predict
```

`POST /predict` 返回：

```text
count
boxes
mean_confidence
all_detections
elapsed_ms
figure_path
model_path
target_class
```

smoke test 命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_test_backend.ps1
```

smoke test 结果：

```text
health: {'status': 'ok', 'model_loaded': True, 'target_class': 'puffin'}
sample image: 1127_maine-puffins-1000x644_jpeg.rf.21367ecf0d8ee509f74ef00728fab9a3.jpg
count: 3
all_detections: 3
elapsed_ms: 693.68
```

依赖变更：

```text
Installed fastapi, uvicorn, python-multipart in cv-train.
pip check: No broken requirements found.
```

环境快照：

```text
本轮按用户此前要求暂不更新 R:\ai-context 环境快照，后续由用户统一处理。
```

## 19. 前端上传原型

日期：

```text
2026-05-26
```

目标：

```text
实现一个非重点的轻量前端，用于展示 deployment prototype。
前端只负责上传图片和展示结果，不做任何模型推理。
```

新增文件：

```text
frontend/index.html
frontend/styles.css
frontend/app.js
```

后端路由更新：

```text
GET /                 返回前端页面
GET /static/app.js    前端脚本
GET /static/styles.css
GET /prediction-files/<file> 访问后端预测可视化图片
```

前端功能：

```text
选择本地图片。
调用 POST /predict。
显示输入预览。
显示后端生成的预测图。
显示 puffin count、mean confidence、elapsed time、total detections。
显示 puffin bounding boxes 表格。
```

验证：

```text
GET / -> 200
GET /static/app.js -> 200
GET /static/styles.css -> 200
GET /health -> 200
GET /model -> 200
POST /predict smoke test -> count=3
```

使用方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
```

然后打开：

```text
http://127.0.0.1:8000/
```

## 20. 浏览器截图与报告素材整理

日期：

```text
2026-05-26
```

来源：

```text
R:\mlcv-labs\workspace\final-demo\temp
```

整理后位置：

```text
report_assets/browser_screenshots/
```

文件映射：

```text
127.0.0.1_8000_ (1).png -> 01_success_complex_23_targets.png
127.0.0.1_8000_ (2).png -> 02_failure_case_001_drone_far_both_under_count.png
127.0.0.1_8000_ (3).png -> 03_failure_case_002_drone_far_under_to_over.png
127.0.0.1_8000_ (4).png -> 04_failure_case_003_near_overlap_both_under_count.png
127.0.0.1_8000_ (5).png -> 05_failure_case_004_single_closeup_yolo11n_over_count.png
127.0.0.1_8000_ (6).png -> 06_case_005_near_overlap_yolo11n_success.png
```

说明：

```text
01 是复杂场景成功样例，23 个目标全部正确识别。
02-06 对应此前整理的 case_001 到 case_005。
06 对应 case_005，是成功示例，因为最终采用的 YOLO11n 模型全部检出。
```

仓库整理：

```text
temp/ 已删除并加入 .gitignore。
Ultralytics 自动下载到项目根目录的 yolo11n.pt 和 yolo26n.pt 已移动到 ignored 的 weights/pretrained/。
```
