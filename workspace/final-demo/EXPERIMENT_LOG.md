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

## 12. 待记录内容

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
