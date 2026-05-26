# Puffin Counting Final：项目指南

这个目录是本课程 final 的工作区。README 用于快速说明当前工程状态、数据集、脚本和下一步训练计划；完整过程记录见 `EXPERIMENT_LOG.md`。本 README 和实验记录都不是最终提交报告，后续英文报告会单独编写。

任务书：

```text
Jens Rittscher-2026-05-cv-assignment.docx
```

## 当前状态

已经完成：

- 确定应用场景：上传图片后，在后端完成 puffin 检测与计数。
- 确定部署边界：暂不考虑手机端部署，前端只负责上传和展示，模型推理全部在后端。
- 建立本机训练环境：`cv-train`，Python 3.11，PyTorch CUDA，Ultralytics，FiftyOne。
- 下载并整理主数据集：Roboflow Seabirds v6，YOLO26 格式。
- 完成数据集类别统计、随机画框抽查、零星漏标修复。
- 搭好非模型端工程框架：数据校验、数据划分、计数评估、密度图工具、FastAPI 后端骨架、模型接口占位。
- 新增实验记录文档：`EXPERIMENT_LOG.md`。

尚未完成：

- 第一轮 YOLO26 训练。
- YOLO11 / YOLOv8 baseline 训练。
- test set 计数误差评估。
- 失败案例整理。
- 真实模型推理接入后端。
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

这样做更接近真实海鸟场景，也能降低其他海鸟被误判成 puffin 的风险。

## 数据审计结果

统计脚本：

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

人工抽查发现 valid 中一张图有明显漏标，已使用 `scripts/add_yolo_box.py` 补充右下角一只 puffin 的 box。

修正后 valid 变化：

```text
valid total boxes: 452 -> 453
valid puffin boxes: 96 -> 97
```

修正记录和命令已写入 `EXPERIMENT_LOG.md`。原始标签文件保留了 `.bak` 备份。

## 工程结构

```text
configs/
  default.yaml                  # 项目路径、推理、评估、后端配置

schemas/
  image_counts_template.csv     # 每张图片真实数量标注模板
  points_template.csv           # 点标注模板
  detections_template.csv       # 检测预测结果模板

scripts/
  dataset_class_stats.py        # YOLO 数据集类别统计、label 检查、puffin count 导出
  sample_yolo_boxes.py          # 随机抽样画框，用于标注质量抽查
  add_yolo_box.py               # 给 YOLO label 追加一个像素框，并生成修正预览

backend/
  app.py                        # FastAPI 上传接口骨架

src/puffin_counting/
  annotations.py                # train / val / test 划分
  config.py                     # YAML 配置读取
  dataset.py                    # 图片 manifest、count/point 标注校验
  density.py                    # 点标注密度图工具
  detection.py                  # YOLO 检测计数入口，后续接 YOLO26
  evaluation.py                 # count 误差评估
  io_utils.py                   # 图片扫描和目录工具
  model_interface.py            # 后端与模型之间的接口协议

run_counting.py                 # 通用 CLI 入口
EXPERIMENT_LOG.md               # 实验全过程记录
```

## 常用命令

类别统计：

```powershell
conda run -n cv-train python scripts/dataset_class_stats.py --dataset-root data/Seabirds.v6i.yolo26 --output-dir outputs/dataset_audit --count-class puffin
```

随机抽查 puffin 标注：

```powershell
conda run -n cv-train python scripts/sample_yolo_boxes.py --dataset-root data/Seabirds.v6i.yolo26 --split valid --count 12 --seed 42 --classes puffin --output-dir outputs/label_audit/puffin_valid
```

追加一个 YOLO 标注框：

```powershell
conda run -n cv-train python scripts/add_yolo_box.py --dataset-root data/Seabirds.v6i.yolo26 --split valid --image image_name.jpg --class-name puffin --xyxy X1 Y1 X2 Y2 --output-preview outputs/label_audit/fixes/image_name_preview.jpg
```

生成图片清单：

```powershell
conda run -n cv-train python run_counting.py manifest --image-dir data/Seabirds.v6i.yolo26/valid/images --output outputs/image_manifest.csv
```

评估预测计数：

```powershell
conda run -n cv-train python run_counting.py evaluate --ground-truth outputs/dataset_audit_after_fix/image_counts.csv --predictions outputs/detection/detection_counts.csv --output-dir outputs/evaluation
```

启动后端上传接口前，需要安装后端依赖：

```powershell
conda run -n cv-train python -m pip install fastapi uvicorn python-multipart
conda run -n cv-train python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

当前后端只完成上传接口和模型接口占位，真实 YOLO26 推理还未接入。

## 第一轮训练计划

建议先训练小模型，确保完整流程跑通：

```text
model: yolo26n.pt
data: data/Seabirds.v6i.yolo26/data.yaml
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

如果 RTX 4060 Laptop GPU 显存不足，先把 `batch` 降到 `4`。如果训练稳定且显存余量足，再考虑 `yolo26s.pt` 或更大模型。

第一次训练不要从 `.yaml` 从零训练，应使用预训练权重 `yolo26n.pt` 微调。如果本地没有权重，Ultralytics 会在首次运行时下载。

建议后续 baseline：

```text
yolo11n.pt
yolov8n.pt
```

对比时应尽量保持相同数据集、输入尺寸、epoch、随机种子和评估方式。

## 评估重点

不能只看 mAP。final 任务是 counting，因此还需要关注：

- 每张图 puffin 预测数量。
- count MAE。
- count RMSE。
- bias。
- 漏检、误检、重复检测。
- puffin 与 guillemot / razorbill 等相近类别的混淆。
- 小目标、遮挡、密集场景失败案例。

任务书要求至少 3 张失败案例，训练后应从 test set 或真实预测结果中挑选并保存。

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
```

当前数据集、审计图片、训练输出和模型权重不会进入 Git。

commit 前建议检查：

```powershell
git status --short --untracked-files=all workspace/final-demo
git status --ignored --short workspace/final-demo/data workspace/final-demo/outputs
```

看到以下内容说明数据和输出被正确忽略：

```text
!! workspace/final-demo/data/
!! workspace/final-demo/outputs/
```

## 当前不做

- 暂不做手机端部署。
- 暂不装 ONNX / TensorRT。
- 暂不做模型压缩。
- 暂不写最终英文报告。
- 暂不 commit/push，commit 由人工确认后执行。
