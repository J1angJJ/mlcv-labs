# VLA Playground 实验日志

本文档记录本机 Windows 环境下 ViT、CLIP 和 VLA 学习实验的全过程。它是内部记录，不是对外说明文档。

## 1. 工作区初始化

日期：

```text
2026-05-27
```

位置：

```text
R:\mlcv-labs\workspace\vla-playground
```

初始目录：

```text
experiments/
data/
outputs/
README.md
README.internal.md
EXPERIMENT_LOG.md
.gitignore
```

Git 策略：

```text
data/、outputs/、models/、weights/、checkpoints/ 均忽略。
大模型权重和导出产物不进入 Git。
```

## 2. 环境准备

使用环境：

```text
conda env: cv-train
Python: 3.11
PyTorch: 2.12.0+cu126
GPU: RTX 4060 Laptop
```

新增依赖：

```text
timm==1.0.27
open-clip-torch==3.3.0
transformers==5.9.0
accelerate==1.13.0
einops==0.8.2
```

安装验证：

```text
torch cuda: True
pip check: No broken requirements found.
```

说明：

```text
本次只安装 Python 包，没有提前下载 CLIP/ViT/VLM 权重。
用户要求暂时不更新 R:\ai-context 环境快照，待本阶段探索完成后再统一更新。
```

## 3. 初始学习规划

当前判断：

```text
本机适合做 CLIP/ViT 推理、embedding、zero-shot、图像检索、t-SNE/PCA 可视化和轻量线性探针。
暂不适合从零训练 ViT、全量微调 CLIP 或直接训练大型 VLA 模型。
```

推荐路线：

1. CLIP zero-shot 图文相似度。
2. CLIP / ViT image embedding 提取。
3. 图像最近邻检索。
4. PCA / t-SNE 可视化 embedding 空间。
5. 冻结 encoder 做 KNN 或 linear probe。
6. 定义最小 image + instruction + action toy 数据格式。

第一阶段拟做：

```text
experiments/clip_zero_shot.py
```

计划输入：

```text
workspace/final-demo/report_assets/browser_screenshots/
```

计划输出：

```text
outputs/clip_zero_shot/scores.csv
outputs/clip_zero_shot/top_matches.csv
```

拟使用 prompt：

```text
a photo of a puffin
a photo of seabirds on a cliff
a drone photo of seabirds
a close-up photo of a puffin
a photo of rocks
```

本阶段目标：

```text
先验证 CLIP 是否能把 final-demo 中的成功样例、无人机远景、近景重叠和单目标图像映射到合理的文本描述。
不做训练，不做大规模数据下载。
```

## 4. 第一阶段脚本准备

日期：

```text
2026-05-27
```

新增文件：

```text
experiments/clip_zero_shot.py
experiments/README.md
```

脚本设计：

```text
默认读取 final-demo 的 report_assets/browser_screenshots/，不迁移图片。
默认使用 open_clip ViT-B-32 / openai。
输出 scores.csv 和 top_matches.csv。
本步骤只写脚本，不运行模型，不下载权重。
```

模型缓存说明：

```text
模型权重首次运行时会下载到本机缓存目录，而不是 Conda 环境目录。
为避免占用 C 盘，clip_zero_shot.py 已默认设置 HF_HOME 和 TORCH_HOME 到 workspace/vla-playground/models/ 下。
models/ 已被 .gitignore 忽略。
默认 ViT-B-32 / openai 权重预计为几百 MB 级别，低于 10GB；若后续模型预计超过 60GB，应停止下载并重新评估。
```
