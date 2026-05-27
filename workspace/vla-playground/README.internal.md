# VLA Playground：内部学习指南

这个目录用于本机 Windows 环境下的 ViT、CLIP 和 VLA 相关学习实验。当前阶段先不追求训练完整 VLA 模型，而是从视觉表征和视觉-语言对齐开始，逐步理解 VLA 系统的基础组件。

对外 `README.md` 暂时不更新，后续形成稳定实验后再整理。

## 当前环境

```text
Conda env: cv-train
Python: 3.11
GPU: NVIDIA RTX 4060 Laptop GPU, 8GB VRAM
PyTorch: CUDA-enabled
```

已安装的探索包：

```text
timm
open_clip_torch
transformers
accelerate
einops
```

说明：

- 本轮只安装库，没有提前下载 CLIP/ViT 模型权重。
- 第一次实际加载模型时，权重会下载到本机缓存。
- 暂停更新 `R:\ai-context` 环境快照，等本阶段探索稳定后再统一更新。

## 学习目标

本工作区的目标不是立即训练机器人策略，而是建立 VLA 学习所需的底层直觉：

1. 理解 ViT/CLIP 如何把图像编码为 embedding。
2. 理解图像 embedding 与文本 prompt 的语义对齐方式。
3. 学会做 zero-shot 分类、图像检索和 embedding 可视化。
4. 理解 vision encoder 在 VLA 系统中的位置。
5. 为后续 image + instruction + action 数据格式打基础。

## 为什么从 CLIP / ViT 开始

VLA 通常不是从原始像素直接预测动作，而是依赖视觉编码器提取图像表征，再结合语言指令和动作头。CLIP 和 ViT 是理解这条链路的合适入口：

- ViT 适合理解 patch embedding、self-attention、CLS token 和全局图像表征。
- CLIP 适合理解 image-text contrastive learning、zero-shot 分类和语义检索。
- open_clip / transformers / timm 都能在本机做轻量推理和 embedding 实验。
- 这些实验不需要大数据集，也不需要长时间训练，适合 RTX 4060 Laptop 本机环境。

## 不建议一开始做什么

暂不建议：

- 从零训练 ViT。
- 微调整个 CLIP 大模型。
- 直接跑大型 VLA / VLM，例如完整 LLaVA、OpenVLA 或机器人策略模型。
- 下载大量数据集或多套大模型权重。

原因：

- 8GB 显存适合推理、embedding、轻量线性探针和小规模可视化，不适合大模型全量训练。
- VLA 的学习曲线较长，先弄清视觉表征和图文对齐更稳。
- 大模型权重和数据集容易快速占用几十 GB，需要先规划缓存与目标。

## 推荐实验路线

### 阶段 1：CLIP zero-shot 与图文相似度

目标：

- 加载一个小型 CLIP 模型。
- 输入若干本地图像。
- 给定多个文本 prompt，计算 image-text similarity。
- 输出每张图最匹配的文本描述。

建议模型：

```text
open_clip: ViT-B-32 或 RN50
pretrained: openai 或 laion2b_s34b_b79k
```

建议输入：

- 先用 `workspace/final-demo/report_assets/browser_screenshots/` 的几张图片。
- 后续再放少量自选图片到 `data/images/`。

预期产物：

```text
outputs/clip_zero_shot/scores.csv
outputs/clip_zero_shot/top_matches.csv
```

### 阶段 2：CLIP / ViT 图像 embedding 检索

目标：

- 对一个小图像集合提取 image embeddings。
- 计算 cosine similarity。
- 给定 query image，找最近邻图片。

价值：

- 理解视觉表征空间。
- 观察 CLIP embedding 是否能按语义而不是像素相似度聚类。
- 为后续 VLA 中的视觉状态表示打基础。

预期产物：

```text
outputs/image_retrieval/embeddings.npy
outputs/image_retrieval/nearest_neighbors.csv
outputs/image_retrieval/contact_sheet.png
```

### 阶段 3：embedding 可视化

目标：

- 使用 PCA / t-SNE / UMAP 对图像 embedding 降维。
- 查看 puffin、非 puffin、复杂场景、失败案例在 embedding 空间中的位置。

建议：

- 先用 sklearn PCA / t-SNE。
- UMAP 可后续再装，不急于引入新依赖。

预期产物：

```text
outputs/embedding_viz/pca.png
outputs/embedding_viz/tsne.png
outputs/embedding_viz/metadata.csv
```

### 阶段 4：线性探针或 KNN 分类

目标：

- 冻结 CLIP/ViT encoder。
- 用少量标注训练一个线性分类器或 KNN 分类器。
- 比较 zero-shot、KNN、linear probe 的表现。

价值：

- 理解“冻结视觉 backbone + 小头训练”的迁移学习方式。
- 这和后续 VLA 中冻结/微调 vision encoder 的取舍相关。

### 阶段 5：面向 VLA 的最小数据格式

目标：

定义一个 toy 数据格式：

```json
{
  "image": "path/to/image.jpg",
  "instruction": "count the puffins",
  "observation_text": "a seabird colony on a cliff",
  "action": "return_count"
}
```

这不是训练 VLA，只是提前整理 image + language + action 的结构，为以后接小型 VLM 或 action head 做准备。

## 当前目录约定

```text
experiments/       # 实验脚本
data/              # 本地图片或小数据集，Git 忽略
outputs/           # 实验输出，Git 忽略
README.md          # 对外 README，暂不更新
README.internal.md # 内部学习指南
EXPERIMENT_LOG.md  # 实验日志
```

## 第一阶段建议

下一步建议先做 `clip_zero_shot`：

1. 写 `experiments/clip_zero_shot.py`。
2. 默认读取 `workspace/final-demo/report_assets/browser_screenshots/`。
3. 使用 open_clip 加载 `ViT-B-32`。
4. 准备一组文本 prompt，例如：

```text
a photo of a puffin
a photo of seabirds on a cliff
a drone photo of seabirds
a close-up photo of a puffin
a photo of rocks
```

5. 输出每张图与每个 prompt 的 similarity。
6. 不做训练，只做推理与表格输出。

这个实验小、直观、风险低，也是从课程 final 过渡到 VLA 学习的自然第一步。

## 模型权重与缓存位置

模型权重通常不会下载到 Conda 环境目录，也不是“直接在线加载到内存后消失”。常见流程是：

```text
第一次运行 -> 从模型源下载权重 -> 存到用户缓存目录 -> 之后从本地缓存加载到内存/GPU
```

默认情况下，很多库会使用：

```text
C:\Users\<user>\.cache\
```

但本项目的 `experiments/clip_zero_shot.py` 已在导入 `open_clip` 前设置默认缓存路径，优先写入：

```text
R:\mlcv-labs\workspace\vla-playground\models\hf_home
R:\mlcv-labs\workspace\vla-playground\models\torch_home
```

默认 CLIP `ViT-B-32 / openai` 权重预计为几百 MB 级别。即使后续增加少量 base 级 ViT/CLIP 权重，预计仍小于 10GB；如果某个模型预计超过 60GB，应停止并重新评估，不在本机 C 盘或默认缓存中下载。

具体位置取决于库：

- `open_clip` 可能使用 PyTorch / HuggingFace 相关缓存。
- `transformers` 主要使用 HuggingFace cache。
- `timm` 可能使用 HuggingFace Hub 或 torch hub cache。

如果希望把缓存固定到 R 盘，可以在运行前设置：

```powershell
$env:HF_HOME="R:\mlcv-labs\workspace\vla-playground\models\hf_home"
$env:TORCH_HOME="R:\mlcv-labs\workspace\vla-playground\models\torch_home"
```

当前 `.gitignore` 已忽略：

```text
models/
weights/
checkpoints/
*.pt
*.pth
*.safetensors
*.bin
```

因此后续即使把模型缓存放到 `workspace/vla-playground/models/`，也不会进入 Git。
