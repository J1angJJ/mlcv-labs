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

## 5. CLIP zero-shot 首次运行

日期：

```text
2026-05-27
```

命令：

```powershell
cd R:\mlcv-labs\workspace\vla-playground
conda run -n cv-train python experiments\clip_zero_shot.py
```

输入：

```text
R:\mlcv-labs\workspace\final-demo\report_assets\browser_screenshots
```

输出：

```text
outputs/clip_zero_shot/scores.csv
outputs/clip_zero_shot/top_matches.csv
```

运行结果：

```text
Processed 6 images with 5 prompts.
```

top match 汇总：

```text
01_success_complex_23_targets.png                         -> a photo of seabirds on a cliff
02_failure_case_001_drone_far_both_under_count.png        -> a photo of seabirds on a cliff
03_failure_case_002_drone_far_under_to_over.png           -> a photo of seabirds on a cliff
04_failure_case_003_near_overlap_both_under_count.png     -> a photo of seabirds on a cliff
05_failure_case_004_single_closeup_yolo11n_over_count.png -> a photo of a puffin
06_case_005_near_overlap_yolo11n_success.png              -> a photo of seabirds on a cliff
```

初步观察：

```text
CLIP 能把多数浏览器截图识别为 cliff / seabird 场景，而不是简单匹配到 rocks。
单目标近景 close-up 样例被识别为 puffin，符合预期。
当前 prompt 集较粗，"a photo of seabirds on a cliff" 对多数样例具有较强吸引力；下一轮可以增加更具体的 prompt，例如 "a group of puffins on a rocky cliff"、"a distant drone photo of tiny birds"、"a single puffin close-up"。
```

缓存情况：

```text
models/ 当前约 605 MB。
权重缓存已落在 R:\mlcv-labs\workspace\vla-playground\models\ 下，没有进入 Conda 环境目录。
```

warning 说明：

```text
HuggingFace symlink warning:
Windows 当前未启用 symlink/developer mode，缓存会退化为非符号链接模式，功能可用但可能略占更多空间。当前模型规模很小，可以忽略。

QuickGELU warning:
open_clip 提示模型配置与 pretrained tag 的 QuickGELU 设置存在提示性差异。实验已正常完成，结果可用。后续如需更严谨比较，可改用 open_clip 推荐的 pretrained/model 组合或测试不同模型。
```

下一步建议：

```text
先不要急着上 VLA。建议第二步做 prompt refinement，对同一批图片测试更细 prompt；第三步再提取 image embeddings 做最近邻检索。
```

## 6. Prompt refinement 脚本准备

日期：

```text
2026-05-27
```

改动：

```text
clip_zero_shot.py 新增 --prompt-file 参数。
新增 experiments/prompts/puffin_scene_prompts.txt。
新增 run_clip_prompt_refinement.cmd。
```

新 prompt 集：

```text
a group of puffins on a rocky cliff
a distant drone photo of tiny seabirds
a close-up photo of a single puffin
overlapping puffins in a group
a rocky cliff with no visible birds
a browser screenshot of an object detection result
a photo of seabirds on a cliff
a photo of rocks
```

运行方式：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_prompt_refinement.cmd
```

预期观察：

```text
检查 CLIP 是否仍然偏向 broad prompt，或是否能把 close-up、drone、overlap、detection screenshot 等更细场景区分开。
```

## 7. Prompt refinement 运行结果

日期：

```text
2026-05-27
```

命令：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_prompt_refinement.cmd
```

输出：

```text
outputs/clip_prompt_refinement/scores.csv
outputs/clip_prompt_refinement/top_matches.csv
```

top match 汇总：

```text
01_success_complex_23_targets.png                         -> a browser screenshot of an object detection result
02_failure_case_001_drone_far_both_under_count.png        -> overlapping puffins in a group
03_failure_case_002_drone_far_under_to_over.png           -> a browser screenshot of an object detection result
04_failure_case_003_near_overlap_both_under_count.png     -> a browser screenshot of an object detection result
05_failure_case_004_single_closeup_yolo11n_over_count.png -> overlapping puffins in a group
06_case_005_near_overlap_yolo11n_success.png              -> a browser screenshot of an object detection result
```

观察：

```text
这轮 prompt 变细后，CLIP 明显捕捉到了“浏览器截图 / object detection result”这个视觉域。多数图片 top prompt 不再是自然图像语义，而是识别出它们是带 UI 和检测框的截图。
这说明当前输入素材包含较强的界面/标注框偏置，会影响 CLIP 对原始场景的判断。
02 和 05 被匹配到 overlapping puffins in a group，说明细 prompt 能在部分图像上触发更具体的场景描述。
```

结论：

```text
prompt refinement 有价值，但当前浏览器截图不是最干净的 CLIP 语义测试输入。
如果想评估 CLIP 对 puffin 场景本身的理解，下一步应该改用原始图片或模型输出的纯预测图，而不是包含浏览器 UI 的整页截图。
```

下一步建议：

```text
优先做 image set cleanup：准备一组不含浏览器 UI 的图像输入。
可以直接复用 final-demo 的 outputs/failure_cases/... 中 original.jpg 和 exp002_yolo11n_pred.jpg，或者从报告截图中裁剪出图像区域。
然后重新跑 zero-shot，对比“原始图像”和“检测结果图”的 CLIP 匹配差异。
```

## 8. Clean image comparison 脚本准备

日期：

```text
2026-05-27
```

目标：

```text
去掉浏览器 UI 截图干扰，改用 final-demo 中已经整理好的失败案例原图和 YOLO11n 预测图。
不复制图片本体，只用 manifest 记录路径。
```

新增/修改：

```text
clip_zero_shot.py 新增 --manifest 参数。
experiments/manifests/final_demo_failure_cases.csv
run_clip_clean_images.cmd
```

manifest 内容：

```text
5 个 failure cases
每个 case 包含 original.jpg 和 exp002_yolo11n_pred.jpg
共 10 张输入图像
```

运行方式：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_clean_images.cmd
```

预期观察：

```text
比较 original 与 yolo11n_pred 的 top prompt 是否不同。
如果 yolo11n_pred 更容易匹配到 detection/result 相关语义，说明检测框和可视化标注会显著影响 CLIP 语义。
如果 original 更容易匹配 puffin/cliff/drone/overlap prompt，则说明 clean image 更适合测试自然图像理解。
```

## 9. Clean image comparison 运行结果

日期：

```text
2026-05-27
```

命令：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_clean_images.cmd
```

输出：

```text
outputs/clip_clean_images/scores.csv
outputs/clip_clean_images/top_matches.csv
```

top match 汇总：

```text
case_001_original -> a photo of seabirds on a cliff
case_001_yolo11n  -> a photo of seabirds on a cliff

case_002_original -> a rocky cliff with no visible birds
case_002_yolo11n  -> a browser screenshot of an object detection result

case_003_original -> a group of puffins on a rocky cliff
case_003_yolo11n  -> overlapping puffins in a group

case_004_original -> overlapping puffins in a group
case_004_yolo11n  -> a browser screenshot of an object detection result

case_005_original -> a group of puffins on a rocky cliff
case_005_yolo11n  -> a browser screenshot of an object detection result
```

观察：

```text
去掉浏览器 UI 后，CLIP 对原始图像的匹配更接近自然场景语义。
带 YOLO11n 检测框的 prediction 图仍然更容易触发 object detection result prompt，说明检测可视化本身会显著改变 CLIP 的语义判断。
case_002 original 被匹配为 rocky cliff with no visible birds，符合超远景小目标难分辨的失败特征。
case_003 和 case_005 original 能匹配到 group/puffins/rocky cliff，说明 clean image 比浏览器截图更适合测试 CLIP 对 puffin 场景本身的理解。
```

结论：

```text
后续做 image-text 或 image-image embedding 实验时，应优先使用原始图像；检测结果图可以作为单独组别，用来研究可视化标注对视觉语义 embedding 的影响。
```

下一步建议：

```text
进入 image-image retrieval：提取 CLIP image embeddings，分别比较 original 与 yolo11n_pred 的最近邻关系。
重点看相同 case 的 original/pred 是否互为近邻，以及远景、重叠、近景样例是否按语义聚类。
```

## 10. Image-image retrieval 脚本准备

日期：
```text
2026-05-27
```

新增文件：
```text
experiments/clip_image_retrieval.py
run_clip_image_retrieval.cmd
```

目标：
```text
在不引入文本 prompt 的情况下，只使用 CLIP image encoder 提取图像 embedding。
随后计算所有输入图片之间的 cosine similarity，并导出最近邻表和相似度矩阵。
```

输入：
```text
experiments/manifests/final_demo_failure_cases.csv
```

输出：
```text
outputs/clip_image_retrieval/metadata.csv
outputs/clip_image_retrieval/nearest_neighbors.csv
outputs/clip_image_retrieval/similarity_matrix.csv
outputs/clip_image_retrieval/embeddings.npy
outputs/clip_image_retrieval/similarity_matrix.npy
```

运行方式：
```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_image_retrieval.cmd
```

设计说明：
```text
该脚本继续使用 ViT-B-32 / openai 这组 CLIP 权重，并复用项目内 models/ 缓存目录。
图片经过 open_clip 的预处理和 image encoder 后进行 L2 normalization，因此 embedding 点积等价于 cosine similarity。
本步骤重点观察同一 failure case 的 original 与 yolo11n_pred 是否彼此接近，以及不同失败模式是否能在 CLIP 视觉空间中形成可解释的近邻关系。
```
