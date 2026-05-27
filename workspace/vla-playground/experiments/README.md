# Experiments

本目录存放 VLA playground 的实验脚本。

## `clip_zero_shot.py`

用途：

```text
使用 open_clip 对一组本地图像和文本 prompt 计算 image-text similarity。
默认复用 final-demo 的 puffin 浏览器截图素材，不迁移图片。
```

默认输入：

```text
R:\mlcv-labs\workspace\final-demo\report_assets\browser_screenshots
```

默认输出：

```text
R:\mlcv-labs\workspace\vla-playground\outputs\clip_zero_shot\scores.csv
R:\mlcv-labs\workspace\vla-playground\outputs\clip_zero_shot\top_matches.csv
```

运行命令：

推荐使用 `cmd` 脚本，避开 PowerShell/conda run 的编码和进度显示问题：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_zero_shot.cmd
```

或手动运行：

```powershell
cd R:\mlcv-labs\workspace\vla-playground
conda run -n cv-train python experiments\clip_zero_shot.py
```

首次运行会下载 open_clip 指定的预训练权重。`clip_zero_shot.py` 已默认把缓存重定向到本项目目录：

```text
R:\mlcv-labs\workspace\vla-playground\models\hf_home
R:\mlcv-labs\workspace\vla-playground\models\torch_home
```

`models/` 已被 `.gitignore` 忽略。默认的 `ViT-B-32 / openai` 权重预计是几百 MB 级别，远低于 10GB。

如需覆盖缓存位置，可以在运行前手动设置环境变量：

```powershell
$env:HF_HOME="R:\mlcv-labs\workspace\vla-playground\models\hf_home"
$env:TORCH_HOME="R:\mlcv-labs\workspace\vla-playground\models\torch_home"
conda run -n cv-train python experiments\clip_zero_shot.py
```

## Prompt refinement

更细的 puffin 场景 prompt 文件：

```text
experiments\prompts\puffin_scene_prompts.txt
```

推荐用 cmd 脚本运行：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_prompt_refinement.cmd
```

输出：

```text
outputs\clip_prompt_refinement\scores.csv
outputs\clip_prompt_refinement\top_matches.csv
```

## Clean image comparison

这一步不复制图片，只通过 manifest 指向 final-demo 中已有的失败案例原图和 YOLO11n 预测图：

```text
experiments\manifests\final_demo_failure_cases.csv
```

推荐运行：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_clean_images.cmd
```

输出：

```text
outputs\clip_clean_images\scores.csv
outputs\clip_clean_images\top_matches.csv
```

用途：

```text
比较 CLIP 对原始自然图像和 YOLO 检测可视化图的 prompt 匹配差异，避免浏览器 UI 截图干扰。
```

## Image-image retrieval

用途：

```text
提取同一批 clean images 的 CLIP image embeddings，并计算图片之间的 cosine similarity。
这一步不再依赖文本 prompt，而是观察原图、检测结果图、远景、重叠和近景样例在视觉表征空间中的近邻关系。
```

推荐运行：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_image_retrieval.cmd
```

输入：

```text
experiments\manifests\final_demo_failure_cases.csv
```

输出：

```text
outputs\clip_image_retrieval\metadata.csv
outputs\clip_image_retrieval\nearest_neighbors.csv
outputs\clip_image_retrieval\similarity_matrix.csv
outputs\clip_image_retrieval\embeddings.npy
outputs\clip_image_retrieval\similarity_matrix.npy
```

阅读重点：

```text
nearest_neighbors.csv 用来快速查看每张图的 top-k 近邻。
similarity_matrix.csv 用来查看全部两两相似度。
如果同一 case 的 original 和 yolo11n_pred 互为近邻，说明检测框可视化没有完全破坏图像语义；
如果预测图更接近其他 prediction 图，说明方框和标注样式本身成为了强视觉特征。
```

## Embedding plot

用途：

```text
读取 image-image retrieval 阶段保存的 CLIP image embeddings，把高维向量投影到二维并生成散点图。
当前默认使用 PCA，因为样本数只有 10 张，PCA 比 t-SNE 更稳定、可解释。
```

推荐运行：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_embedding_plot.cmd
```

输出：

```text
outputs\clip_embedding_plot\pca_points.csv
outputs\clip_embedding_plot\pca_plot.png
```

可选：

```cmd
cd /d R:\mlcv-labs\workspace\vla-playground
C:\Users\JJ406\.conda\envs\cv-train\python.exe experiments\visualize_clip_embeddings.py --method tsne --perplexity 3
```

说明：

```text
PCA 是线性降维，适合先看整体方向和相对距离。
t-SNE 是非线性邻域可视化，适合样本更多时观察局部簇结构；样本很少时不宜过度解读。
```

## Probe

用途：

```text
在冻结的 CLIP image embeddings 上训练很小的分类器，检查 embedding 是否包含某类可分信息。
当前是教学实验，不是正式评估，因为样本只有 10 张。
```

任务：

```text
source_style: original vs yolo11n_pred
scene_distance: far_scene vs close_scene
```

推荐运行：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_clip_probe.cmd
```

输出：

```text
outputs\clip_probe\summary.csv
outputs\clip_probe\source_style_predictions.csv
outputs\clip_probe\scene_distance_predictions.csv
```

说明：

```text
KNN probe 不训练参数，只看最近邻类别。
Linear probe 使用 logistic regression，只训练一个线性分类边界。
如果简单 probe 能分开，说明 CLIP embedding 原本就包含相应属性的信息。
```

## Annotation manifest

用途：

```text
从 final-demo 的 Seabirds.v6i.yolo26 数据集中分层抽样 300 张原图，生成后续 CLIP probe、linear classifier、LoRA 或 VLA toy task 可复用的多字段 manifest。
```

生成命令：

```cmd
R:\mlcv-labs\workspace\vla-playground\run_build_annotation_manifest.cmd
```

输出：

```text
experiments\manifests\seabirds_annotation_300.csv
experiments\annotation\annotate_seabirds_300.html
```

抽样策略：

```text
按 split 近似抽取 train 216、valid 48、test 36。
每个 split 内优先按 puffin 标注框数量分层：none、single、few、group。
不复制图片，只在 manifest 中记录原始图片路径和 YOLO label 路径。
```

标注方式：

```text
用浏览器打开 experiments\annotation\annotate_seabirds_300.html。
逐张补充 distance、scene、difficulty、occlusion、text_prompt 等字段。
页面会把草稿保存在浏览器 localStorage；完成后点击 Download CSV，得到 seabirds_annotation_300_annotated.csv。
浏览器不能直接覆盖仓库里的 CSV，下载后需要手动替换或另存为新的 manifest。
```

字段说明：

```text
image_id/image_path/split/source/dataset/label_path: 样本来源。
total_boxes/puffin_boxes/non_puffin_boxes/puffin_bin: 从 YOLO 标签自动统计。
distance/scene/difficulty/occlusion/text_prompt/notes: 人工补充字段。
contains_puffin/density/has_detection_overlay: 已给出初始值，可人工修正。
```
