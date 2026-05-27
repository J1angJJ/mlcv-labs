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
