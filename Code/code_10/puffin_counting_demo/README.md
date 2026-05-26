# Puffin Counting Demo — TA10 综合实验

本实验演示如何用预训练的YOLOv8模型完成鸟类检测与计数，串联课程中目标检测、迁移学习、NMS等核心知识点。

## 环境要求

- Python 3.8+
- 无需GPU，CPU即可运行（有GPU会自动加速）
- 预计运行时间：CPU约3-5分钟

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行实验

```bash
python main.py
```

运行后会在 `outputs/` 目录下生成所有可视化结果图片。

## 实验内容

1. 加载COCO预训练YOLOv8n，检测鸟类目标
2. 可视化检测结果（bounding box + 置信度）
3. 提取backbone中间层特征并生成热力图
4. NMS IoU阈值对比实验（0.3 / 0.5 / 0.7）
5. 高斯密度图生成演示（点标注→密度图→计数）
