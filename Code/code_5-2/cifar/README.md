# CIFAR-10图像分类实验使用指南

## 环境要求

```
Python 3.6+
PyTorch 1.7+
torchvision
numpy
matplotlib
scikit-learn
seaborn
```

安装依赖:
```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

## 使用方法

### 数据加载与可视化

加载CIFAR-10数据集并显示样本:

```bash
python cifar10_data.py --samples 5 --output cifar10_samples.png
```

参数说明:
- `--samples`: 每个类别显示的样本数量
- `--output`: 输出图像保存路径

### 模型测试

测试模型结构和速度:

**基础CNN模型**:
```bash
python cifar10_cnn.py --save
```

参数说明:
- `--classes`: 类别数量(默认10)
- `--save`: 是否保存模型

**ResNet模型**:
```bash
python cifar10_resnet.py --model resnet20 --save
```

参数说明:
- `--model`: 模型变体(resnet20, resnet32)
- `--classes`: 类别数量(默认10)
- `--save`: 是否保存模型

### 模型训练

训练分类模型:

```bash
python cifar10_train.py --model resnet20 --epochs 100 --batch-size 128 --lr 0.1 --output-dir ./results
```

参数说明:
- `--model`: 模型类型(cnn, resnet20, resnet32)
- `--batch-size`: 批次大小
- `--epochs`: 训练轮次
- `--lr`: 初始学习率
- `--momentum`: 动量参数(默认0.9)
- `--weight-decay`: 权重衰减(默认5e-4)
- `--output-dir`: 输出目录

### 结果可视化

可视化训练好的模型:

```bash
python cifar10_visualize.py --model-path ./results/resnet20_best.pth --model-type resnet20 --output-dir ./visualizations
```

参数说明:
- `--model-path`: 模型文件路径
- `--model-type`: 模型类型(cnn, resnet20, resnet32)
- `--output-dir`: 可视化结果保存目录
- `--samples`: t-SNE可视化使用的样本数量(默认800)

## 单独运行模块

每个模块都可以作为独立程序运行:

1. **字体设置测试**:
```bash
python chinese_font.py
```

2. **数据集分析**:
```bash
python cifar10_data.py --batch-size 64 --samples 3
```

3. **CNN模型速度测试**:
```bash
python cifar10_cnn.py
```

4. **ResNet模型比较**:
```bash
python cifar10_resnet.py --model resnet32
```

5. **快速训练**:
```bash
python cifar10_train.py --model cnn --epochs 5 --batch-size 64
```

6. **模型预测可视化**:
```bash
python cifar10_visualize.py --model-path ./models/cnn.pth --model-type cnn
```