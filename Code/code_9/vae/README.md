# 自编码器实验

自编码器(Autoencoder)模型实验。

## 项目概述

自编码器是一种无监督学习算法，其目标是学习输入数据的有效编码(encoding)，并能够从这些编码中重建原始数据。本项目实现了以下几种自编码器模型：

1. 基本自编码器 (Basic Autoencoder)
2. 欠完备自编码器 (Undercomplete Autoencoder)
3. 正则化自编码器 (Regularized Autoencoder)
4. 去噪自编码器 (Denoising Autoencoder)
5. 变分自编码器 (Variational Autoencoder, VAE)
6. 卷积自编码器 (Convolutional Autoencoder) - 主要用于CIFAR-10数据集

每种模型都可以在MNIST和CIFAR-10数据集上进行训练和测试。

### 训练模型
训练所有模型
python main.py --mode train --model all

训练指定模型（例如VAE）
python main.py --mode train --model vae --epochs 10

在CIFAR-10数据集上训练模型
python main.py --mode train --model vae --dataset cifar10

### 可视化结果
可视化所有模型
python main.py --mode visualize --model all

可视化指定模型
python main.py --mode visualize --model denoising

### 训练并可视化
训练并可视化所有模型
python main.py --mode all --model all

在CIFAR-10上训练并可视化VAE模型
python main.py --mode all --model vae --dataset cifar10