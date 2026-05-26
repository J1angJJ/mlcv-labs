# 迁移学习综合实验

本实验探索不同迁移学习策略在CIFAR-10图像分类任务上的效果对比，包括特征提取、微调和渐进式解冻三种方法。

## 实验内容

本实验将使用预训练的卷积神经网络作为基础模型，在CIFAR-10数据集上进行迁移学习，对比不同迁移学习策略的效果：

1. **特征提取**：冻结预训练模型的所有权重，仅训练新添加的分类器层
2. **微调**：使用不同学习率同时训练预训练模型和分类器层
3. **渐进式解冻**：从浅层到深层逐步解冻模型层，实现更好的知识迁移

## 命令行参数
实验支持多种命令行参数：
python main.py [--model MODEL] [--epochs EPOCHS] [--use_subset] [--subset_size SUBSET_SIZE]

## 使用示例
1. **使用ResNet18(默认)**：python main.py
2. **使用MobileNetV2进行5轮训练**：python main.py --model mobilenet_v2 --epochs 5
3. **在数据子集上快速实验**：python main.py --use_subset --subset_size 0.2

## config修改
### 数据配置
BATCH_SIZE = 64             # 批处理大小
NUM_WORKERS = 4             # 数据加载线程数
IMG_SIZE = 224              # 图像大小

### 学习率配置
FEATURE_EXTRACT_LR = 0.001  # 特征提取模型学习率
FINE_TUNE_LR_FC = 0.001     # 微调分类器学习率
FINE_TUNE_LR_BACKBONE = 0.0001  # 微调主干网络学习率

### 渐进式解冻配置
GRAD_UNFREEZE_EVERY = 2     # 每隔多少个epoch解冻一组新层