# arcface_data.py - 数据加载与预处理

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# 导入中文字体设置
from chinese_font import set_chinese_font
set_chinese_font()  # 设置中文显示

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 调整图像大小为112x112
    transforms.ToTensor(),  # 转换为张量，范围[0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化为[-1,1]
])


def load_data(data_path, batch_size=32, train_ratio=0.8):
    """
    加载人脸数据集并划分训练集和测试集

    参数:
    data_path: 数据集路径
    batch_size: 批次大小
    train_ratio: 训练集比例

    返回:
    train_loader: 训练数据加载器
    test_loader: 测试数据加载器
    num_classes: 类别数量
    """
    # 加载数据集
    dataset = datasets.ImageFolder(data_path, transform=transform)

    # 获取类别数量（人物数量）
    num_classes = len(dataset.classes)
    print(f"数据集包含 {num_classes} 个人物类别")

    # 划分训练集和测试集
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    print(f"数据集大小: {len(dataset)}张图像")
    print(f"训练集: {len(train_set)}张图像")
    print(f"测试集: {len(test_set)}张图像")
    return train_loader, test_loader, num_classes


def show_batch(dataloader):
    """显示一批数据的图像"""
    # 获取一批数据
    images, labels = next(iter(dataloader))
    # 反归一化
    images = images * 0.5 + 0.5
    # 创建图像网格
    grid = torchvision.utils.make_grid(images[:16])
    # 显示图像
    plt.figure(figsize=(10, 6))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title('数据集样本图像')
    plt.show()


if __name__ == "__main__":
    # 测试数据加载
    train_loader, test_loader, _ = load_data("./data/lfw_subset")
    show_batch(train_loader)