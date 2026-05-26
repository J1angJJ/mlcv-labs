# cifar10_data.py - 数据加载和预处理

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from chinese_font import set_chinese_font


def load_cifar10_data(batch_size=128, augment=True, download=True):
    """加载CIFAR-10数据集并应用预处理"""
    # 设置中文字体显示
    set_chinese_font()

    # CIFAR-10数据集均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # 训练集变换
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # 测试集变换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 加载训练集
    print("正在加载CIFAR-10训练集...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=download,
        transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # 加载测试集
    print("正在加载CIFAR-10测试集...")
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # CIFAR-10类别（中文）
    classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

    print(f"数据集加载完成，训练集: {len(trainset)}张图像，测试集: {len(testset)}张图像")
    return train_loader, test_loader, classes


def show_images(loader, classes, num_images=5, save_path='cifar10_samples.png'):
    """
    显示数据集中的一些图像

    参数:
        loader: 数据加载器
        classes: 类别名称列表
        num_images: 每个类别显示的图像数量
        save_path: 保存路径
    """
    # 收集每个类别的图像
    class_images = {i: [] for i in range(10)}

    # CIFAR-10数据集均值和标准差（用于反标准化）
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    # 从加载器中获取图像
    for images, labels in loader:
        for img, label in zip(images, labels):
            label_idx = label.item()
            if len(class_images[label_idx]) < num_images:
                class_images[label_idx].append(img)

        # 检查是否已经收集了足够的图像
        if all(len(imgs) >= num_images for imgs in class_images.values()):
            break

    # 创建图形显示图像
    plt.figure(figsize=(15, 10))
    for i, class_name in enumerate(classes):
        for j, img in enumerate(class_images[i]):
            # 计算子图位置
            plt.subplot(len(classes), num_images, i * num_images + j + 1)

            # 反标准化：将标准化的图像转换回原始像素值
            img = img.numpy().transpose((1, 2, 0))  # (C,H,W) -> (H,W,C)
            img = std * img + mean  # 应用反标准化
            img = np.clip(img, 0, 1)  # 裁剪到[0,1]范围

            # 显示图像
            plt.imshow(img)

            # 在第一张图像上显示类别名称
            if j == 0:
                plt.title(class_name)
            plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"已保存样本图像到 {save_path}")
    return save_path


def main():
    """当作为脚本运行时的主函数"""
    parser = argparse.ArgumentParser(description='CIFAR-10数据加载与可视化')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--samples', type=int, default=5, help='每类显示的样本数量')
    parser.add_argument('--output', type=str, default='cifar10_samples.png', help='输出图像路径')
    args = parser.parse_args()

    # 加载数据
    train_loader, _, classes = load_cifar10_data(batch_size=args.batch_size)

    # 显示样本图像
    output_path = show_images(train_loader, classes, args.samples, args.output)
    print(f"完成! 查看样本图像: {output_path}")


if __name__ == "__main__":
    main()