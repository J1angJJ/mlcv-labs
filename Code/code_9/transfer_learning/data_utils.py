"""
数据加载与处理模块
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import random
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, IMG_SIZE


def get_data_transforms():
    """定义数据变换"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def load_cifar10(use_subset=False, subset_size=0.3):
    """加载CIFAR-10数据集"""
    train_transform, test_transform = get_data_transforms()

    # 训练集
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    # 是否使用子集（加速实验）
    if use_subset:
        # 随机选择子集
        train_indices = list(range(len(trainset)))
        random.shuffle(train_indices)
        train_indices = train_indices[:int(len(trainset) * subset_size)]

        test_indices = list(range(len(testset)))
        random.shuffle(test_indices)
        test_indices = test_indices[:int(len(testset) * subset_size)]

        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)

    # 创建数据加载器
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 类别名称
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def visualize_sample_batch(dataloader, classes):
    """可视化一个批次的图像"""
    # 获取一批数据
    images, labels = next(iter(dataloader))

    # 逆标准化
    images_np = images.numpy()
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((1, 3, 1, 1))
    images_np = images_np * std + mean
    images_np = np.clip(images_np, 0, 1)

    # 显示图像
    plt.figure(figsize=(12, 6))
    for i in range(min(8, BATCH_SIZE)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.transpose(images_np[i], (1, 2, 0)))
        plt.title(classes[labels[i]], fontsize=12, fontweight='bold')
        plt.axis('off')
    plt.suptitle("CIFAR-10样本图像", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()