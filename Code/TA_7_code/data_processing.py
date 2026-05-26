# data_processing.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import numpy as np


class PetSegmentationDataset(Dataset):
    def __init__(self, root, split='trainval', transform=None, target_transform=None):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
            transform=transform,
            target_transform=target_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # 处理掩码：Pet数据集的掩码有3个类别（1:宠物, 2:背景, 3:边界）
        # 将它们映射到0,1,2以便于使用CrossEntropyLoss
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = mask - 1  # 将类别从[1,2,3]映射到[0,1,2]

        return img, mask


def get_data_loaders(data_root='./data', batch_size=8, image_size=128):
    """
    创建训练和验证数据加载器

    参数:
        data_root (str): 数据集保存的根目录
        batch_size (int): 批次大小
        image_size (int): 图像缩放的尺寸

    返回:
        tuple: (train_loader, val_loader)
    """
    # 确保目录存在
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # 加载数据集
    print("正在下载并准备Oxford-IIIT Pet数据集...")
    train_dataset = PetSegmentationDataset(
        root=data_root,
        split='trainval',
        transform=transform,
        target_transform=target_transform
    )

    val_dataset = PetSegmentationDataset(
        root=data_root,
        split='test',
        transform=transform,
        target_transform=target_transform
    )

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader = get_data_loaders()
    print("数据加载器创建成功！")

    # 查看一个批次的数据
    images, masks = next(iter(train_loader))
    print(f"图像张量形状: {images.shape}")
    print(f"掩码张量形状: {masks.shape}")
    print(f"图像数值范围: ({images.min():.2f}, {images.max():.2f})")
    print(f"掩码类别: {torch.unique(masks)}")