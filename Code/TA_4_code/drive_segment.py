"""
DRIVE视网膜血管分割实验（完整GPU支持版）
数据集结构：
DRIVE/
├── training/
│   ├── images/  # 包含21_training.tif等文件
│   └── masks/   # 包含21_training_mask.gif等文件
└── test/
    ├── images/  # 包含01_test.tif等文件
    └── masks/   # 包含01_test_mask.gif等文件
用法：
训练：python drive_segment.py --train
推理：python drive_segment.py --model drive_segment.pth
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------- CUDA诊断 --------------------
def check_cuda_support():
    """检查CUDA支持情况并打印诊断信息"""
    print("="*50)
    print("CUDA支持诊断:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"检测到GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n警告：未检测到可用GPU!")
    print("="*50 + "\n")

# -------------------- 命令行参数解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description='视网膜血管分割实验')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--data_root', type=str, default='./DRIVE',
                       help='数据集根目录（默认：./DRIVE）')
    return parser.parse_args()

# -------------------- 数据集加载器 --------------------
class DRIVEDataset(Dataset):
    """
    DRIVE数据集加载器
    命名规则：
    - 训练图像：XX_training.tif
    - 训练掩码：XX_training_mask.gif
    - 测试图像：XX_test.tif
    - 测试掩码：XX_test_mask.gif
    """
    def __init__(self, root_dir, train=True, size=(128, 128)):
        self.root_dir = root_dir
        self.train = train
        self.size = size
        self.mode = 'training' if train else 'test'
        self.img_keyword = '_training.tif' if train else '_test.tif'
        self.mask_keyword = '_training_mask.gif' if train else '_test_mask.gif'

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

        # 初始化路径
        self.img_dir = os.path.join(root_dir, self.mode, 'images')
        self.mask_dir = os.path.join(root_dir, self.mode, 'masks')

        # 验证数据集
        self._validate_dataset()
        self.img_files = sorted(
            [f for f in os.listdir(self.img_dir) if f.endswith(self.img_keyword)],
            key=lambda x: int(x.split('_')[0])
        )

    def _validate_dataset(self):
        """执行严格的数据验证"""
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")

        sample_files = os.listdir(self.img_dir)
        if not any(f.endswith(self.img_keyword) for f in sample_files):
            raise FileNotFoundError(f"目录中未找到{self.img_keyword}文件: {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = f"{img_name.split('_')[0]}{self.mask_keyword}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 加载数据
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        mask = Image.open(mask_path).convert('L')
        mask = self.transform(mask)
        mask = (mask > 0.5).float()  # 二值化

        return img, mask

# -------------------- U-Net模型 --------------------
class UNet(nn.Module):
    """U-Net医学图像分割网络"""
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128x128 -> 64x64
        )
        self.middle = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64x64 -> 32x32
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 32x32 -> 64x64
        )
        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 64x64 -> 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return self.final(x)

# -------------------- 训练函数 --------------------
def train_model(model, train_loader, device, epochs=50):
    """模型训练函数"""
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-8)

    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]')

        for images, masks in progress_bar:
            # 数据移至设备
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        print(f"Epoch [{epoch+1}/{epochs}] 平均损失: {total_loss/len(train_loader):.4f}")

# -------------------- 可视化函数 --------------------
def visualize_results(model, dataset, device, num_samples=3):
    """可视化预测结果"""
    model.eval()
    plt.rc("font", family='SimHei')
    fig = plt.figure(figsize=(15, 5*num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            img, true_mask = dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            pred_mask = model(img_tensor).cpu().squeeze()


            # 绘制图像
            ax = fig.add_subplot(num_samples, 3, i*3+1)
            ax.imshow(img.permute(1,2,0))
            ax.set_title(f'样本 {i+1} 输入')
            ax.axis('off')

            ax = fig.add_subplot(num_samples, 3, i*3+2)
            ax.imshow(true_mask.squeeze(), cmap='gray')
            ax.set_title(f'样本 {i+1} 真实血管')
            ax.axis('off')

            ax = fig.add_subplot(num_samples, 3, i*3+3)
            ax.imshow(pred_mask > 0.5, cmap='gray')
            ax.set_title(f'样本 {i+1} 预测血管')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# -------------------- 主函数 --------------------
def main():
    args = parse_args()
    check_cuda_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 初始化数据集
        print("\n初始化数据集...")
        drive_train = DRIVEDataset(args.data_root, train=True)
        drive_test = DRIVEDataset(args.data_root, train=False)
        print(f"训练集样本数: {len(drive_train)}")
        print(f"测试集样本数: {len(drive_test)}")

        # 初始化模型
        print("\n初始化模型...")
        model = UNet().to(device)
        print(f"模型已加载到设备: {device}")

        # 加载已有模型
        if args.model:
            model.load_state_dict(torch.load(args.model, map_location=device))
            print(f"成功加载模型: {args.model}")

        # 训练模式
        if args.train:
            print("\n启动训练...")
            train_loader = DataLoader(drive_train, batch_size=8, shuffle=True, num_workers=2)
            train_model(model, train_loader, device)
            torch.save(model.state_dict(), 'drive_segment.pth')
            print("\n训练完成！模型已保存为 drive_segment.pth")

        # 可视化结果
        if args.model or args.train:
            print("\n生成可视化结果...")
            visualize_results(model, drive_test, device)

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()