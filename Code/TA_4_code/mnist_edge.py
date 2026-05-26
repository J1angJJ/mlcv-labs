"""
MNIST边缘检测实验
功能：训练CNN模型预测手写数字的边缘图
用法：
  训练：python mnist_edge.py --train
  推理：python mnist_edge.py --model mnist_edge.pth
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

# 命令行参数解析
parser = argparse.ArgumentParser(description='MNIST边缘检测')
parser.add_argument('--model', type=str, help='模型路径')
parser.add_argument('--train', action='store_true', help='训练模式')
args = parser.parse_args()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EdgeDataset(Dataset):
    """生成边缘检测标签的自定义数据集"""

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset  # 原始MNIST数据集

    def __len__(self):
        return len(self.mnist_dataset)  # 返回样本总数

    def __getitem__(self, idx):
        # 获取原始图像
        img, _ = self.mnist_dataset[idx]
        img_np = img.squeeze().numpy()  # 转换为NumPy数组

        # 使用Sobel算子计算边缘
        dx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
        dy = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
        edge = np.sqrt(dx ** 2 + dy ** 2)  # 梯度幅值
        edge = (edge / edge.max()).astype(np.float32)  # 归一化

        return img, torch.from_numpy(edge).unsqueeze(0)  # 返回图像-边缘对


# 自动下载MNIST数据集
mnist_train = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 创建边缘检测数据集
train_set = EdgeDataset(mnist_train)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)


class EdgeDetector(nn.Module):
    """边缘检测CNN模型"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 1输入通道，16输出通道
            nn.ReLU(),  # 激活函数
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # 输出单通道边缘图
            nn.Sigmoid()  # 概率输出
        )

    def forward(self, x):
        return self.layers(x)


model = EdgeDetector().to(device)

# 加载已有模型
if args.model:
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"已加载模型: {args.model}")

# 训练模式
if args.train:
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练20个epoch
    for epoch in range(20):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/20]')

        for images, edges in progress_bar:
            images, edges = images.to(device), edges.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, edges)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    torch.save(model.state_dict(), 'mnist_edge.pth')
    print("模型已保存为: mnist_edge.pth")

# 可视化结果
model.eval()
with torch.no_grad():
    sample_img, sample_edge = train_set[0]
    pred = model(sample_img.unsqueeze(0).to(device)).cpu().squeeze()

    plt.rc("font", family='SimHei')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_img.squeeze(), cmap='gray')
    plt.title('输入图像')
    plt.subplot(1, 3, 2)
    plt.imshow(sample_edge.squeeze(), cmap='gray')
    plt.title('真实边缘')
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('预测边缘')
    plt.show()