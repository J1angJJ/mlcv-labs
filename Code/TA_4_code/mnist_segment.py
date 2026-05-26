"""
MNIST手写数字二值分割实验
功能：训练全卷积网络将MNIST图像中的数字区域分割出来
用法：
  训练：python mnist_segment.py --train
  推理：python mnist_segment.py --model mnist_segment.pth
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# 命令行参数解析
parser = argparse.ArgumentParser(description='MNIST分割实验')
parser.add_argument('--model', type=str, help='模型路径')
parser.add_argument('--train', action='store_true', help='训练模式')
args = parser.parse_args()

# 设备配置（自动检测GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理管道
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为[0,1]范围的张量
])

# 自动下载并加载MNIST数据集
train_set = datasets.MNIST(
    root='./data',  # 数据集存储路径
    train=True,  # 加载训练集
    download=True,  # 如果本地不存在则自动下载
    transform=transform
)
test_set = datasets.MNIST(
    root='./data',
    train=False,  # 加载测试集
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,  # 每批加载32张图像
    shuffle=True,  # 打乱数据顺序
    num_workers=0  # 使用主线程加载数据
)


# 定义分割模型
class MNISTSegmenter(nn.Module):
    """全卷积分割网络，输入输出尺寸相同"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 第一卷积层：1输入通道，8输出通道，3x3卷积核，填充1保持尺寸
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # 原地激活节省内存
            # 第二卷积层：8输入通道，1输出通道
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出[0,1]概率图
        )

    def forward(self, x):
        return self.layers(x)


# 初始化模型并移至设备
model = MNISTSegmenter().to(device)

# 加载已有模型
if args.model:
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"已加载预训练模型: {args.model}")

# 训练模式
if args.train:
    criterion = nn.BCELoss()  # 二值交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # 训练15个epoch
    for epoch in range(15):
        model.train()  # 设置训练模式
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/15]')

        for images, _ in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, images)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 更新进度条显示
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    # 保存训练好的模型
    torch.save(model.state_dict(), 'mnist_segment.pth')
    print("模型已保存为: mnist_segment.pth")

# 结果可视化（无论是否训练都执行）
model.eval()  # 设置评估模式
with torch.no_grad():
    # 获取测试集第一个样本
    test_image, _ = test_set[0]
    test_image = test_image.unsqueeze(0).to(device)  # 增加批次维度

    # 生成预测掩码
    prediction = model(test_image).cpu().squeeze()

    # 绘制对比图
    plt.rc("font", family='SimHei')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image.cpu().squeeze(), cmap='gray')
    plt.title('输入图像')
    plt.subplot(1, 3, 2)
    plt.imshow((test_image.cpu().squeeze() > 0.5), cmap='gray')
    plt.title('真实掩码')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction > 0.5, cmap='gray')
    plt.title('预测结果')
    plt.show()