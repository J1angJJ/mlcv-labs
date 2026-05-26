# model_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 设置环境变量解决OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):  # Pet数据集有3个类别
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = DoubleConv(512, 1024)

        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        # Output layer
        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.conv5(p4)

        # Decoder
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        # Output layer
        out = self.conv10(c9)
        return out


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
          model_save_path='best_pet_unet_model.pth'):
    """
    训练UNet模型

    参数:
        model: UNet模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备 (CPU/GPU)
        model_save_path: 模型保存路径

    返回:
        tuple: (train_losses, val_losses)
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 保存损失曲线
    loss_curve_path = 'pet_loss_curve.png'
    plt.savefig(loss_curve_path)
    print(f"损失曲线图像已保存至: {os.path.abspath(loss_curve_path)}")

    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图像: {e}")

    return train_losses, val_losses


def setup_training(train_loader, val_loader, n_channels=3, n_classes=3, learning_rate=0.001, num_epochs=5):
    """
    设置并启动训练过程

    参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_channels: 输入通道数
        n_classes: 分类类别数
        learning_rate: 学习率
        num_epochs: 训练轮数

    返回:
        model: 训练好的模型
    """
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建模型
    model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print(f"开始训练，将进行{num_epochs}个epochs...")
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_pet_unet_model.pth', map_location=device))

    return model


if __name__ == "__main__":
    # 导入数据处理模块
    from data_processing import get_data_loaders

    # 获取数据加载器
    train_loader, val_loader = get_data_loaders()

    # 询问用户是否开始训练
    user_input = input("数据集已准备好。开始训练模型? (y/n): ")
    if user_input.lower() != 'y':
        print("训练已取消。")
    else:
        # 设置并开始训练
        model = setup_training(train_loader, val_loader, num_epochs=5)
        print("训练完成!")