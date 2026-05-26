# cifar10_cnn.py - 基础CNN模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import os


class BasicCNN(nn.Module):
    """基础卷积神经网络模型，用于CIFAR-10分类"""

    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(F.relu(self.conv1(x)))

        # 第二个卷积块
        x = self.pool2(F.relu(self.conv2(x)))

        # 第三个卷积块
        x = self.pool3(F.relu(self.conv3(x)))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_features(self, x):
        """提取特征，用于可视化"""
        # 第一个卷积块
        x = self.pool1(F.relu(self.conv1(x)))

        # 第二个卷积块
        x = self.pool2(F.relu(self.conv2(x)))

        # 第三个卷积块
        x = self.pool3(F.relu(self.conv3(x)))

        # 展平
        features = x.view(x.size(0), -1)
        return features


def create_cnn_model(num_classes=10):
    """创建并初始化BasicCNN模型"""
    model = BasicCNN(num_classes)

    # 初始化模型权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model


def test_model_speed(model, input_size=(100, 3, 32, 32), num_runs=50):
    """测试模型的前向传播速度"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 测速
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time


def main():
    """当作为脚本运行时的主函数"""
    parser = argparse.ArgumentParser(description='CNN模型测试')
    parser.add_argument('--classes', type=int, default=10, help='类别数量')
    parser.add_argument('--save', action='store_true', help='保存模型')
    args = parser.parse_args()

    # 创建CNN模型
    model = create_cnn_model(args.classes)

    # 打印模型信息
    print("基础CNN模型结构:")
    print(model)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数总数: {total_params:,}")

    # 测试前向传播
    x = torch.randn(4, 3, 32, 32)  # 4张32x32的3通道图像
    output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 测试模型速度
    if torch.cuda.is_available():
        model.cuda()
        print("模型已移至GPU进行速度测试")

    batch_sizes = [1, 16, 64, 128]
    for batch_size in batch_sizes:
        avg_time = test_model_speed(model, (batch_size, 3, 32, 32))
        print(f"批次大小 {batch_size}: 平均推理时间 = {avg_time * 1000:.2f} ms, "
              f"吞吐量 = {batch_size / avg_time:.1f} 图像/秒")

    # 保存模型
    if args.save:
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/basic_cnn.pth')
        print("模型已保存到 models/basic_cnn.pth")


if __name__ == "__main__":
    main()