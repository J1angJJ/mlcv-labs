# cifar10_resnet.py - 简化版ResNet模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import os


class BasicBlock(nn.Module):
    """ResNet的基本残差块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet模型，针对CIFAR-10优化的变体"""

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        # 初始通道数
        self.in_planes = 16

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 三个残差层组
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 分类器
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始层
        out = F.relu(self.bn1(self.conv1(x)))

        # 三个残差层组
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # 全局平均池化
        out = F.avg_pool2d(out, out.size()[3])

        # 展平
        out = out.view(out.size(0), -1)

        # 分类器
        out = self.linear(out)

        return out

    def get_features(self, x):
        """提取特征，用于可视化"""
        # 初始层
        out = F.relu(self.bn1(self.conv1(x)))

        # 三个残差层组
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # 全局平均池化
        out = F.avg_pool2d(out, out.size()[3])

        # 展平
        features = out.view(out.size(0), -1)
        return features


def create_resnet_model(model_type='resnet20', num_classes=10):
    """创建指定类型的ResNet模型"""
    if model_type == 'resnet20':
        model = ResNet(BasicBlock, [3, 3, 3], num_classes)
        model_name = "ResNet-20"
    elif model_type == 'resnet32':
        model = ResNet(BasicBlock, [5, 5, 5], num_classes)
        model_name = "ResNet-32"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model, model_name


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
    parser = argparse.ArgumentParser(description='ResNet模型测试')
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'resnet32'], help='ResNet模型变体')
    parser.add_argument('--classes', type=int, default=10, help='类别数量')
    parser.add_argument('--save', action='store_true', help='保存模型')
    args = parser.parse_args()

    # 创建ResNet模型
    model, model_name = create_resnet_model(args.model, args.classes)

    # 打印模型信息
    print(f"{model_name}模型结构:")
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
        torch.save(model.state_dict(), f'models/{args.model}.pth')
        print(f"模型已保存到 models/{args.model}.pth")


if __name__ == "__main__":
    main()