# cifar10_main.py - 主函数

import torch
import os
import numpy as np
from cifar10_data import load_cifar10_data, show_images
from cifar10_cnn import create_cnn_model
from cifar10_resnet import create_resnet_model
from cifar10_train import run_experiment
from chinese_font import set_chinese_font


# 设置随机种子以确保结果可重现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 主函数
def main():
    # 设置随机种子
    set_seed(42)

    # 设置中文字体
    set_chinese_font()

    # 加载数据
    data_loaders = load_cifar10_data(batch_size=128)
    train_loader, test_loader, classes = data_loaders

    # 显示样本图像
    show_images(train_loader, classes)

    # 运行CNN实验
    cnn_history, cnn_model_path = run_experiment(
        model_type='BasicCNN',
        model_creator=lambda: create_cnn_model(),
        data_loaders=data_loaders,
        num_epochs=100,  # 为了演示减少轮次
        save_dir='./models'
    )

    # 运行ResNet实验
    resnet_history, resnet_model_path = run_experiment(
        model_type='ResNet20',
        model_creator=lambda: create_resnet_model('resnet20'),
        data_loaders=data_loaders,
        num_epochs=100,  # 为了演示减少轮次
        save_dir='./models'
    )

    print("\n==== 实验完成 ====")
    print(f"CNN最佳模型保存在: {cnn_model_path}")
    print(f"ResNet最佳模型保存在: {resnet_model_path}")


if __name__ == "__main__":
    main()