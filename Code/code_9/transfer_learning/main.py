"""
主程序入口
"""
import torch
import torch.nn as nn
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_cifar10, visualize_sample_batch
from models import FeatureExtractor, FineTuningModel, GradualUnfreezingModel
from train import (train_feature_extractor, train_fine_tuning_model,
                   train_gradual_unfreezing_model, save_model)
from evaluation import (evaluate_model, plot_confusion_matrix,
                        plot_learning_curves, visualize_predictions)
from config import DEVICE, NUM_EPOCHS, MODELS


def main():
    parser = argparse.ArgumentParser(description='迁移学习实验')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=MODELS, help='预训练模型架构')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='训练轮数')
    parser.add_argument('--use_subset', action='store_true',
                        help='是否使用数据集子集（加速实验）')
    parser.add_argument('--subset_size', type=float, default=0.3,
                        help='数据集子集大小（占比）')
    args = parser.parse_args()

    print("=" * 50)
    print(f"迁移学习实验 - 基于{args.model}的CIFAR-10分类")
    print(f"设备: {DEVICE}")
    print(f"训练轮数: {args.epochs}")
    if args.use_subset:
        print(f"使用数据集子集: {args.subset_size * 100:.0f}%数据")
    print("=" * 50)

    # 加载数据
    print("\n1. 加载CIFAR-10数据集...")
    trainloader, testloader, classes = load_cifar10(use_subset=args.use_subset,
                                                    subset_size=args.subset_size)

    # 计算数据集大小
    dataset_sizes = {
        'train': len(trainloader.dataset),
        'val': len(testloader.dataset)
    }

    # 构建数据加载器字典
    dataloaders = {
        'train': trainloader,
        'val': testloader
    }

    # 可视化样本数据
    print("\n2. 可视化样本数据...")
    visualize_sample_batch(trainloader, classes)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    print("\n3. 开始迁移学习实验...")
    # 实验1: 特征提取
    print("\n实验1: 特征提取（仅训练分类器）")
    feature_extractor = FeatureExtractor(args.model, num_classes=len(classes))
    trained_feature_extractor, history_feature_ext = train_feature_extractor(
        feature_extractor, dataloaders, dataset_sizes, num_epochs=args.epochs
    )
    save_model(trained_feature_extractor, args.model, "feature_extractor")

    # 实验2: 微调
    print("\n实验2: 微调（调整预训练层与分类器）")
    fine_tuning_model = FineTuningModel(args.model, num_classes=len(classes))
    trained_fine_tuning_model, history_fine_tune = train_fine_tuning_model(
        fine_tuning_model, dataloaders, dataset_sizes, num_epochs=args.epochs
    )
    save_model(trained_fine_tuning_model, args.model, "fine_tuning")

    # 实验3: 渐进式解冻
    print("\n实验3: 渐进式解冻（逐步解冻模型层）")
    gradual_unfreezing_model = GradualUnfreezingModel(args.model, num_classes=len(classes))
    trained_gradual_model, history_gradual = train_gradual_unfreezing_model(
        gradual_unfreezing_model, dataloaders, dataset_sizes, num_epochs=args.epochs
    )
    save_model(trained_gradual_model, args.model, "gradual_unfreezing")

    # 绘制学习曲线比较
    print("\n4. 绘制学习曲线比较...")
    histories = [history_feature_ext, history_fine_tune, history_gradual]
    labels = ['特征提取', '微调', '渐进式解冻']
    plot_learning_curves(histories, labels, title=f"{args.model} 学习曲线对比")

    # 评估模型
    print("\n5. 评估模型性能...")

    # 评估特征提取模型
    print("\n特征提取模型评估:")
    acc_ft, loss_ft, cm_ft, report_ft, _, _ = evaluate_model(
        trained_feature_extractor, testloader, criterion, classes
    )
    print(f"准确率: {acc_ft:.4f}, 损失: {loss_ft:.4f}")
    print(f"分类报告:\n{report_ft}")
    plot_confusion_matrix(cm_ft, classes, title=f"{args.model} 特征提取 - 混淆矩阵")

    # 评估微调模型
    print("\n微调模型评估:")
    acc_finetuning, loss_finetuning, cm_finetuning, report_finetuning, _, _ = evaluate_model(
        trained_fine_tuning_model, testloader, criterion, classes
    )
    print(f"准确率: {acc_finetuning:.4f}, 损失: {loss_finetuning:.4f}")
    print(f"分类报告:\n{report_finetuning}")
    plot_confusion_matrix(cm_finetuning, classes, title=f"{args.model} 微调 - 混淆矩阵")

    # 评估渐进式解冻模型
    print("\n渐进式解冻模型评估:")
    acc_gradual, loss_gradual, cm_gradual, report_gradual, _, _ = evaluate_model(
        trained_gradual_model, testloader, criterion, classes
    )
    print(f"准确率: {acc_gradual:.4f}, 损失: {loss_gradual:.4f}")
    print(f"分类报告:\n{report_gradual}")
    plot_confusion_matrix(cm_gradual, classes, title=f"{args.model} 渐进式解冻 - 混淆矩阵")

    # 可视化预测结果
    print("\n6. 可视化模型预测...")
    print("\n特征提取模型预测可视化:")
    visualize_predictions(trained_feature_extractor, testloader, classes)

    print("\n微调模型预测可视化:")
    visualize_predictions(trained_fine_tuning_model, testloader, classes)

    print("\n渐进式解冻模型预测可视化:")
    visualize_predictions(trained_gradual_model, testloader, classes)

    # 显示错误预测样例
    print("\n7. 显示错误预测样例...")
    print("\n特征提取模型错误预测:")
    visualize_predictions(trained_feature_extractor, testloader, classes, incorrect_only=True)

    # 输出结果总结
    print("\n8. 实验结果总结:")
    print(f"\n模型架构: {args.model}")
    print(f"训练轮数: {args.epochs}")
    print("\n准确率比较:")
    print(f"特征提取模型: {acc_ft:.4f}")
    print(f"微调模型: {acc_finetuning:.4f}")
    print(f"渐进式解冻模型: {acc_gradual:.4f}")

    print("\n实验完成！")


if __name__ == "__main__":
    main()