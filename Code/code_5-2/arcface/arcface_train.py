# arcface_train.py - 训练与评估

import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from arcface_model import FeatureExtractor, ArcFaceClassifier, SoftmaxClassifier
from arcface_data import load_data
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 导入中文字体设置
from chinese_font import set_chinese_font
set_chinese_font()  # 设置中文显示

def train_model(feature_extractor, classifier, train_loader, criterion, optimizer, device, epochs=10):
    """训练模型函数"""
    feature_extractor.train()
    classifier.train()

    # 记录训练过程
    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            features = feature_extractor(inputs)
            outputs = classifier(features, labels)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练时间、平均损失和准确率
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

        print(f'轮次 {epoch + 1}/{epochs}, 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%, 用时: {epoch_time:.2f}秒')

    return history


def evaluate_model(feature_extractor, classifier, test_loader, device):
    """评估模型函数"""
    feature_extractor.eval()
    classifier.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 提取特征并分类
            features = feature_extractor(inputs)
            outputs = classifier(features)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)

            # 统计准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy, np.array(all_labels), np.array(all_preds)


def plot_training_history(softmax_history, arcface_history, epochs):
    """绘制训练历史对比图"""
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)

    # 绘制损失对比
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), softmax_history['loss'], 'o-', label='Softmax')
    plt.plot(range(1, epochs + 1), arcface_history['loss'], 'o-', label='ArcFace')
    plt.title('训练损失对比')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # 绘制准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), softmax_history['acc'], 'o-', label='Softmax')
    plt.plot(range(1, epochs + 1), arcface_history['acc'], 'o-', label='ArcFace')
    plt.title('训练准确率对比')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./results/training_history.png')
    plt.show()


def plot_confusion_matrix(labels, predictions, class_names, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    # 归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    data_path = "./data/lfw_subset"  # LFW子集路径
    batch_size = 32
    train_loader, test_loader, num_classes = load_data(data_path, batch_size)

    # 获取类别名称
    dataset = train_loader.dataset.dataset  # 获取底层数据集
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        class_names = [f"Person_{i}" for i in range(num_classes)]

    # 创建实验结果目录
    os.makedirs('./results', exist_ok=True)

    # 设置模型参数
    embedding_size = 512
    epochs = 10

    # 训练Softmax模型
    print("\n========== 训练Softmax模型 ==========")
    feature_extractor = FeatureExtractor(embedding_size).to(device)
    softmax_classifier = SoftmaxClassifier(embedding_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(softmax_classifier.parameters()),
        lr=0.001
    )

    softmax_history = train_model(feature_extractor, softmax_classifier, train_loader,
                                  criterion, optimizer, device, epochs=epochs)
    softmax_acc, softmax_labels, softmax_preds = evaluate_model(
        feature_extractor, softmax_classifier, test_loader, device)

    # 保存Softmax模型
    torch.save(feature_extractor.state_dict(), './results/softmax_feature_extractor.pth')
    torch.save(softmax_classifier.state_dict(), './results/softmax_classifier.pth')

    # 重新初始化特征提取器
    feature_extractor = FeatureExtractor(embedding_size).to(device)

    # 训练ArcFace模型
    print("\n========== 训练ArcFace模型 ==========")
    arcface_classifier = ArcFaceClassifier(embedding_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(arcface_classifier.parameters()),
        lr=0.001
    )

    arcface_history = train_model(feature_extractor, arcface_classifier, train_loader,
                                  criterion, optimizer, device, epochs=epochs)
    arcface_acc, arcface_labels, arcface_preds = evaluate_model(
        feature_extractor, arcface_classifier, test_loader, device)

    # 保存ArcFace模型
    torch.save(feature_extractor.state_dict(), './results/arcface_feature_extractor.pth')
    torch.save(arcface_classifier.state_dict(), './results/arcface_classifier.pth')

    # 绘制训练历史对比
    plot_training_history(softmax_history, arcface_history, epochs)

    # 绘制混淆矩阵
    plot_confusion_matrix(softmax_labels, softmax_preds,
                          class_names, 'Softmax混淆矩阵',
                          './results/softmax_confusion_matrix.png')

    plot_confusion_matrix(arcface_labels, arcface_preds,
                          class_names, 'ArcFace混淆矩阵',
                          './results/arcface_confusion_matrix.png')

    # 输出对比结果
    print(f"\n========== 结果对比 ==========")
    print(f"Softmax准确率: {softmax_acc:.2f}%")
    print(f"ArcFace准确率: {arcface_acc:.2f}%")
    print(f"ArcFace相对提升: {(arcface_acc - softmax_acc):.2f}%")

    # 保存结果
    with open('./results/experiment_results.txt', 'w') as f:
        f.write(f"数据集: LFW子集\n")
        f.write(f"类别数: {num_classes}\n")
        f.write(f"Softmax准确率: {softmax_acc:.2f}%\n")
        f.write(f"ArcFace准确率: {arcface_acc:.2f}%\n")
        f.write(f"相对提升: {(arcface_acc - softmax_acc):.2f}%\n")


if __name__ == "__main__":
    main()