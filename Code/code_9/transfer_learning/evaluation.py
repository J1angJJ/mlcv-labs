"""
模型评估与可视化模块
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from config import DEVICE, RESULT_DIR

# 设置matplotlib使用黑体字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 对于中文使用黑体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def evaluate_model(model, test_loader, criterion, classes):
    """评估模型"""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算总体指标
    accuracy = correct / total
    avg_loss = total_loss / total

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算分类报告
    report = classification_report(all_labels, all_preds, target_names=classes)

    return accuracy, avg_loss, cm, report, all_preds, all_labels

def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.Blues, normalize=False):
    """绘制混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图像
    if title:
        filename = f"{title.replace(' ', '_')}.png"
        filepath = os.path.join(RESULT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

def plot_learning_curves(histories, labels, title=None):
    """绘制学习曲线对比"""
    plt.figure(figsize=(15, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    for i, (history, label) in enumerate(zip(histories, labels)):
        plt.plot(history['train_acc'], linestyle='-', label=f'{label} (训练)')
        plt.plot(history['val_acc'], linestyle='--', label=f'{label} (验证)')

    plt.title('模型准确率', fontsize=14, fontweight='bold')
    plt.ylabel('准确率', fontsize=12, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=12, fontweight='bold')
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    for i, (history, label) in enumerate(zip(histories, labels)):
        plt.plot(history['train_loss'], linestyle='-', label=f'{label} (训练)')
        plt.plot(history['val_loss'], linestyle='--', label=f'{label} (验证)')

    plt.title('模型损失', fontsize=14, fontweight='bold')
    plt.ylabel('损失', fontsize=12, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=12, fontweight='bold')
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 保存图像
    if title:
        filename = f"{title.replace(' ', '_')}.png"
        filepath = os.path.join(RESULT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

def visualize_predictions(model, test_loader, classes, num_images=12, incorrect_only=False):
    """可视化模型预测结果"""
    model.eval()

    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if incorrect_only and preds[j] == labels[j]:
                    continue

                images_so_far += 1

                # 逆标准化图像
                img = inputs.cpu()[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2470, 0.2435, 0.2616])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                # 显示图像
                ax = plt.subplot(3, 4, images_so_far)
                ax.axis('off')
                title = f'预测: {classes[preds[j]]}\n实际: {classes[labels[j]]}'
                ax.set_title(title, fontsize=11, fontweight='bold',
                            color='green' if preds[j] == labels[j] else 'red')
                plt.imshow(img)

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    return

        plt.tight_layout()
        plt.show()

def visualize_feature_maps(model, test_loader, layer_name, num_filters=16):
    """可视化模型的特征图"""
    # 这是一个更高级的功能，需要进一步实现
    # 留作拓展练习
    pass