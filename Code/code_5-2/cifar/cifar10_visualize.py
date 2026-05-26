# cifar10_visualize.py - 模型可视化与解释

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import os
import seaborn as sns
import argparse
from chinese_font import set_chinese_font

# 尝试导入其他模块
try:
    from cifar10_data import load_cifar10_data
    from cifar10_cnn import create_cnn_model
    from cifar10_resnet import create_resnet_model
except ImportError:
    print("警告: 无法导入一些模块，在单独运行可视化时可能需要它们")


def show_predictions(model, data_loader, classes, device, num_samples=25, save_path='predictions.png'):
    """显示模型预测结果"""
    # 设置中文字体
    set_chinese_font()

    # 将模型设为评估模式
    model.eval()

    # CIFAR-10数据集均值和标准差
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    # 获取一批图像和标签
    images, labels = next(iter(data_loader))

    # 限制样本数量
    images = images[:num_samples]
    labels = labels[:num_samples]

    # 将图像移到设备上进行预测
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # 将预测结果转移到CPU
    predicted = predicted.cpu().numpy()

    # 计算每行和每列的样本数
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    # 创建图形
    plt.figure(figsize=(15, 12))

    # 显示图像和预测结果
    for i, (img, label, pred) in enumerate(zip(images, labels, predicted)):
        if i >= num_samples:
            break

        # 反标准化图像
        img = img.numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # 添加子图
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)

        # 设置标题颜色，正确为绿色，错误为红色
        title_color = 'green' if label == pred else 'red'
        plt.title(f"真实: {classes[label]}\n预测: {classes[pred]}",
                  color=title_color)
        plt.axis('off')

    # 保存图形
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"预测可视化已保存到 {save_path}")
    return save_path


def plot_confusion_matrix(all_preds, all_targets, classes, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    # 设置中文字体
    set_chinese_font()

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    # 设置标题和标签
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 保存图形
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")
    return save_path


def plot_class_accuracies(all_preds, all_targets, classes, save_path='class_accuracies.png'):
    """绘制每个类别的准确率"""
    # 设置中文字体
    set_chinese_font()

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(len(classes)):
        # 获取当前类别的样本索引
        idx = all_targets == i
        # 如果有该类别的样本
        if np.sum(idx) > 0:
            # 计算该类别的准确率
            acc = np.mean(all_preds[idx] == i) * 100
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)

    # 创建柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), class_accuracies, color='skyblue')

    # 添加数据标签
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom')

    # 设置标题和标签
    plt.title('各类别准确率')
    plt.xlabel('类别')
    plt.ylabel('准确率 (%)')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim(0, 105)  # 设置y轴限制，给数据标签留出空间

    # 添加平均准确率线
    avg_acc = np.mean(class_accuracies)
    plt.axhline(y=avg_acc, color='r', linestyle='-',
                label=f'平均准确率: {avg_acc:.2f}%')
    plt.legend()

    # 保存图形
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"类别准确率图已保存到 {save_path}")
    return save_path


def plot_tsne(model, data_loader, device, num_samples=1000, save_path='tsne_visualization.png'):
    """使用t-SNE可视化特征"""
    # 设置中文字体
    set_chinese_font()

    # 将模型设为评估模式
    model.eval()

    # 存储特征和标签
    features = []
    labels = []
    count = 0

    # 获取特征
    with torch.no_grad():
        for images, targets in data_loader:
            # 如果已收集足够样本，则停止
            if count >= num_samples:
                break

            # 计算当前批次要取的样本数量
            batch_size = images.size(0)
            samples_to_take = min(batch_size, num_samples - count)

            # 提取特征
            batch_images = images[:samples_to_take].to(device)
            batch_targets = targets[:samples_to_take]

            try:
                # 尝试使用模型的特征提取方法
                batch_features = model.get_features(batch_images)
            except (AttributeError, NotImplementedError):
                # 如果没有特定方法，则使用完整前向传播
                # 这不是一个好方法，但在这里作为备选
                print("警告: 模型没有get_features方法，尝试使用完整前向传播")
                outputs = model(batch_images)
                batch_features = outputs  # 这可能不正确，取决于模型结构

            # 存储特征和标签
            features.append(batch_features.cpu().numpy())
            labels.append(batch_targets.numpy())

            # 更新计数
            count += samples_to_take

    if not features:
        print("错误: 未能收集到特征")
        return None

    # 连接所有特征和标签
    features = np.vstack(features)
    labels = np.concatenate(labels)

    print(f"收集了{len(features)}个样本的特征，形状为{features.shape}")

    # 使用t-SNE降维
    print("正在应用t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 绘制t-SNE散点图
    plt.figure(figsize=(12, 10))

    # 为每个类别使用不同的颜色
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(
            features_2d[labels == label, 0],
            features_2d[labels == label, 1],
            label=f'类别 {label}',
            alpha=0.6
        )

    plt.title('t-SNE特征可视化')
    plt.legend()

    # 保存图形
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"t-SNE可视化已保存到 {save_path}")
    return save_path


def visualize_model(model_path, model_type='resnet20', output_dir='./visualizations'):
    """加载模型并执行可视化"""
    # 设置输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    _, test_loader, classes = load_cifar10_data(batch_size=64)
    print("测试数据加载完成")

    # 创建模型
    if model_type == 'cnn':
        model = create_cnn_model()
    else:
        model, _ = create_resnet_model(model_type)

    model = model.to(device)
    print(f"已创建{model_type}模型")

    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 如果保存的是整个checkpoint（包括epoch，optimizer等）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从checkpoint加载模型，轮次: {checkpoint.get('epoch', 'unknown')}")
            if 'test_acc' in checkpoint:
                print(f"测试准确率: {checkpoint['test_acc']:.2f}%")
        # 如果只保存了模型参数
        else:
            model.load_state_dict(checkpoint)
            print("模型参数加载成功")

        print(f"模型加载完成: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 设置为评估模式
    model.eval()

    # 执行预测获取结果
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算准确率
    accuracy = np.mean(all_preds == all_targets) * 100
    print(f"模型在测试集上的准确率: {accuracy:.2f}%")

    # 生成分类报告
    report = classification_report(all_targets, all_preds, target_names=classes)
    print("\n分类报告:")
    print(report)

    # 保存分类报告到文件
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"模型: {model_type}\n")
        f.write(f"准确率: {accuracy:.2f}%\n\n")
        f.write(report)

    # 1. 显示预测样本
    pred_path = show_predictions(
        model, test_loader, classes, device,
        num_samples=16, save_path=os.path.join(output_dir, 'predictions.png'))

    # 2. 绘制混淆矩阵
    cm_path = plot_confusion_matrix(
        all_preds, all_targets, classes,
        save_path=os.path.join(output_dir, 'confusion_matrix.png'))

    # 3. 绘制类别准确率
    acc_path = plot_class_accuracies(
        all_preds, all_targets, classes,
        save_path=os.path.join(output_dir, 'class_accuracies.png'))

    # 4. t-SNE可视化
    tsne_path = plot_tsne(
        model, test_loader, device, num_samples=800,
        save_path=os.path.join(output_dir, 'tsne_visualization.png'))

    print(f"\n所有可视化已保存到: {output_dir}")
    return {
        'predictions': pred_path,
        'confusion_matrix': cm_path,
        'class_accuracies': acc_path,
        'tsne': tsne_path
    }


def main():
    """当作为脚本运行时的主函数"""
    parser = argparse.ArgumentParser(description='CIFAR-10模型可视化')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--model-type', type=str, default='resnet20',
                        choices=['cnn', 'resnet20', 'resnet32'],
                        help='模型类型')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                        help='输出目录')
    parser.add_argument('--samples', type=int, default=800,
                        help='t-SNE可视化使用的样本数量')
    args = parser.parse_args()

    # 验证模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    # 执行可视化
    results = visualize_model(
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir
    )

    if results:
        print("\n可视化完成! 生成的文件:")
        for name, path in results.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()