# cifar10_train.py - 训练和评估模型

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix
from chinese_font import set_chinese_font

# 引入其他模块
try:
    from cifar10_data import load_cifar10_data
    from cifar10_cnn import create_cnn_model
    from cifar10_resnet import create_resnet_model
except ImportError:
    print("警告: 无法导入一些模块，在单独运行训练时可能需要它们")


def train(model, train_loader, test_loader, optimizer, criterion, device, epochs=200,
          lr_scheduler=None, save_dir='./models', model_name='model'):
    """训练模型"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 初始化训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }

    # 最佳精度和模型路径
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')

    # 开始训练
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n轮次 {epoch + 1}/{epochs}")

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"当前学习率: {current_lr:.6f}")

        # 训练一个轮次
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)

        # 记录训练损失和准确率
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 在测试集上评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)

        # 记录测试损失和准确率
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 打印当前轮次结果
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"测试集 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, best_model_path)
            print(f"保存新的最佳模型，测试准确率: {test_acc:.2f}%")

        # 更新学习率
        if lr_scheduler:
            lr_scheduler.step()

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成！总时间: {total_time / 60:.2f}分钟")
    print(f"最佳测试准确率: {best_acc:.2f}%")

    # 最后一个轮次的模型
    final_model_path = os.path.join(save_dir, f'{model_name}_final.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
    }, final_model_path)
    print(f"最终模型已保存到: {final_model_path}")

    return history, best_model_path


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个轮次"""
    # 将模型设置为训练模式
    model.train()

    # 初始化统计变量
    running_loss = 0.0
    correct = 0
    total = 0

    # 遍历数据批次
    for i, (inputs, targets) in enumerate(train_loader):
        # 将数据移动到指定设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每100个批次打印一次进度
        if (i + 1) % 100 == 0:
            print(f"批次: {i + 1}/{len(train_loader)}, "
                  f"损失: {running_loss / (i + 1):.4f}, "
                  f"准确率: {100. * correct / total:.2f}%")

    # 计算本轮次的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """在测试集上评估模型"""
    # 将模型设置为评估模式
    model.eval()

    # 初始化统计变量
    test_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计损失和准确率
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def predict(model, test_loader, device):
    """使用模型对测试集进行预测"""
    # 将模型设置为评估模式
    model.eval()

    all_preds = []
    all_targets = []

    # 不计算梯度
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 将数据移动到指定设备
            inputs = inputs.to(device)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 收集预测结果和标签
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史曲线"""
    # 设置中文字体
    set_chinese_font()

    # 创建图形
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], 'b-', label='训练损失')
    plt.plot(history['test_loss'], 'r-', label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], 'b-', label='训练准确率')
    plt.plot(history['test_acc'], 'r-', label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)

    # 绘制学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], 'g-')
    plt.title('学习率曲线')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.yscale('log')  # 对数刻度，便于观察学习率变化
    plt.grid(True)

    # 保存图形
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练历史曲线已保存到 {save_path}")
    return save_path


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    # 设置中文字体
    set_chinese_font()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 使用matplotlib绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()

    # 添加类别标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在每个单元格中显示数字
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()

    # 保存图形
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")
    return save_path


def main():
    """当作为脚本运行时的主函数"""
    parser = argparse.ArgumentParser(description='CIFAR-10模型训练')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet20', 'resnet32'], help='模型类型')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, test_loader, classes = load_cifar10_data(batch_size=args.batch_size)

    # 创建模型
    if args.model == 'cnn':
        model = create_cnn_model()
        model_name = 'cnn'
    else:
        model, model_name = create_resnet_model(args.model)

    model = model.to(device)
    print(f"已创建 {model_name} 模型")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # 学习率调度器
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # 训练模型
    print(f"\n开始训练 {model_name} 模型，训练 {args.epochs} 轮...")
    history, best_model_path = train(
        model, train_loader, test_loader, optimizer, criterion, device,
        epochs=args.epochs, lr_scheduler=lr_scheduler,
        save_dir=args.output_dir, model_name=model_name)

    # 绘制训练历史
    history_path = plot_training_history(
        history, os.path.join(args.output_dir, f'{model_name}_history.png'))

    # 加载最佳模型进行预测
    print("\n加载最佳模型进行预测...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 在测试集上进行预测
    y_pred, y_true = predict(model, test_loader, device)

    # 绘制混淆矩阵
    cm_path = plot_confusion_matrix(
        y_true, y_pred, classes, os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png'))

    # 打印最终结果
    print(f"\n{model_name}模型训练完成:")
    print(f"- 最佳测试准确率: {checkpoint['test_acc']:.2f}%")
    print(f"- 模型已保存到: {best_model_path}")
    print(f"- 训练历史图: {history_path}")
    print(f"- 混淆矩阵: {cm_path}")


if __name__ == "__main__":
    main()