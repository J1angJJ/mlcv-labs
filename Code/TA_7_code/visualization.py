# visualization.py

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置环境变量解决OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def set_chinese_font():
    """设置matplotlib的中文字体"""
    try:
        # 尝试设置黑体字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        print("成功设置中文字体为黑体")
    except Exception as e:
        print(f"设置中文字体时出现错误: {e}")
        print("将使用系统默认字体")


def visualize_prediction(model, val_loader, device, num_samples=3, save_path='pet_prediction_results.png'):
    """
    可视化分割预测结果

    参数:
        model: 训练好的UNet模型
        val_loader: 验证数据加载器
        device: 使用的设备 (CPU/GPU)
        num_samples: 要可视化的样本数量
        save_path: 结果图像保存路径
    """
    # 设置中文字体
    set_chinese_font()

    model.eval()

    # 获取一些样本
    dataiter = iter(val_loader)
    images, masks = next(dataiter)
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # 转回CPU并转换为numpy数组
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    images = images.transpose(0, 2, 3, 1)
    for i in range(num_samples):
        images[i] = images[i] * std + mean
        images[i] = np.clip(images[i], 0, 1)

    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab10', 3)  # 3个类别

    # 绘制结果
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, -1)

    for i in range(num_samples):
        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title('输入图像')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(masks[i], cmap=cmap, vmin=0, vmax=2)
        axs[i, 1].set_title('真实标签')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(preds[i], cmap=cmap, vmin=0, vmax=2)
        axs[i, 2].set_title('预测结果')
        axs[i, 2].axis('off')

    plt.tight_layout()

    # 保存并显示结果
    plt.savefig(save_path)
    abs_path = os.path.abspath(save_path)
    print(f"分割预测结果图像已保存至: {abs_path}")

    try:
        plt.show()
        print("图像已显示")
    except Exception as e:
        print(f"无法显示图像: {e}")
        print(f"但图像已成功保存到: {abs_path}")
        print("您可以手动打开保存的图像文件查看结果")


if __name__ == "__main__":
    # 导入必要的模块
    from data_processing import get_data_loaders
    from model_training import UNet

    # 获取数据加载器
    _, val_loader = get_data_loaders()

    # 检查模型文件是否存在
    model_path = 'best_pet_unet_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        print("请先运行 model_training.py 训练模型")
    else:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')

        # 加载模型
        model = UNet(n_channels=3, n_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # 可视化预测结果
        print("生成分割预测可视化...")
        visualize_prediction(model, val_loader, device)