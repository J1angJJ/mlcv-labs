import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import os

from model import BasicAutoencoder, RegularizedAutoencoder, DenoisingAutoencoder, VariationalAutoencoder, \
    ConvAutoencoder
from train import load_data, device

# 设置matplotlib中文显示 - 仅使用黑体
matplotlib.rcParams['font.family'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def visualize_reconstruction(model, test_loader, model_type="basic", n=10):
    """可视化原始图像和重建图像"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:n].to(device)

        if model_type == "basic" or model_type == "regularized" or model_type == "conv":
            recon, _ = model(data)
        elif model_type == "denoising":
            recon, _, corrupted = model(data)
        elif model_type == "vae":
            recon, _, _ = model(data)

        # 创建图像网格
        plt.figure(figsize=(12, 5))
        for i in range(n):
            # 显示原始图像
            ax = plt.subplot(3 if model_type == "denoising" else 2, n, i + 1)
            if data.size(1) == 1:  # 灰度图像
                plt.imshow(data[i].cpu().numpy().reshape(data.size(2), data.size(3)), cmap='gray')
            else:  # 彩色图像
                plt.imshow(np.transpose(data[i].cpu().numpy(), (1, 2, 0)))
            plt.title("原始图像")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if model_type == "denoising":
                # 显示加噪声的图像
                ax = plt.subplot(3, n, i + 1 + n)
                if corrupted.size(1) == 1:  # 灰度图像
                    plt.imshow(corrupted[i].cpu().numpy().reshape(corrupted.size(2), corrupted.size(3)), cmap='gray')
                else:  # 彩色图像
                    plt.imshow(np.transpose(corrupted[i].cpu().numpy(), (1, 2, 0)))
                plt.title("噪声图像")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # 显示重建图像
                ax = plt.subplot(3, n, i + 1 + 2 * n)
            else:
                # 显示重建图像
                ax = plt.subplot(2, n, i + 1 + n)

            if recon.size(1) == 1:  # 灰度图像
                plt.imshow(recon[i].cpu().numpy().reshape(recon.size(2), recon.size(3)), cmap='gray')
            else:  # 彩色图像
                plt.imshow(np.transpose(recon[i].cpu().numpy(), (1, 2, 0)))
            plt.title("重建图像")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()

        # 保存图像
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{model_type}_reconstruction.png")
        plt.show()


def visualize_latent_space(model, test_loader, model_type="basic", n=1000):
    """可视化潜在空间"""
    model.eval()
    encoded_data = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)

            if model_type == "basic" or model_type == "regularized" or model_type == "conv":
                _, encoded = model(data)
                encoded_data.append(encoded.cpu().numpy())
            elif model_type == "denoising":
                _, encoded, _ = model(data)
                encoded_data.append(encoded.cpu().numpy())
            elif model_type == "vae":
                mu, _ = model.encode(data)
                encoded_data.append(mu.cpu().numpy())

            labels.append(target.numpy())

            if batch_idx * len(data) >= n:
                break

    encoded_data = np.vstack(encoded_data)
    labels = np.concatenate(labels)

    # 使用t-SNE进行降维可视化（如果潜在空间维度大于2）
    if encoded_data.shape[1] > 2:
        print("使用t-SNE进行降维...")
        encoded_data = TSNE(n_components=2).fit_transform(encoded_data[:n])
    else:
        encoded_data = encoded_data[:n]

    labels = labels[:n]

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, alpha=0.5, cmap='viridis')
    plt.colorbar(scatter, label='类别')
    plt.title(f"{model_type}自编码器的潜在空间")
    plt.xlabel("维度1")
    plt.ylabel("维度2")

    # 保存图像
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_type}_latent_space.png")
    plt.show()


def generate_from_latent(model, model_type="vae", n=20):
    """从潜在空间生成新图像（仅适用于VAE）"""
    if model_type != "vae":
        print("仅VAE模型支持从潜在空间生成图像")
        return

    model.eval()
    with torch.no_grad():
        # 从正态分布中采样潜在向量
        z = torch.randn(n, 20).to(device)

        # 从潜在向量生成图像
        gen_imgs = model.decode(z)

        # 显示生成的图像
        plt.figure(figsize=(15, 3))
        for i in range(n):
            ax = plt.subplot(2, n // 2, i + 1)
            if gen_imgs.size(1) == 1:  # 灰度图像
                plt.imshow(gen_imgs[i].cpu().numpy().reshape(gen_imgs.size(2), gen_imgs.size(3)), cmap='gray')
            else:  # 彩色图像
                plt.imshow(np.transpose(gen_imgs[i].cpu().numpy(), (1, 2, 0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()

        # 保存图像
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{model_type}_generation.png")
        plt.show()


def interpolate_in_latent_space(model, test_loader, model_type="vae", steps=10):
    """在潜在空间中进行插值（仅适用于VAE）"""
    if model_type != "vae":
        print("仅VAE模型支持在潜在空间中进行插值")
        return

    model.eval()
    with torch.no_grad():
        # 获取两个不同的数字
        data, labels = next(iter(test_loader))
        data = data.to(device)
        img1 = data[0:1]  # 第一个图像
        img2 = data[1:2]  # 第二个图像

        # 编码到潜在空间
        z1, _ = model.encode(img1)
        z2, _ = model.encode(img2)

        # 在潜在空间中插值
        z_interp = torch.zeros(steps, z1.size(1)).to(device)
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp[i] = z1 * (1 - alpha) + z2 * alpha

        # 从插值的潜在向量生成图像
        gen_imgs = model.decode(z_interp)

        # 显示原始图像和插值结果
        plt.figure(figsize=(15, 3))

        # 显示第一个原始图像
        ax = plt.subplot(1, steps + 2, 1)
        if img1.size(1) == 1:  # 灰度图像
            plt.imshow(img1.cpu().numpy().reshape(img1.size(2), img1.size(3)), cmap='gray')
        else:  # 彩色图像
            plt.imshow(np.transpose(img1.cpu().numpy()[0], (1, 2, 0)))
        plt.title(f"原始 {labels[0].item()}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 显示插值结果
        for i in range(steps):
            ax = plt.subplot(1, steps + 2, i + 2)
            if gen_imgs.size(1) == 1:  # 灰度图像
                plt.imshow(gen_imgs[i].cpu().numpy().reshape(gen_imgs.size(2), gen_imgs.size(3)), cmap='gray')
            else:  # 彩色图像
                plt.imshow(np.transpose(gen_imgs[i].cpu().numpy(), (1, 2, 0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # 显示第二个原始图像
        ax = plt.subplot(1, steps + 2, steps + 2)
        if img2.size(1) == 1:  # 灰度图像
            plt.imshow(img2.cpu().numpy().reshape(img2.size(2), img2.size(3)), cmap='gray')
        else:  # 彩色图像
            plt.imshow(np.transpose(img2.cpu().numpy()[0], (1, 2, 0)))
        plt.title(f"原始 {labels[1].item()}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()

        # 保存图像
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{model_type}_interpolation.png")
        plt.show()


def visualize_training_process(model_type="basic", dataset="mnist"):
    """可视化训练过程中的损失变化"""
    # 加载训练记录
    checkpoint = torch.load(f"models/{model_type}_{dataset}_autoencoder.pth")
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='测试损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失')
    plt.title(f"{model_type}自编码器训练过程")
    plt.legend()
    plt.grid(True)

    # 保存图像
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_type}_{dataset}_training_process.png")
    plt.show()


def load_model(model_type="basic", dataset="mnist"):
    """加载训练好的模型"""
    # 加载模型参数
    model_path = f"models/{model_type}_{dataset}_autoencoder.pth"
    checkpoint = torch.load(model_path)
    encoding_dim = checkpoint['encoding_dim']
    input_shape = checkpoint['input_shape']

    # 创建模型
    if model_type == "basic":
        model = BasicAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    elif model_type == "regularized":
        model = RegularizedAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    elif model_type == "denoising":
        noise_factor = checkpoint['noise_factor']
        model = DenoisingAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim, noise_factor=noise_factor).to(
            device)
    elif model_type == "vae":
        model = VariationalAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    elif model_type == "conv":
        model = ConvAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    else:
        raise ValueError("不支持的模型类型")

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def visualize_model(model_type="basic", dataset="mnist"):
    """对指定类型的模型进行可视化"""
    print(f"\n\n=== 可视化 {model_type} 自编码器 (数据集: {dataset}) ===")

    # 加载数据
    _, test_loader, _ = load_data(dataset)

    # 加载模型
    try:
        model = load_model(model_type, dataset)
        print(f"成功加载{model_type}模型")

        # 可视化训练过程
        visualize_training_process(model_type, dataset)

        # 重建可视化
        visualize_reconstruction(model, test_loader, model_type)

        # 潜在空间可视化
        visualize_latent_space(model, test_loader, model_type)

        # VAE特有功能
        if model_type == "vae":
            generate_from_latent(model, model_type)
            interpolate_in_latent_space(model, test_loader, model_type)

    except Exception as e:
        print(f"加载或可视化{model_type}模型时出错: {e}")


if __name__ == "__main__":
    print("开始自编码器可视化...")

    # 定义模型类型和数据集
    dataset = "mnist"  # 或 "cifar10"

    # 可视化各种类型的自编码器
    model_types = [
        "basic",  # 基本自编码器
        "basic",  # 欠完备自编码器(实际上使用basic模型，但encoding_dim不同)
        "regularized",  # 正则化自编码器
        "denoising",  # 去噪自编码器
        "vae"  # 变分自编码器
    ]

    for model_type in model_types:
        visualize_model(model_type, dataset)

    print("可视化完成！")