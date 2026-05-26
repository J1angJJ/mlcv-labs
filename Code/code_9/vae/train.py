import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from model import BasicAutoencoder, RegularizedAutoencoder, DenoisingAutoencoder, VariationalAutoencoder, \
    ConvAutoencoder

# 解决OpenMP重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(dataset='mnist', batch_size=128):
    """加载数据集，支持MNIST和CIFAR-10"""
    if dataset.lower() == 'mnist':
        # MNIST数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_shape = (1, 28, 28)  # 通道，高，宽

    elif dataset.lower() == 'cifar10':
        # CIFAR-10数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_shape = (3, 32, 32)  # 通道，高，宽

    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, input_shape


def train_basic_autoencoder(model, train_loader, optimizer, epoch):
    """训练基本自编码器"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(data)
        loss = F.mse_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'====> Epoch: {epoch} 平均损失: {train_loss / len(train_loader):.4f}')
    return train_loss / len(train_loader)


def train_regularized_autoencoder(model, train_loader, optimizer, epoch, lambda_l1=1e-5):
    """训练带L1正则化的自编码器"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, encoded = model(data)

        # 重建损失 + L1正则化
        mse_loss = F.mse_loss(recon_batch, data)
        l1_loss = torch.mean(torch.abs(encoded))
        loss = mse_loss + lambda_l1 * l1_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'====> Epoch: {epoch} 平均损失: {train_loss / len(train_loader):.4f}')
    return train_loss / len(train_loader)


def train_denoising_autoencoder(model, train_loader, optimizer, epoch):
    """训练去噪自编码器"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, _, _ = model(data)
        loss = F.mse_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'====> Epoch: {epoch} 平均损失: {train_loss / len(train_loader):.4f}')
    return train_loss / len(train_loader)


def train_vae(model, train_loader, optimizer, epoch):
    """训练变分自编码器"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)

        # VAE损失 = 重建损失 + KL散度
        recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kld_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'====> Epoch: {epoch} 平均损失: {train_loss / len(train_loader):.4f}')
    return train_loss / len(train_loader)


def train_conv_autoencoder(model, train_loader, optimizer, epoch):
    """训练卷积自编码器"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(data)
        loss = F.mse_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'====> Epoch: {epoch} 平均损失: {train_loss / len(train_loader):.4f}')
    return train_loss / len(train_loader)


def test_model(model, test_loader, model_type="basic"):
    """测试模型性能"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            if model_type == "basic" or model_type == "regularized" or model_type == "conv":
                recon, _ = model(data)
                test_loss += F.mse_loss(recon, data).item()
            elif model_type == "denoising":
                recon, _, _ = model(data)
                test_loss += F.mse_loss(recon, data).item()
            elif model_type == "vae":
                recon, mu, log_var = model(data)
                recon_loss = F.binary_cross_entropy(recon, data, reduction='sum').item()
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).item()
                test_loss += recon_loss + kld_loss

    test_loss /= len(test_loader.dataset)
    print(f'测试集损失: {test_loss:.4f}')
    return test_loss


def train_model(model_type="basic", dataset="mnist", encoding_dim=128, epochs=3, noise_factor=0.3, lambda_l1=1e-4):
    """训练指定类型的自编码器模型"""
    print(f"\n\n=== 训练 {model_type} 自编码器（数据集: {dataset}）===")
    start_time = time.time()

    # 加载数据
    train_loader, test_loader, input_shape = load_data(dataset)

    # 创建模型
    if model_type == "basic":
        model = BasicAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    elif model_type == "regularized":
        model = RegularizedAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    elif model_type == "denoising":
        model = DenoisingAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim, noise_factor=noise_factor).to(
            device)
    elif model_type == "vae":
        model = VariationalAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    elif model_type == "conv":
        model = ConvAutoencoder(input_shape=input_shape, encoding_dim=encoding_dim).to(device)
    else:
        raise ValueError("不支持的模型类型")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 记录训练过程中的损失
    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        # 训练
        if model_type == "basic":
            train_loss = train_basic_autoencoder(model, train_loader, optimizer, epoch)
        elif model_type == "regularized":
            train_loss = train_regularized_autoencoder(model, train_loader, optimizer, epoch, lambda_l1)
        elif model_type == "denoising":
            train_loss = train_denoising_autoencoder(model, train_loader, optimizer, epoch)
        elif model_type == "vae":
            train_loss = train_vae(model, train_loader, optimizer, epoch)
        elif model_type == "conv":
            train_loss = train_conv_autoencoder(model, train_loader, optimizer, epoch)

        train_losses.append(train_loss)

        # 测试
        test_loss = test_model(model, test_loader, model_type)
        test_losses.append(test_loss)

    print(f"{model_type}自编码器训练耗时: {time.time() - start_time:.2f} 秒")

    # 保存模型
    os.makedirs("models", exist_ok=True)
    model_filename = f"{model_type}_{dataset}_autoencoder.pth"
    save_path = os.path.join("models", model_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'encoding_dim': encoding_dim,
        'input_shape': input_shape,
        'noise_factor': noise_factor if model_type == "denoising" else None,
        'lambda_l1': lambda_l1 if model_type == "regularized" else None,
        'dataset': dataset
    }, save_path)

    print(f"模型已保存至 {save_path}")

    return model, train_losses, test_losses


if __name__ == "__main__":
    print("开始自编码器训练...")
    print(f"使用设备: {device}")

    # 训练各种类型的自编码器
    epochs = 3  # 实验课中的训练轮数
    dataset = "mnist"  # 默认数据集

    # 1. 基本自编码器
    train_model("basic", dataset, encoding_dim=128, epochs=epochs)

    # 2. 欠完备自编码器
    train_model("basic", dataset, encoding_dim=32, epochs=epochs)

    # 3. 正则化自编码器
    train_model("regularized", dataset, encoding_dim=128, epochs=epochs, lambda_l1=1e-4)

    # 4. 去噪自编码器
    train_model("denoising", dataset, encoding_dim=128, epochs=epochs, noise_factor=0.3)

    # 5. 变分自编码器
    train_model("vae", dataset, encoding_dim=20, epochs=epochs)

    print("训练完成！")