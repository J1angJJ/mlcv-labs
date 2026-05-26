import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义基本自编码器模型
class BasicAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), encoding_dim=128):
        super(BasicAutoencoder, self).__init__()
        self.input_shape = input_shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # 计算输入维度

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, encoding_dim),
            nn.ReLU(True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 将输出限制在0-1之间
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 将图像展平
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, *self.input_shape), encoded


# 定义带L1正则化的自编码器
class RegularizedAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), encoding_dim=128):
        super(RegularizedAutoencoder, self).__init__()
        self.input_shape = input_shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, encoding_dim),
            nn.ReLU(True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, *self.input_shape), encoded


# 随机自编码器（添加噪声）
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), encoding_dim=128, noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__()
        self.input_shape = input_shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.noise_factor = noise_factor

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, encoding_dim),
            nn.ReLU(True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        corrupted_x = x + noise
        return torch.clamp(corrupted_x, 0., 1.)

    def forward(self, x):
        batch_size = x.size(0)
        original_x = x.view(batch_size, -1)
        corrupted_x = self.add_noise(original_x)
        encoded = self.encoder(corrupted_x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, *self.input_shape), encoded, corrupted_x.view(batch_size, *self.input_shape)


# 变分自编码器(VAE)
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), encoding_dim=20):
        super(VariationalAutoencoder, self).__init__()
        self.input_shape = input_shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True)
        )

        # 均值和方差
        self.fc_mu = nn.Linear(256, encoding_dim)
        self.fc_var = nn.Linear(256, encoding_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return h.view(-1, *self.input_shape)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# 卷积自编码器 - 为CIFAR-10添加的额外模型
class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), encoding_dim=128):
        super(ConvAutoencoder, self).__init__()
        self.input_shape = input_shape

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算编码后的特征图大小
        self.encoded_size = (128, input_shape[1] // 8, input_shape[2] // 8)
        self.fc_encoder = nn.Linear(self.encoded_size[0] * self.encoded_size[1] * self.encoded_size[2], encoding_dim)
        self.fc_decoder = nn.Linear(encoding_dim, self.encoded_size[0] * self.encoded_size[1] * self.encoded_size[2])

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码
        x = self.encoder(x)
        # 展平并进一步编码
        x_flat = x.view(x.size(0), -1)
        encoded = self.fc_encoder(x_flat)
        # 解码
        x_decoded = self.fc_decoder(encoded)
        x_decoded = x_decoded.view(x.size(0), *self.encoded_size)
        decoded = self.decoder(x_decoded)
        return decoded, encoded