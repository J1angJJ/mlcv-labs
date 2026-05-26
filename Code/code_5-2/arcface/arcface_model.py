# arcface_model.py - 模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入中文字体设置
from chinese_font import set_chinese_font
set_chinese_font()  # 设置中文显示

class FeatureExtractor(nn.Module):
    """特征提取网络"""

    def __init__(self, embedding_size=512):
        super(FeatureExtractor, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 第四个卷积块
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # 前向传播
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.dropout(F.relu(self.bn5(self.fc1(x))))
        x = self.bn6(self.fc2(x))

        # L2标准化，使特征向量的长度为1
        return F.normalize(x, p=2, dim=1)


class ArcFaceClassifier(nn.Module):
    """ArcFace分类器"""

    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super(ArcFaceClassifier, self).__init__()
        # 权重参数，每个类别对应一个权重向量
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)  # 初始化权重

        self.scale = scale  # 缩放因子
        self.margin = margin  # 角度间隔

        # 预计算角度间隔的三角函数值
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        # 计算余弦相似度
        cosine = F.linear(embeddings, F.normalize(self.weight))

        if labels is None:  # 预测时直接返回相似度
            return cosine * self.scale

        # 训练时计算带角度间隔的相似度
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m)

        # 处理θ+m > π的情况
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 创建one-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 应用角度间隔：对目标类别使用cos(θ+m)，其他类别使用cos(θ)
        output = torch.where(one_hot == 1, phi, cosine)

        # 应用缩放因子
        output = output * self.scale

        return output


class SoftmaxClassifier(nn.Module):
    """传统Softmax分类器，用于对比"""

    def __init__(self, embedding_size, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, embeddings, labels=None):
        return self.fc(embeddings)