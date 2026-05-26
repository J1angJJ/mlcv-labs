"""
特征提取模型模块
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE


class FeatureExtractor(nn.Module):
    """
    特征提取模型：冻结预训练模型的所有权重，仅训练分类器
    """

    def __init__(self, model_name, num_classes=10):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name

        # 加载预训练模型
        if model_name == "resnet18":
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻分类器参数
        if model_name == "resnet18":
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif model_name in ["mobilenet_v2", "efficientnet_b0"]:
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        self.model = self.model.to(DEVICE)

    def forward(self, x):
        return self.model(x)

    def get_trainable_params(self):
        """获取可训练参数"""
        if self.model_name == "resnet18":
            return self.model.fc.parameters()
        else:
            return self.model.classifier.parameters()