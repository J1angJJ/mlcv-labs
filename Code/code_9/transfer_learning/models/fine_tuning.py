"""
微调模型模块
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE


class FineTuningModel(nn.Module):
    """
    微调模型：部分解冻预训练模型，使用不同的学习率
    """

    def __init__(self, model_name, num_classes=10):
        super(FineTuningModel, self).__init__()
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

        self.model = self.model.to(DEVICE)

    def forward(self, x):
        return self.model(x)

    def get_parameter_groups(self):
        """获取参数组，用于设置不同的学习率"""
        if self.model_name == "resnet18":
            feature_params = [p for n, p in self.model.named_parameters() if 'fc' not in n]
            classifier_params = self.model.fc.parameters()
        elif self.model_name in ["mobilenet_v2", "efficientnet_b0"]:
            feature_params = [p for n, p in self.model.named_parameters() if 'classifier' not in n]
            classifier_params = self.model.classifier.parameters()

        return feature_params, classifier_params