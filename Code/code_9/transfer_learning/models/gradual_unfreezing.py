"""
渐进式解冻模型模块
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE


class GradualUnfreezingModel(nn.Module):
    """
    渐进式解冻模型：从浅层到深层逐步解冻模型层
    """

    def __init__(self, model_name, num_classes=10):
        super(GradualUnfreezingModel, self).__init__()
        self.model_name = model_name

        # 加载预训练模型
        if model_name == "resnet18":
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

            # 定义层组，按照后向顺序
            self.layer_groups = [
                self.model.fc,  # 分类器层
                self.model.layer4,  # 最深层
                self.model.layer3,
                self.model.layer2,
                self.model.layer1,
                self.model.conv1  # 最浅层
            ]

        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

            # 为MobileNetV2定义层组
            features_len = len(self.model.features)
            step = features_len // 5  # 将特征层分为5组

            self.layer_groups = [
                self.model.classifier,  # 分类器层
                self.model.features[step * 4:],  # 最深层特征
                self.model.features[step * 3:step * 4],
                self.model.features[step * 2:step * 3],
                self.model.features[step:step * 2],
                self.model.features[:step]  # 最浅层特征
            ]

        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

            # 为EfficientNet定义层组
            features_len = len(self.model.features)
            step = features_len // 5  # 将特征层分为5组

            self.layer_groups = [
                self.model.classifier,  # 分类器层
                self.model.features[step * 4:],  # 最深层特征
                self.model.features[step * 3:step * 4],
                self.model.features[step * 2:step * 3],
                self.model.features[step:step * 2],
                self.model.features[:step]  # 最浅层特征
            ]

        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 初始冻结所有层
        self.freeze_all()

        # 首先解冻分类器层
        self.unfreeze_layer_group(0)

        self.model = self.model.to(DEVICE)

    def freeze_all(self):
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_layer_group(self, group_idx):
        """解冻指定的层组"""
        if group_idx < len(self.layer_groups):
            for param in self.layer_groups[group_idx].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)