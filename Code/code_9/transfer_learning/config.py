import os
import torch

# 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 创建必要的目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据配置
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224  # 预训练模型的输入大小

# 训练配置
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 迁移学习配置
FEATURE_EXTRACT_LR = 0.001      # 特征提取的学习率
FINE_TUNE_LR_FC = 0.001         # 微调FC层的学习率
FINE_TUNE_LR_BACKBONE = 0.0001  # 微调主干网络的学习率
GRAD_UNFREEZE_EVERY = 2         # 每隔多少个epoch解冻一组层

# 预训练模型选择
MODELS = [
    "resnet18", 
    "mobilenet_v2", 
    "efficientnet_b0"
]

# 训练进度显示配置
USE_TQDM = True