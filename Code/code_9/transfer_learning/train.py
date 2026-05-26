"""
模型训练函数模块
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import DEVICE, MODEL_SAVE_DIR, FEATURE_EXTRACT_LR, FINE_TUNE_LR_FC, FINE_TUNE_LR_BACKBONE, GRAD_UNFREEZE_EVERY

def train_feature_extractor(model, dataloaders, dataset_sizes, num_epochs=10):
    """训练特征提取模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.get_trainable_params(), lr=FEATURE_EXTRACT_LR, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs)

def train_fine_tuning_model(model, dataloaders, dataset_sizes, num_epochs=10):
    """训练微调模型"""
    criterion = nn.CrossEntropyLoss()

    # 获取两组参数，分别设置不同的学习率
    feature_params, classifier_params = model.get_parameter_groups()

    optimizer = optim.SGD([
        {'params': feature_params, 'lr': FINE_TUNE_LR_BACKBONE},
        {'params': classifier_params, 'lr': FINE_TUNE_LR_FC}
    ], momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs)

def train_gradual_unfreezing_model(model, dataloaders, dataset_sizes, num_epochs=10):
    """训练渐进式解冻模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=FEATURE_EXTRACT_LR, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return train_model_with_unfreezing(model, criterion, optimizer, scheduler,
                                      dataloaders, dataset_sizes, num_epochs)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    """通用模型训练函数"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm创建进度条
            data_loader = dataloaders[phase]
            total_batches = len(data_loader)
            pbar = tqdm(data_loader, total=total_batches,
                        desc=f'{phase} - Batch', leave=False)

            # 迭代数据
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 如果是训练阶段，则进行反向传播和参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data).item()
                batch_acc = batch_corrects / inputs.size(0)

                running_loss += batch_loss
                running_corrects += batch_corrects

                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{batch_loss/inputs.size(0):.4f}',
                    'acc': f'{batch_acc:.4f}'
                })

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存历史记录
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def train_model_with_unfreezing(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    """带渐进式解冻的模型训练函数"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    # 初始已解冻层组索引
    current_unfrozen_group = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每GRAD_UNFREEZE_EVERY个epoch解冻一组新层
        if epoch > 0 and epoch % GRAD_UNFREEZE_EVERY == 0:
            current_unfrozen_group += 1
            if current_unfrozen_group < len(model.layer_groups):
                print(f"解冻层组 {current_unfrozen_group}")
                model.unfreeze_layer_group(current_unfrozen_group)

                # 更新优化器以包含新解冻的参数
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=FEATURE_EXTRACT_LR, momentum=0.9)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm创建进度条
            data_loader = dataloaders[phase]
            total_batches = len(data_loader)
            pbar = tqdm(data_loader, total=total_batches,
                        desc=f'{phase} - Batch', leave=False)

            # 迭代数据
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 如果是训练阶段，则进行反向传播和参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data).item()
                batch_acc = batch_corrects / inputs.size(0)

                running_loss += batch_loss
                running_corrects += batch_corrects

                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{batch_loss/inputs.size(0):.4f}',
                    'acc': f'{batch_acc:.4f}'
                })

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存历史记录
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def save_model(model, model_name, model_type):
    """保存模型"""
    filename = f"{model_name}_{model_type}.pth"
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    torch.save(model.state_dict(), filepath)
    print(f"模型保存到 {filepath}")