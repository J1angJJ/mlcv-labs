# arcface_visualize.py - 特征可视化与分析

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from arcface_model import FeatureExtractor, ArcFaceClassifier, SoftmaxClassifier
from arcface_data import load_data
from PIL import Image
import torchvision.transforms as transforms

# 导入中文字体设置
from chinese_font import set_chinese_font
set_chinese_font()  # 设置中文显示

def extract_features(model, dataloader, device):
    """从数据集中提取特征向量"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # 提取特征
            batch_features = model(inputs)
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.concatenate(labels)


def visualize_features(features, labels, title, class_names=None):
    """使用TSNE降维并可视化特征"""
    # TSNE降维
    print(f"正在进行TSNE降维, 这可能需要一些时间...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=(12, 10))

    # 获取唯一标签
    unique_labels = np.unique(labels)

    # 为每个类别绘制散点
    for i in unique_labels:
        idx = labels == i
        label_name = f"类别 {i}" if class_names is None else class_names[i]
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label_name, alpha=0.7)

    plt.legend()
    plt.title(title)
    plt.savefig(f"./results/{title.replace(' ', '_')}.png")
    plt.show()


def compute_intra_inter_distances(features, labels):
    """计算类内距离和类间距离"""
    # 计算类内距离
    intra_class_dist = []

    # 获取唯一标签
    unique_labels = np.unique(labels)

    # 对每个类别计算类内平均距离
    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) <= 1:
            continue

        # 计算类内所有样本对之间的距离
        distances = []
        for i in range(len(class_features)):
            for j in range(i + 1, len(class_features)):
                dist = np.sqrt(np.sum((class_features[i] - class_features[j]) ** 2))
                distances.append(dist)

        # 计算该类的平均距离
        if len(distances) > 0:
            intra_class_dist.append(np.mean(distances))

    # 计算类间距离
    inter_class_dist = []

    # 对每对不同类别计算类间平均距离
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label_i = unique_labels[i]
            label_j = unique_labels[j]

            features_i = features[labels == label_i]
            features_j = features[labels == label_j]

            # 计算两个类的所有样本对之间的距离
            distances = []
            for fi in features_i:
                for fj in features_j:
                    dist = np.sqrt(np.sum((fi - fj) ** 2))
                    distances.append(dist)

            # 计算这对类别的平均距离
            if len(distances) > 0:
                inter_class_dist.append(np.mean(distances))

    # 返回平均类内距离和平均类间距离
    return np.mean(intra_class_dist), np.mean(inter_class_dist)


def plot_distance_comparison(softmax_intra, softmax_inter, arcface_intra, arcface_inter):
    """绘制类内/类间距离对比图"""
    plt.figure(figsize=(10, 6))

    # 设置柱状图
    x = np.arange(2)  # 位置索引
    width = 0.35  # 柱宽

    plt.bar(x - width / 2, [softmax_intra, softmax_inter], width, label='Softmax')
    plt.bar(x + width / 2, [arcface_intra, arcface_inter], width, label='ArcFace')

    plt.ylabel('平均欧氏距离')
    plt.title('特征空间距离对比')
    plt.xticks(x, ['类内距离', '类间距离'])
    plt.legend()

    # 添加数值标签
    for i, v in enumerate([softmax_intra, softmax_inter]):
        plt.text(i - width / 2, v + 0.02, f'{v:.3f}', ha='center')

    for i, v in enumerate([arcface_intra, arcface_inter]):
        plt.text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center')

    # 添加类间/类内比率
    plt.text(0.5, 0.9, f'Softmax类间/类内比率: {softmax_inter / softmax_intra:.3f}',
             ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.85, f'ArcFace类间/类内比率: {arcface_inter / arcface_intra:.3f}',
             ha='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig('./results/distance_comparison.png')
    plt.show()


def face_recognition_demo(feature_extractor, classifier, test_loader, class_names, device):
    """人脸识别演示"""
    # 获取一批测试图像
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # 选择最多5张图像进行演示
    num_images = min(5, len(images))

    plt.figure(figsize=(15, 8))
    plt.suptitle("ArcFace人脸识别演示", fontsize=16)

    for i in range(num_images):
        # 获取图像和标签
        img = images[i]
        label = labels[i]

        # 预测
        feature_extractor.eval()
        classifier.eval()
        with torch.no_grad():
            input_img = img.unsqueeze(0).to(device)
            feature = feature_extractor(input_img)
            output = classifier(feature)

            # 获取Top-3预测结果
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top3_prob, top3_class = torch.topk(probabilities, min(3, len(class_names)))

        # 显示原始图像
        plt.subplot(num_images, 2, 2 * i + 1)
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * 0.5 + 0.5  # 反归一化
        plt.imshow(img_np)
        plt.title(f"真实标签: {class_names[label]}")
        plt.axis('off')

        # 显示预测结果
        plt.subplot(num_images, 2, 2 * i + 2)
        # 获取类别名称和概率
        class_names_list = [class_names[idx] for idx in top3_class[0].cpu().numpy()]
        probs_list = top3_prob[0].cpu().numpy()

        # 绘制条形图
        colors = ['green' if class_idx == label else 'red'
                  for class_idx in top3_class[0].cpu().numpy()]
        plt.bar(range(len(class_names_list)), probs_list, color=colors)
        plt.xticks(range(len(class_names_list)), class_names_list, rotation=45)
        plt.ylim(0, 1.0)
        plt.title("预测概率")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('./results/face_recognition_demo.png')
    plt.show()


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data_path = "./data/lfw_subset"
    train_loader, test_loader, num_classes = load_data(data_path)

    # 获取类别名称
    dataset = train_loader.dataset.dataset  # 获取底层数据集
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        class_names = [f"Person_{i}" for i in range(num_classes)]

    # 创建结果目录
    os.makedirs('./results', exist_ok=True)

    # 设置模型参数
    embedding_size = 512

    # 加载已训练的Softmax模型
    softmax_feature_extractor = FeatureExtractor(embedding_size).to(device)
    softmax_classifier = SoftmaxClassifier(embedding_size, num_classes).to(device)

    try:
        softmax_feature_extractor.load_state_dict(torch.load('./results/softmax_feature_extractor.pth'))
        softmax_classifier.load_state_dict(torch.load('./results/softmax_classifier.pth'))
    except:
        print("未找到Softmax模型，请先运行arcface_train.py")
        return

    # 加载已训练的ArcFace模型
    arcface_feature_extractor = FeatureExtractor(embedding_size).to(device)
    arcface_classifier = ArcFaceClassifier(embedding_size, num_classes).to(device)

    try:
        arcface_feature_extractor.load_state_dict(torch.load('./results/arcface_feature_extractor.pth'))
        arcface_classifier.load_state_dict(torch.load('./results/arcface_classifier.pth'))
    except:
        print("未找到ArcFace模型，请先运行arcface_train.py")
        return

    # 提取特征
    print("正在提取Softmax模型特征...")
    softmax_features, softmax_labels = extract_features(softmax_feature_extractor, test_loader, device)

    print("正在提取ArcFace模型特征...")
    arcface_features, arcface_labels = extract_features(arcface_feature_extractor, test_loader, device)

    # 可视化特征
    visualize_features(softmax_features, softmax_labels, "Softmax特征分布", class_names)
    visualize_features(arcface_features, arcface_labels, "ArcFace特征分布", class_names)

    # 计算类内和类间距离
    print("计算类内和类间距离...")
    softmax_intra, softmax_inter = compute_intra_inter_distances(softmax_features, softmax_labels)
    arcface_intra, arcface_inter = compute_intra_inter_distances(arcface_features, arcface_labels)

    print(f"Softmax - 平均类内距离: {softmax_intra:.4f}, 平均类间距离: {softmax_inter:.4f}")
    print(f"ArcFace - 平均类内距离: {arcface_intra:.4f}, 平均类间距离: {arcface_inter:.4f}")
    print(f"Softmax - 类间/类内比率: {softmax_inter / softmax_intra:.4f}")
    print(f"ArcFace - 类间/类内比率: {arcface_inter / arcface_intra:.4f}")

    # 绘制距离对比图
    plot_distance_comparison(softmax_intra, softmax_inter, arcface_intra, arcface_inter)

    # 人脸识别演示
    print("执行人脸识别演示...")
    face_recognition_demo(arcface_feature_extractor, arcface_classifier, test_loader, class_names, device)

    print("\n特征可视化和分析完成，请查看保存的图像文件。")


if __name__ == "__main__":
    main()