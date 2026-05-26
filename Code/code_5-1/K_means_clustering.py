# ======= 聚类实验1：简单K-means聚类 =======
# 这个实验展示了K-means聚类算法在鸢尾花数据集上的应用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans  # K-means聚类算法
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.metrics import silhouette_score  # 用于评估聚类质量

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 步骤1: 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 真实标签(仅用于评估，聚类是无监督的，不使用标签进行训练)
feature_names = iris.feature_names
target_names = iris.target_names

# 为特征名称和目标类别名称创建中文对应
feature_names_zh = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
target_names_zh = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

# 步骤2: 查看数据基本信息
print("===== 数据集基本信息 =====")
print(f"数据集大小: {X.shape[0]}个样本, {X.shape[1]}个特征")
print(f"特征名称: {feature_names_zh}")
print("\n注意: 聚类是无监督学习，训练时不使用真实标签，但我们可以用它们来评估聚类效果\n")

# 步骤3: 数据预处理 - 标准化特征
# 标准化使所有特征具有相似的尺度(均值=0, 方差=1)，这对于基于距离的算法如K-means很重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("特征已标准化，均值接近0，标准差接近1")

# 步骤4: 数据可视化 - 使用前两个特征
plt.figure(figsize=(10, 6))
# 绘制散点图，根据真实标签着色(仅用于理解数据分布)
for i, target_name in enumerate(target_names_zh):
    plt.scatter(X_scaled[y == i, 0], X_scaled[y == i, 1],
                label=target_name)
plt.xlabel(f'标准化的{feature_names_zh[0]}')
plt.ylabel(f'标准化的{feature_names_zh[1]}')
plt.title('鸢尾花数据分布 (标准化后，真实标签着色)')
plt.legend()
plt.grid(True)
plt.show()

# 步骤5: 尝试不同的聚类数量并评估
# 使用肘部法则和轮廓系数来确定最佳聚类数
k_range = range(2, 10)  # 尝试2到9个聚类
inertia_values = []  # 存储每个k值的WCSS(组内平方和，即簇内误差)
silhouette_values = []  # 存储每个k值的轮廓系数

for k in k_range:
    # 创建并训练K-means模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # 计算并存储WCSS
    inertia_values.append(kmeans.inertia_)

    # 计算并存储轮廓系数(值越接近1越好)
    silhouette_values.append(silhouette_score(X_scaled, kmeans.labels_))

    print(f"k={k}: WCSS={kmeans.inertia_:.2f}, 轮廓系数={silhouette_score(X_scaled, kmeans.labels_):.4f}")

# 步骤6: 绘制肘部法则图和轮廓系数图
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# 肘部法则图
ax[0].plot(k_range, inertia_values, 'bo-')
ax[0].set_xlabel('聚类数量 (k)')
ax[0].set_ylabel('WCSS (组内平方和)')
ax[0].set_title('肘部法则图 (寻找拐点)')
ax[0].grid(True)

# 轮廓系数图
ax[1].plot(k_range, silhouette_values, 'ro-')
ax[1].set_xlabel('聚类数量 (k)')
ax[1].set_ylabel('轮廓系数')
ax[1].set_title('轮廓系数图 (值越大越好)')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# 根据上面的分析，我们假设最佳聚类数为3(这也恰好是我们知道的鸢尾花种类数)
best_k = 3
print(f"\n根据肘部法则和轮廓系数分析，我们选择k={best_k}作为最佳聚类数")

# 步骤7: 使用最佳聚类数进行K-means聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)  # 获取聚类标签

# 步骤8: 可视化聚类结果(使用前两个特征)
plt.figure(figsize=(12, 6))

# 左图: 基于聚类标签着色
plt.subplot(1, 2, 1)
for cluster_id in range(best_k):
    # 筛选属于当前聚类的样本
    cluster_samples = X_scaled[cluster_labels == cluster_id]
    # 绘制散点图
    plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1],
                label=f'聚类 {cluster_id}')

# 标记聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X',
            label='聚类中心')

plt.xlabel(f'标准化的{feature_names_zh[0]}')
plt.ylabel(f'标准化的{feature_names_zh[1]}')
plt.title('K-means聚类结果')
plt.legend()
plt.grid(True)

# 右图: 基于真实标签着色，但用符号表示聚类
plt.subplot(1, 2, 2)
for i, target_name in enumerate(target_names_zh):
    # 筛选属于当前真实类别的样本
    idx = y == i
    # 绘制散点图，不同的真实类别用不同颜色
    plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1],
                label=target_name)

plt.xlabel(f'标准化的{feature_names_zh[0]}')
plt.ylabel(f'标准化的{feature_names_zh[1]}')
plt.title('真实类别分布(颜色)与聚类结果(形状)对比')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 步骤9: 分析聚类结果与真实标签的对应关系
# 创建一个交叉表，显示每个聚类中各个真实标签的样本数量
cross_tab = pd.crosstab(index=cluster_labels, columns=y,
                        rownames=['聚类'], colnames=['真实标签'])
# 替换列名为实际的花种名称
cross_tab.columns = target_names_zh

print("\n聚类结果与真实标签的对应关系:")
print(cross_tab)
print("\n每个聚类中的主要花种:")
for cluster_id in range(best_k):
    # 找出当前聚类中数量最多的花种
    main_species = cross_tab.iloc[cluster_id].idxmax()
    print(f"聚类 {cluster_id} 主要包含 {main_species} ({cross_tab.iloc[cluster_id].max()}个样本)")

# 步骤10: 可视化每个聚类的特征分布
plt.figure(figsize=(15, 10))
# 为每个特征创建一个子图
for i, feature in enumerate(feature_names_zh):
    plt.subplot(2, 2, i + 1)
    # 为每个聚类绘制一个箱线图，显示该特征在各个聚类中的分布
    data = []
    labels = []
    for cluster_id in range(best_k):
        # 获取当前聚类中的样本在当前特征上的值
        values = X[cluster_labels == cluster_id, i]
        data.append(values)
        labels.append(f'聚类 {cluster_id}')

    # 绘制箱线图
    plt.boxplot(data, tick_labels=labels)
    plt.title(f'{feature}在各聚类中的分布')
    plt.grid(True)

plt.tight_layout()
plt.show()