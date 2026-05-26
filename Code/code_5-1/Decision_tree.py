# ======= 分类实验2：决策树分类 =======
# 这个实验展示了决策树分类算法在鸢尾花数据集上的应用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 决策树相关
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns  # 用于绘制更美观的图形

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 步骤1: 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标变量
feature_names = iris.feature_names  # 特征名称
target_names = iris.target_names  # 目标类别名称

# 为特征名称和目标类别名称创建中文对应
feature_names_zh = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
target_names_zh = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

# 步骤2: 查看数据基本信息
print("===== 数据集基本信息 =====")
print(f"数据集大小: {X.shape[0]}个样本, {X.shape[1]}个特征")
print(f"特征名称: {feature_names_zh}")
print(f"目标类别: {target_names_zh}")

# 步骤3: 创建DataFrame以便更好地查看数据
# 创建一个包含所有特征和目标变量的DataFrame
df = pd.DataFrame(data=X, columns=feature_names_zh)
df['品种'] = [target_names_zh[i] for i in y]  # 添加类别名称列
print("\n数据集预览 (前5行):")
print(df.head())

# 步骤4: 数据集的统计描述
print("\n数据集统计描述:")
print(df.describe())

# 步骤5: 探索各个类别的特征分布
plt.figure(figsize=(12, 8))
# 使用箱线图显示不同种类的鸢尾花在各个特征上的分布
for i, feature in enumerate(feature_names_zh):
    plt.subplot(2, 2, i+1)  # 创建2×2的子图
    # 使用seaborn的箱线图函数，按种类分组绘制箱线图
    sns.boxplot(x='品种', y=feature, data=df)
    plt.title(f'{feature}分布')
    plt.xticks(rotation=45)  # 旋转x轴标签以避免重叠

plt.tight_layout()  # 调整子图布局
plt.show()

# 步骤6: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\n划分后的训练集大小: {X_train.shape[0]}个样本")
print(f"划分后的测试集大小: {X_test.shape[0]}个样本")

# 步骤7: 创建并训练决策树模型
# max_depth=3设置决策树最大深度，防止过拟合
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(X_train, y_train)  # 训练模型
print("\n决策树模型训练完成!")

# 步骤8: 在测试集上进行预测
y_pred = dt_clf.predict(X_test)
print("\n前10个测试样本的预测结果:")
for i in range(10):
    print(f"样本{i+1}: 真实类别={target_names_zh[y_test[i]]}, 预测类别={target_names_zh[y_pred[i]]}")

# 步骤9: 计算并输出模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n决策树模型准确率: {accuracy:.4f} (即{accuracy*100:.2f}%)")

# 步骤10: 可视化混淆矩阵
# 混淆矩阵显示了预测类别与真实类别的对比情况
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
# 使用seaborn的热图函数绘制混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_zh, yticklabels=target_names_zh)
plt.title('决策树分类混淆矩阵')
plt.ylabel('真实类别')
plt.xlabel('预测类别')
plt.show()

# 步骤11: 特征重要性分析
# 决策树可以计算各个特征对分类决策的重要性
importances = dt_clf.feature_importances_
print("\n特征重要性:")
for feature, importance in zip(feature_names_zh, importances):
    print(f"{feature}: {importance:.4f}")

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
# 对特征重要性进行降序排序
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names_zh[i] for i in indices], rotation=45)
plt.title('决策树特征重要性')
plt.tight_layout()
plt.show()

# 步骤12: 可视化决策树
plt.figure(figsize=(15, 10))
# plot_tree函数可视化决策树的结构
plot_tree(dt_clf, filled=True, feature_names=feature_names_zh,
          class_names=target_names_zh, rounded=True, fontsize=10)
plt.title('鸢尾花分类决策树')
plt.show()

# 步骤13: 可视化决策边界(使用前两个最重要的特征)
# 找出两个最重要的特征的索引
top_two_features = np.argsort(importances)[-2:]

# 创建一个新的决策树，只使用两个最重要的特征
dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_2d.fit(X_train[:, top_two_features], y_train)

# 创建网格以绘制决策边界
h = 0.02
x_min, x_max = X[:, top_two_features[0]].min() - 1, X[:, top_two_features[0]].max() + 1
y_min, y_max = X[:, top_two_features[1]].min() - 1, X[:, top_two_features[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格点的类别
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# 绘制散点图(按类别着色)
for i, color, marker in zip(range(3), ['blue', 'red', 'green'], ['o', 's', '^']):
    # 筛选当前类别的测试样本
    idx = y_test == i
    # 绘制散点图
    plt.scatter(X_test[idx, top_two_features[0]], X_test[idx, top_two_features[1]],
                c=color, label=target_names_zh[i], marker=marker, edgecolor='k')

plt.xlabel(feature_names_zh[top_two_features[0]])
plt.ylabel(feature_names_zh[top_two_features[1]])
plt.title('决策树分类边界 (使用两个最重要特征)')
plt.legend()
plt.show()