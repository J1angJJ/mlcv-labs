# ======= 分类实验1：简单KNN分类 =======
# 这个实验展示了最基本的K近邻(KNN)分类算法应用于鸢尾花数据集

import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理
import matplotlib.pyplot as plt  # 用于可视化
from sklearn.datasets import load_iris  # 导入鸢尾花数据集
from sklearn.model_selection import train_test_split  # 用于分割训练集和测试集
from sklearn.neighbors import KNeighborsClassifier  # KNN分类器
from sklearn.metrics import accuracy_score  # 计算准确率

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 步骤1: 加载鸢尾花数据集
# load_iris()函数返回一个包含数据和元数据的对象
iris = load_iris()
X = iris.data  # 特征数据 (花萼长度、花萼宽度、花瓣长度、花瓣宽度)
y = iris.target  # 目标变量 (鸢尾花种类: 0=setosa, 1=versicolor, 2=virginica)
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
print("\n前5个样本:")
for i in range(5):
    print(f"样本{i+1}: 特征={X[i]}, 类别={target_names_zh[y[i]]}")

# 步骤3: 数据可视化 - 只使用前两个特征(花萼长度和宽度)以便于二维可视化
plt.figure(figsize=(10, 6))  # 创建图形，设置大小
# 为每个类别绘制散点图
for i, target_name in enumerate(target_names_zh):
    # 筛选当前类别的样本
    indices = y == i
    # 绘制散点图: x=花萼长度, y=花萼宽度
    plt.scatter(X[indices, 0], X[indices, 1],
                label=target_name)  # 每个类别用不同颜色和标签

# 添加图例和标签
plt.xlabel(feature_names_zh[0])  # x轴标签
plt.ylabel(feature_names_zh[1])  # y轴标签
plt.title('鸢尾花数据分布 (只显示花萼长度和宽度)')  # 图标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 步骤4: 划分训练集和测试集
# train_test_split函数将数据随机分为训练集(70%)和测试集(30%)
# random_state参数确保每次运行得到相同的划分结果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\n划分后的训练集大小: {X_train.shape[0]}个样本")
print(f"划分后的测试集大小: {X_test.shape[0]}个样本")

# 步骤5: 创建并训练KNN模型
# n_neighbors=3表示我们将使用最近的3个邻居来进行分类决策
knn = KNeighborsClassifier(n_neighbors=3)
# 使用训练数据拟合(训练)模型
knn.fit(X_train, y_train)
print("\n模型训练完成！")

# 步骤6: 在测试集上进行预测
y_pred = knn.predict(X_test)  # 预测测试集的类别
print("\n前10个测试样本的预测结果:")
for i in range(10):
    print(f"样本{i+1}: 真实类别={target_names_zh[y_test[i]]}, 预测类别={target_names_zh[y_pred[i]]}")

# 步骤7: 计算并输出模型准确率
# 准确率 = 正确预测的样本数 / 总样本数
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f} (即{accuracy*100:.2f}%)")

# 步骤8: 可视化测试集的预测结果 (仅使用前两个特征)
plt.figure(figsize=(10, 6))
# 创建网格以绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格点的类别，我们只使用前两个特征，其他特征用平均值填充
# 这样可以在二维空间中可视化决策边界
Z = knn.predict(np.c_[xx.ravel(), yy.ravel(),
                      np.ones(xx.ravel().shape) * np.mean(X[:, 2]),
                      np.ones(xx.ravel().shape) * np.mean(X[:, 3])])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3)

# 绘制测试集点，正确分类和错误分类的点用不同的标记
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:  # 正确分类
        plt.scatter(X_test[i, 0], X_test[i, 1], c='green', marker='o',
                   edgecolors='k', label='正确' if i == 0 else "")
    else:  # 错误分类
        plt.scatter(X_test[i, 0], X_test[i, 1], c='red', marker='x',
                   edgecolors='k', label='错误' if i == 0 else "")

plt.xlabel(feature_names_zh[0])
plt.ylabel(feature_names_zh[1])
plt.title('KNN分类结果 (绿色=正确分类, 红色=错误分类)')
plt.legend()
plt.grid(True)
plt.show()

# 步骤9: 尝试不同的K值(邻居数量)并比较准确率
k_values = [1, 3, 5, 7, 9, 11, 13, 15]  # 不同的K值
accuracies = []  # 存储不同K值对应的准确率

for k in k_values:
    # 创建并训练具有不同k值的KNN模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # 预测并计算准确率
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 绘制K值与准确率的关系
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel('K值 (邻居数量)')
plt.ylabel('准确率')
plt.title('不同K值对KNN分类准确率的影响')
plt.grid(True)
plt.show()

print("\n不同K值对应的准确率:")
for k, acc in zip(k_values, accuracies):
    print(f"K={k}: {acc:.4f} (即{acc*100:.2f}%)")