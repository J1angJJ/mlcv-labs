#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from skimage import data, color
from sklearn.cluster import KMeans


def main():
    coffee_img = data.coffee()  # 彩色图
    h, w, ch = coffee_img.shape

    # reshape 成 (H*W, 3)
    flatten_img = coffee_img.reshape(h*w, ch).astype(float)

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(flatten_img)

    labels = kmeans.labels_
    labels_2d = labels.reshape((h, w))

    # 用 label2rgb 将同一标签的像素替换为其均值色
    segmented_kmeans = color.label2rgb(labels_2d, coffee_img, kind='avg')

    # 显示
    plt.rc("font", family='SimHei')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(coffee_img)
    axes[0].set_title("原图 (coffee)")
    axes[0].axis('off')

    axes[1].imshow(labels_2d, cmap='nipy_spectral')
    axes[1].set_title("K-means 分割(k=4)")
    axes[1].axis('off')

    axes[2].imshow(segmented_kmeans)
    axes[2].set_title("分割后平均色")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()