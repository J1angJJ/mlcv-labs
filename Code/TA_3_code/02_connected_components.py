#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from skimage import data, filters, measure, morphology


def main():
    # 读图并做Otsu二值化
    coins_img = data.coins()
    otsu_val = filters.threshold_otsu(coins_img)
    binary_otsu = coins_img > otsu_val

    # 连通域标记
    labeled = measure.label(binary_otsu, connectivity=2)  # 8-连通
    props = measure.regionprops(labeled)
    print(f"[连通域标记] 找到 {len(props)} 个连通域")

    # 形态学操作: 移除小区域
    cleaned = morphology.remove_small_objects(binary_otsu, min_size=50)

    # 显示结果
    plt.rc("font", family='SimHei')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(coins_img, cmap='gray')
    axes[0].set_title("原图 (coins)")
    axes[0].axis('off')

    axes[1].imshow(labeled, cmap='nipy_spectral')
    axes[1].set_title("连通域标记结果")
    axes[1].axis('off')

    axes[2].imshow(cleaned, cmap='gray')
    axes[2].set_title("移除小区域后")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()