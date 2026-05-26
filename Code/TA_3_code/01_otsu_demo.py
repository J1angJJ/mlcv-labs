#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from skimage import data, filters


def main():
    coins_img = data.coins()

    otsu_val = filters.threshold_otsu(coins_img)
    binary_otsu = coins_img > otsu_val

    # 绘图显示
    plt.rc("font", family='SimHei')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(coins_img, cmap='gray')
    axes[0].set_title("原图 (coins)")
    axes[0].axis('off')

    axes[1].hist(coins_img.ravel(), bins=256)
    axes[1].axvline(otsu_val, color='r', linestyle='--', label=f"Otsu阈值={otsu_val}")
    axes[1].set_title("灰度直方图")
    axes[1].legend()

    axes[2].imshow(binary_otsu, cmap='gray')
    axes[2].set_title("Otsu分割结果")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()