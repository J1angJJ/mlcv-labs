#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from skimage import data, segmentation, color
from skimage.segmentation import quickshift
from skimage.util import img_as_float


def main():
    flower_img = data.astronaut()  # 彩色
    flower_img = img_as_float(flower_img)

    # quickshift 参数: kernel_size, max_dist, ratio
    seg_quick = quickshift(flower_img, kernel_size=3, max_dist=10, ratio=0.5)

    # 可视化
    plt.rc("font", family='SimHei')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(flower_img)
    axes[0].set_title("原图 (flowers)")
    axes[0].axis('off')

    axes[1].imshow(color.label2rgb(seg_quick, flower_img, kind='avg'))
    axes[1].set_title("Quickshift分割结果")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()