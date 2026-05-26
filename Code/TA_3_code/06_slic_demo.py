#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from skimage import data, color
from skimage.segmentation import slic
from skimage.util import img_as_float


def main():
    flower_img = data.astronaut()
    flower_img = img_as_float(flower_img)

    segments_slic = slic(flower_img, n_segments=200, compactness=10, start_label=1)

    plt.rc("font", family='SimHei')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(flower_img)
    axes[0].set_title("原图 (flowers)")
    axes[0].axis('off')

    axes[1].imshow(color.label2rgb(segments_slic, flower_img, kind='avg'))
    axes[1].set_title("SLIC 超像素 (n_segments=200)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()