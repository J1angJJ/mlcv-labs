#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, segmentation, feature, color, measure
from scipy import ndimage as ndi


def main():
    cam_img = data.camera()

    otsu_val = filters.threshold_otsu(cam_img)
    binary_cam = cam_img > otsu_val

    distance_map = ndi.distance_transform_edt(binary_cam)

    coords = feature.peak_local_max(distance_map, min_distance=20)

    # 将坐标转换为布尔蒙版
    markers_mask = np.zeros(distance_map.shape, dtype=bool)
    markers_mask[coords[:, 0], coords[:, 1]] = True

    # 标记markers
    markers = measure.label(markers_mask)

    # 分水岭
    labels_ws = segmentation.watershed(-distance_map, markers, mask=binary_cam)

    # 可视化
    plt.rc("font", family='SimHei')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(cam_img, cmap='gray')
    axes[0].set_title("原图 (camera)")
    axes[0].axis('off')

    axes[1].imshow(binary_cam, cmap='gray')
    axes[1].set_title("Otsu阈值后")
    axes[1].axis('off')

    axes[2].imshow(distance_map, cmap='jet')
    axes[2].set_title("距离变换 (EDT)")
    axes[2].axis('off')

    axes[3].imshow(color.label2rgb(labels_ws, bg_label=0))
    axes[3].set_title("分水岭结果")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()