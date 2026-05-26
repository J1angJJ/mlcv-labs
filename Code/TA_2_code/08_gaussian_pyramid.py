import cv2
import numpy as np
import matplotlib.pyplot as plt

 
def main():
    # 读取彩色图像
    img_bgr = cv2.imread('lena.png')
    if img_bgr is None:
        print("未能读取到 lena.png，请检查文件路径！")
        return

    # 转 RGB 以便用 matplotlib 正确显示
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 建立多层金字塔
    img_level0 = img_rgb.copy()
    img_level1 = cv2.pyrDown(img_level0)
    img_level2 = cv2.pyrDown(img_level1)

    # 显示多层对比
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_level0)
    axs[0].set_title(f"Level 0: {img_level0.shape}")
    axs[0].axis('off')

    axs[1].imshow(img_level1)
    axs[1].set_title(f"Level 1: {img_level1.shape}")
    axs[1].axis('off')

    axs[2].imshow(img_level2)
    axs[2].set_title(f"Level 2: {img_level2.shape}")
    axs[2].axis('off')

    plt.show()


if __name__ == '__main__':
    main()