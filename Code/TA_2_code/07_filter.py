import cv2
import numpy as np
import matplotlib.pyplot as plt

 
def main():
    # 读取彩色图像
    img_bgr = cv2.imread('lena.png')
    if img_bgr is None:
        print("未能读取到 lena.png，请检查文件路径！")
        return

    # 转灰度
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    gaussian_blur = cv2.GaussianBlur(img_gray, (7, 7), 1.5)

    # 中值滤波
    median_blur = cv2.medianBlur(img_gray, 5)

    # 显示对比
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title('Original Gray (PNG)')
    axs[0].axis('off')

    axs[1].imshow(gaussian_blur, cmap='gray')
    axs[1].set_title('Gaussian Filter (7x7, sigma=1.5)')
    axs[1].axis('off')

    axs[2].imshow(median_blur, cmap='gray')
    axs[2].set_title('Median Filter (kernel=5)')
    axs[2].axis('off')

    plt.show()


if __name__ == '__main__':
    main()