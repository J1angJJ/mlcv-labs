import cv2
import numpy as np
import matplotlib.pyplot as plt

 
def main():
    # 读取彩色图像
    img_bgr = cv2.imread('lena.png')

    if img_bgr is None:
        print("未能读取到 lena.png，请检查文件路径！")
        return

    # 转到 HSV 空间
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # 同时准备好 RGB 格式以便展示原彩色图片
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 显示结果
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img_rgb)
    axs[0].set_title('Original (PNG) in RGB')
    axs[0].axis('off')

    axs[1].imshow(h, cmap='gray')
    axs[1].set_title('Hue channel')
    axs[1].axis('off')

    axs[2].imshow(s, cmap='gray')
    axs[2].set_title('Saturation channel')
    axs[2].axis('off')

    axs[3].imshow(v, cmap='gray')
    axs[3].set_title('Value channel')
    axs[3].axis('off')

    plt.show()


if __name__ == '__main__':
    main()