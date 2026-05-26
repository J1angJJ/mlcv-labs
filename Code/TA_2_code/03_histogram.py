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

    # 绘制直方图
    plt.figure(figsize=(8, 4))
    plt.hist(img_gray.flatten(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title('Histogram of Lena (PNG Gray)')
    plt.xlabel('Gray Value')
    plt.ylabel('Pixel Count')
    plt.show()


if __name__ == '__main__':
    main()