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

    # 线性拉伸
    f_min, f_max = np.min(img_gray), np.max(img_gray)
    stretched_float = (img_gray - f_min) / (f_max - f_min) * 255.0
    stretched = stretched_float.astype(np.uint8)

    # 直方图均衡
    equalized = cv2.equalizeHist(img_gray)

    # 显示对比
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title(f'Original (min={f_min}, max={f_max})')
    axs[0].axis('off')

    axs[1].imshow(stretched, cmap='gray')
    axs[1].set_title('Linearly Stretched')
    axs[1].axis('off')

    axs[2].imshow(equalized, cmap='gray')
    axs[2].set_title('Histogram Equalization')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()