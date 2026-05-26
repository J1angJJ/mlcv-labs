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

    # 进行傅里叶变换
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)  # 将零频移动到频谱中心
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 显示原图和幅度谱
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title('Original Gray (PNG)')
    axs[0].axis('off')

    axs[1].imshow(magnitude_spectrum, cmap='gray')
    axs[1].set_title('Magnitude Spectrum')
    axs[1].axis('off')

    plt.show()


if __name__ == '__main__':
    main()