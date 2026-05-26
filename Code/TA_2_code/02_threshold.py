import cv2
import numpy as np
import matplotlib.pyplot as plt

 
def main():
    # 读取彩色图像
    img_bgr = cv2.imread('lena.png')

    if img_bgr is None:
        print("未能读取到 lena.png，请检查文件路径！")
        return

    # 转成灰度图
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 调整亮度：为所有像素加上 50（注意clip，防止像素溢出）
    brighter_gray = np.clip(img_gray + 50, 0, 255).astype(np.uint8)

    # 阈值操作：设定阈值为 128，大于此阈值的置 255，小于则 0
    threshold_val = 128
    _, binary_img = cv2.threshold(brighter_gray, threshold_val, 255, cv2.THRESH_BINARY)

    # 显示结果
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title('Original Gray')
    axs[0].axis('off')

    axs[1].imshow(brighter_gray, cmap='gray')
    axs[1].set_title('Brighter Gray (+50)')
    axs[1].axis('off')

    axs[2].imshow(binary_img, cmap='gray')
    axs[2].set_title('Thresholded Image')
    axs[2].axis('off')

    plt.show()


if __name__ == '__main__':
    main()