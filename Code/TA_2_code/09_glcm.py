import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

 
def main():
    # 读取彩色图像
    img_bgr = cv2.imread('lena.png')
    if img_bgr is None:
        print("未能读取到 lena.png，请检查文件路径！")
        return

    # 转灰度
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 计算灰度共生矩阵
    glcm = graycomatrix(img_gray,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True)

    # 提取常见纹理特征
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    print("GLCM 纹理特征：")
    print(f"Contrast: {contrast}")
    print(f"Dissimilarity: {dissimilarity}")
    print(f"Homogeneity: {homogeneity}")
    print(f"Energy: {energy}")
    print(f"Correlation: {correlation}")

    # 显示原图
    plt.imshow(img_gray, cmap='gray')
    plt.title('Lena (PNG) Gray for GLCM')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()