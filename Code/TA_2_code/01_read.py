import cv2
import matplotlib.pyplot as plt

 
def main():
    # 读取彩色图像
    img_bgr = cv2.imread('lena.png')

    if img_bgr is None:
        print("未能读取到 lena.png，请检查文件路径！")
        return

    # 转换为 RGB 以便用 matplotlib 正确显示色彩
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 显示图像
    plt.imshow(img_rgb)
    plt.title("Lena (PNG) in RGB")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()