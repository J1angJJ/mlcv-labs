import cv2
import numpy as np

def edge_sobel(img_gray):
    """
    使用 Sobel 算子检测边缘
    返回: Sobel 混合结果 (dx+dy) 的绝对值并归一化到 0~255
    """
    # Sobel X 方向
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    # Sobel Y 方向
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # 取绝对值并组合
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel = cv2.convertScaleAbs(abs_sobelx) + cv2.convertScaleAbs(abs_sobely)
    return sobel


def edge_prewitt(img_gray):
    """
    使用 Prewitt 算子检测边缘
    自定义卷积核进行操作（类似于Sobel，但没有高斯加权）
    """
    # Prewitt X 卷积核
    kernelx = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float32)
    # Prewitt Y 卷积核
    kernely = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]], dtype=np.float32)

    prewittx = cv2.filter2D(img_gray, -1, kernelx)
    prewitty = cv2.filter2D(img_gray, -1, kernely)

    prewitt = cv2.convertScaleAbs(prewittx) + cv2.convertScaleAbs(prewitty)
    return prewitt


def edge_roberts(img_gray):
    """
    使用 Roberts 算子检测边缘
    Roberts 算子 kernel:
       Gx = [[1,  0],
             [0, -1]]
       Gy = [[0,  1],
             [-1, 0]]
    """
    # Roberts X 卷积核
    kernelx = np.array([[1, 0],
                        [0, -1]], dtype=np.float32)
    # Roberts Y 卷积核
    kernely = np.array([[0, 1],
                        [-1, 0]], dtype=np.float32)

    robertsx = cv2.filter2D(img_gray, cv2.CV_64F, kernelx)
    robertsy = cv2.filter2D(img_gray, cv2.CV_64F, kernely)

    roberts = cv2.convertScaleAbs(robertsx) + cv2.convertScaleAbs(robertsy)
    return roberts


def edge_laplacian(img_gray):
    """
    使用 OpenCV 提供的 Laplacian 算子进行边缘检测
    """
    lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return lap


def edge_canny(img_gray):
    """
    使用 OpenCV Canny 算子
    阈值可根据实际场景调整
    """
    return cv2.Canny(img_gray, 100, 200)


def hessian_analysis(img_gray, threshold=2000):
    """
    简易 Hessian 分析示例：
    1. 先对图像做高斯平滑
    2. 计算二阶导数(即 Hessian 矩阵的4个元素)
    3. 分析特征值，若满足一定条件则视为候选边缘或角点
    :param threshold: 特征值阈值(简单判断)
    """
    # 高斯平滑
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 1.0)

    # 一阶导数
    Ix = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    Iy = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)

    # 二阶导数
    Ixx = cv2.Scharr(Ix, cv2.CV_64F, 1, 0)
    Iyy = cv2.Scharr(Iy, cv2.CV_64F, 0, 1)
    Ixy = cv2.Scharr(Ix, cv2.CV_64F, 0, 1)  # or cv2.Scharr(Iy, 1, 0)

    # 将结果可视化, 简单用彩色图来展示
    hessian_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    rows, cols = img_gray.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            # 取 Hessian
            H = np.array([[Ixx[y, x], Ixy[y, x]],
                          [Ixy[y, x], Iyy[y, x]]], dtype=np.float64)
            # 求特征值
            vals, _ = np.linalg.eig(H)
            lam1, lam2 = vals
            # 简单判断 lam1*lam2 是否大于某个阈值来判定可能是“角点”或类似结构
            # 这里只是演示，实际应用可更复杂
            val = lam1 * lam2
            if val < -threshold:
                # 负值太大 -> 可能是条纹/边缘
                hessian_vis[y, x] = (0, 0, 255)  # 红点
            elif val > threshold:
                # 正值太大 -> 可能是斑点状角点
                hessian_vis[y, x] = (255, 0, 0)  # 蓝点
            # 否则忽略

    return hessian_vis


def blob_detection_dog(img_gray, num_scales=5, sigma=1.6, k=np.sqrt(2)):
    """
    简易 DoG 多尺度 blob 检测
    返回标记好的彩色输出，用小圆圈可视化检测到的blob
    """
    img_f = img_gray.astype(np.float32) / 255.0
    gaussian_pyramid = []

    # 构建高斯金字塔
    for s in range(num_scales + 1):
        sigma_current = (k ** s) * sigma
        blur = cv2.GaussianBlur(img_f, (0, 0), sigma_current)
        gaussian_pyramid.append(blur)

    # 构建DoG金字塔
    dog_pyramid = []
    for i in range(num_scales):
        dog = gaussian_pyramid[i + 1] - gaussian_pyramid[i]
        dog_pyramid.append(dog)

    # 在 DoG 立体中(空间+尺度)搜索极值 (简化实现)
    keypoints = []
    for s in range(1, num_scales - 1):
        dog_prev = dog_pyramid[s - 1]
        dog_curr = dog_pyramid[s]
        dog_next = dog_pyramid[s + 1]
        rows, cols = dog_curr.shape

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                val = dog_curr[y, x]
                # 取邻域
                region = []
                region.extend(dog_prev[y - 1:y + 2, x - 1:x + 2].flatten())
                region.extend(dog_curr[y - 1:y + 2, x - 1:x + 2].flatten())
                region.extend(dog_next[y - 1:y + 2, x - 1:x + 2].flatten())

                if val == max(region) or val == min(region):
                    # 记录 (x,y) 以及近似blob大小
                    kp_size = (k ** s) * sigma * 2
                    keypoints.append((x, y, kp_size))

    # 在可视化输出上绘制圆
    out_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (x, y, size) in keypoints:
        radius = int(size / 2)
        cv2.circle(out_img, (x, y), radius, (0, 0, 255), 1)

    return out_img


def sift_keypoints(img_gray):
    """
    使用 OpenCV SIFT 检测关键点并返回绘制结果
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    out_img = cv2.drawKeypoints(img_gray, kp, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img, kp, des


def sift_match(img_gray1, img_gray2):
    """
    使用 SIFT 并匹配两张灰度图的关键点
    返回匹配结果图像
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1, None)
    kp2, des2 = sift.detectAndCompute(img_gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    ratio_thresh = 0.75
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    matched_img = cv2.drawMatches(img_gray1, kp1,
                                  img_gray2, kp2,
                                  good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img


def main():
    # 1) 读取测试图像
    img_path = 'test.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取图像，请检查路径：", img_path)
        return

    # 2) Sobel
    sobel_result = edge_sobel(img)
    cv2.imshow("Sobel Edge", sobel_result)

    # 3) Prewitt
    prewitt_result = edge_prewitt(img)
    cv2.imshow("Prewitt Edge", prewitt_result)

    # 4) Roberts
    roberts_result = edge_roberts(img)
    cv2.imshow("Roberts Edge", roberts_result)

    # 5) Laplacian
    laplacian_result = edge_laplacian(img)
    cv2.imshow("Laplacian Edge", laplacian_result)

    # 6) Canny
    canny_result = edge_canny(img)
    cv2.imshow("Canny Edge", canny_result)

    # 7) Hessian 分析
    hess_vis = hessian_analysis(img, threshold=2000)
    cv2.imshow("Hessian Analysis", hess_vis)

    # 8) Blob 检测 (DoG)
    blob_out = blob_detection_dog(img)
    cv2.imshow("DoG Blob Detection", blob_out)

    # 9) SIFT 关键点
    sift_out, _, _ = sift_keypoints(img)
    cv2.imshow("SIFT Keypoints", sift_out)

    # 如果需要做 SIFT 匹配，请自行读取第二张图像
    # img2 = cv2.imread("test2.jpg", cv2.IMREAD_GRAYSCALE)
    # if img2 is not None:
    #     match_out = sift_match(img, img2)
    #     cv2.imshow("SIFT Matching", match_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()