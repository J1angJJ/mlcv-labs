# chinese_font.py - 设置matplotlib中文显示

import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def set_chinese_font():
    """设置matplotlib使用黑体显示中文"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    # 检查字体是否存在
    try:
        plt.title('测试中文')
        plt.close()
        print("已成功设置中文显示")
    except:
        print("警告: 可能无法正确显示中文，请确保系统安装了黑体字体")