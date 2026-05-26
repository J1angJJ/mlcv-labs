# chinese_font.py - 设置matplotlib中文显示

import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import os


def set_chinese_font():
    """设置matplotlib使用中文字体显示"""
    system = platform.system()

    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

    # 解决保存图像时负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    return True


def test_chinese_display():
    """测试中文显示是否正常"""
    set_chinese_font()

    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
    plt.title('中文显示测试')
    plt.xlabel('横轴 (x)')
    plt.ylabel('纵轴 (y)')
    plt.grid(True)

    plt.savefig('chinese_test.png')
    plt.close()
    print("中文显示测试完成，图像已保存为'chinese_test.png'")
    return 'chinese_test.png'


if __name__ == "__main__":
    test_chinese_display()