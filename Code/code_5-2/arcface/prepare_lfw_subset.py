# prepare_lfw_subset.py - 从已下载的LFW数据集创建子集

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 导入中文字体设置
from chinese_font import set_chinese_font
set_chinese_font()  # 设置中文显示


def create_lfw_subset(lfw_path, output_path="./data/lfw_subset", min_photos=10, num_people=10):
    """
    从LFW数据集中创建一个子集，选择拥有足够多照片的人

    参数:
    lfw_path: LFW数据集路径
    output_path: 输出路径
    min_photos: 每个人至少需要的照片数量
    num_people: 选择的人数
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 检查输入路径是否存在
    if not os.path.exists(lfw_path):
        print(f"错误: 路径 {lfw_path} 不存在")
        return None

    # 查找所有子目录（每个子目录对应一个人）
    person_dirs = []
    for item in os.listdir(lfw_path):
        item_path = os.path.join(lfw_path, item)
        if os.path.isdir(item_path):
            # 统计目录中的图像文件数量
            num_images = len([f for f in os.listdir(item_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if num_images >= min_photos:
                person_dirs.append((item, num_images))

    if not person_dirs:
        # 如果没有找到人物目录，可能LFW直接解压为平面结构，尝试按名称前缀分组
        print("未发现符合标准的人物目录，尝试按文件名组织...")
        return organize_by_filename(lfw_path, output_path, min_photos, num_people)

    # 按照照片数量排序
    person_dirs.sort(key=lambda x: x[1], reverse=True)

    # 选择前num_people个人
    selected_people = person_dirs[:num_people]

    print(f"已选择{len(selected_people)}个人，每人至少有{min_photos}张照片:")
    for person, count in selected_people:
        print(f"  - {person}: {count}张照片")

        # 创建此人的目录
        person_output_dir = os.path.join(output_path, person)
        os.makedirs(person_output_dir, exist_ok=True)

        # 复制照片
        person_dir = os.path.join(lfw_path, person)
        photos = [f for f in os.listdir(person_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for photo in photos:
            src = os.path.join(person_dir, photo)
            dst = os.path.join(person_output_dir, photo)
            shutil.copy2(src, dst)

    return output_path


def organize_by_filename(lfw_path, output_path, min_photos, num_people):
    """
    当LFW解压为平面结构时，按文件名前缀组织

    LFW命名格式: [人名]_[序号].jpg
    """
    print("按文件名分析人物...")

    # 获取所有图像文件
    all_images = [f for f in os.listdir(lfw_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 分析文件名，提取人名
    person_photos = {}
    for img in all_images:
        parts = img.split('_')
        if len(parts) >= 2:
            # 假设名字可能包含多个下划线分隔的部分
            person = '_'.join(parts[:-1])  # 除最后一部分外都是人名
            if person not in person_photos:
                person_photos[person] = []
            person_photos[person].append(img)

    # 过滤掉照片少于min_photos的人
    qualified_people = [(person, photos) for person, photos in person_photos.items()
                        if len(photos) >= min_photos]

    # 按照照片数量排序
    qualified_people.sort(key=lambda x: len(x[1]), reverse=True)

    # 选择前num_people个人
    selected_people = qualified_people[:num_people]

    print(f"已选择{len(selected_people)}个人，每人至少有{min_photos}张照片:")
    for person, photos in selected_people:
        print(f"  - {person}: {len(photos)}张照片")

        # 创建此人的目录
        person_output_dir = os.path.join(output_path, person)
        os.makedirs(person_output_dir, exist_ok=True)

        # 复制照片
        for photo in photos:
            src = os.path.join(lfw_path, photo)
            dst = os.path.join(person_output_dir, photo)
            shutil.copy2(src, dst)

    return output_path


def show_dataset_samples(dataset_path, num_people=5, num_photos=5):
    """显示数据集样本"""
    if not os.path.exists(dataset_path):
        print(f"错误: 未找到数据集路径 {dataset_path}")
        return

    people = sorted(os.listdir(dataset_path))

    plt.figure(figsize=(15, 10))
    for i, person in enumerate(people[:num_people]):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue

        photos = [f for f in os.listdir(person_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for j, photo in enumerate(photos[:num_photos]):
            photo_path = os.path.join(person_dir, photo)
            img = Image.open(photo_path)

            plt.subplot(num_people, num_photos, i * num_photos + j + 1)
            plt.imshow(img)
            plt.axis('off')
            if j == 0:
                plt.title(person, fontsize=8)

    plt.tight_layout()
    plt.savefig('lfw_samples.png')
    plt.show()


if __name__ == "__main__":
    # 设置LFW数据集路径 - 请修改为您的LFW路径
    lfw_path = "./lfw"  # 修改为您的LFW数据集目录

    # 创建子集
    subset_path = create_lfw_subset(lfw_path)

    if subset_path:
        # 显示样本
        show_dataset_samples(subset_path)
        print(f"\n数据集准备完成，保存在: {subset_path}")
        print("请在arcface_train.py中设置数据路径为: ./data/lfw_subset")