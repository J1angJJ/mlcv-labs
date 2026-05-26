import os
import urllib.request
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR = Path("test_images")
IMAGE_DIR.mkdir(exist_ok=True)

BIRD_CLASS_ID = 14  # COCO数据集中bird的类别ID


def download_test_images():
    """准备测试图片"""
    downloaded = []
    
    # 1. 使用ultralytics自带的图片
    from ultralytics.utils import ASSETS
    import shutil
    
    for asset_name in ["bus.jpg", "zidane.jpg"]:
        asset_path = Path(ASSETS) / asset_name
        if asset_path.exists():
            local_path = IMAGE_DIR / asset_name
            if not local_path.exists():
                shutil.copy(asset_path, local_path)
            downloaded.append(local_path)
    
    # 2. 尝试下载真实鸟类图片
    bird_urls = [
        ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01530575_magpie.JPEG", "bird_magpie.jpg"),
        ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01532829_house_finch.JPEG", "bird_finch.jpg"),
    ]
    
    for url, filename in bird_urls:
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as response:
                    with open(filepath, 'wb') as f:
                        f.write(response.read())
                downloaded.append(filepath)
            except Exception:
                pass
        else:
            downloaded.append(filepath)
    
    return downloaded


# ============================================================
# 实验1：YOLOv8目标检测 + 鸟类计数
# ============================================================
def experiment_1_detection(model, images):
    print("\n" + "="*60)
    print("Experiment 1: Object Detection & Bird Counting")
    print("="*60)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False)
        result = results[0]
        boxes = result.boxes

        # 统计bird类别
        bird_mask = boxes.cls == BIRD_CLASS_ID
        bird_count = int(bird_mask.sum())
        total_count = len(boxes)

        # 可视化：所有检测框都画出来，bird用绿色高亮，其他用蓝色
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.imshow(img_rgb)

        names = result.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls_id = int(box.cls[0].cpu().item())
            cls_name = names[cls_id]
            is_bird = (cls_id == BIRD_CLASS_ID)

            color = 'lime' if is_bird else 'deepskyblue'
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{cls_name} {conf:.2f}', color=color,
                    fontsize=9, backgroundcolor='black')

        ax.set_title(f'{img_path.name} | Total: {total_count} | Birds: {bird_count}',
                     fontsize=12, family='DejaVu Sans')
        ax.axis('off')
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"exp1_detection_{img_path.stem}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  {img_path.name}: {total_count} objects, {bird_count} birds -> {save_path.name}")


# ============================================================
# 实验2：特征图可视化
# ============================================================
def experiment_2_feature_visualization(model, images):
    print("\n" + "="*60)
    print("Experiment 2: Backbone Feature Map Visualization")
    print("="*60)

    import torch

    img_path = images[0]
    img = cv2.imread(str(img_path))
    if img is None:
        return

    device = next(model.model.parameters()).device

    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    backbone = model.model.model[:10]  # YOLOv8的前10层作为backbone

    features = []
    x = img_tensor
    with torch.no_grad():
        for i, layer in enumerate(backbone):
            x = layer(x)
            if i in [1, 3, 5, 7, 9]:  # 选取不同深度的层
                features.append((f"Layer {i}", x.clone()))

    fig, axes = plt.subplots(1, min(5, len(features)), figsize=(20, 4))
    if len(features) == 1:
        axes = [axes]

    for idx, (name, feat) in enumerate(features[:5]):
        if idx >= len(axes):
            break
        feat_map = feat[0].mean(dim=0).cpu().numpy()
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        axes[idx].imshow(feat_map, cmap='jet')
        axes[idx].set_title(f"{name}\n{feat.shape[1]}ch, {feat.shape[2]}x{feat.shape[3]}", fontsize=9)
        axes[idx].axis('off')

    plt.suptitle("YOLOv8 Backbone Feature Maps (Shallow -> Deep Layers)", fontsize=12)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "exp2_feature_maps.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 实验3：NMS阈值对比
# ============================================================
def experiment_3_nms_comparison(model, images):
    print("\n" + "="*60)
    print("Experiment 3: NMS IoU Threshold Comparison")
    print("="*60)

    iou_thresholds = [0.3, 0.5, 0.7]
    img_path = images[0]
    img = cv2.imread(str(img_path))
    if img is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, iou_thresh in enumerate(iou_thresholds):
        results = model(img, iou=iou_thresh, conf=0.25, verbose=False)
        result = results[0]
        boxes = result.boxes
        total_count = len(boxes)

        axes[idx].imshow(img_rgb)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1.5, edgecolor='red', facecolor='none')
            axes[idx].add_patch(rect)

        axes[idx].set_title(f"NMS IoU={iou_thresh}\nDetections: {total_count}", fontsize=11)
        axes[idx].axis('off')

    plt.suptitle("Effect of NMS IoU Threshold on Detection Count", fontsize=13)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "exp3_nms_comparison.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  IoU thresholds: {iou_thresholds} -> {save_path.name}")


# ============================================================
# 实验4：高斯密度图计数
# ============================================================
def experiment_4_density_map():
    print("\n" + "="*60)
    print("Experiment 4: Density Map Counting Demo")
    print("="*60)

    from scipy.ndimage import gaussian_filter

    H, W = 300, 400
    # 模拟标注点（假设这些是puffin的位置）
    np.random.seed(42)
    n_birds = 25
    points_x = np.random.randint(30, W-30, n_birds)
    points_y = np.random.randint(30, H-30, n_birds)

    # 生成密度图：每个点放一个高斯核
    density_map = np.zeros((H, W), dtype=np.float32)
    for x, y in zip(points_x, points_y):
        if 0 <= y < H and 0 <= x < W:
            density_map[y, x] += 1.0

    # 用高斯滤波平滑
    sigma = 15
    density_map = gaussian_filter(density_map, sigma=sigma)

    # 积分得到计数
    estimated_count = density_map.sum()

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 原始点标注
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)
    axes[0].scatter(points_x, points_y, c='red', s=30, zorder=5)
    axes[0].set_title(f"Point Annotations\nGT Count: {n_birds}", fontsize=11)
    axes[0].set_facecolor('lightgray')
    axes[0].set_aspect('equal')

    # 密度图
    im = axes[1].imshow(density_map, cmap='jet', interpolation='bilinear')
    axes[1].set_title(f"Density Map (sigma={sigma})", fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 积分计数
    axes[2].bar(['GT Count', 'Estimated'], [n_birds, estimated_count],
                color=['steelblue', 'coral'])
    axes[2].set_title(f"Count: GT={n_birds}, Est={estimated_count:.1f}", fontsize=11)
    axes[2].set_ylabel("Number of birds")

    plt.suptitle("Density Map Counting: Point Annotation -> Gaussian Density -> Integration",
                 fontsize=12)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "exp4_density_map.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  GT: {n_birds}, Estimated: {estimated_count:.1f} -> {save_path.name}")


# ============================================================
# 实验5：Fine-tune演示（使用COCO128小数据集）
# ============================================================
def experiment_5_finetune():
    print("\n" + "="*60)
    print("Experiment 5: Fine-tune Demo (COCO128, 2 epochs)")
    print("="*60)

    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')

    results = model.train(
        data='coco128.yaml',
        epochs=2,
        imgsz=320,
        batch=8,
        device='0' if __import__('torch').cuda.is_available() else 'cpu',
        workers=2,
        project=str(OUTPUT_DIR / 'finetune_demo'),
        name='train',
        exist_ok=True,
        verbose=True,
    )

    print(f"\n  Results: {OUTPUT_DIR / 'finetune_demo' / 'train'}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("="*60)
    print("  TA10 Lab: Bird Detection & Counting Pipeline")
    print("="*60)

    print("\n[Step 0] Preparing test images...")
    images = download_test_images()
    print(f"  {len(images)} images ready")

    print("\n[Step 1] Loading YOLOv8n pretrained model...")
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print("  Done")

    experiment_1_detection(model, images)
    experiment_2_feature_visualization(model, images)
    experiment_3_nms_comparison(model, images)
    experiment_4_density_map()
    experiment_5_finetune()

    print("\n" + "="*60)
    print("  All done! Check outputs/ for results.")
    print("="*60)


if __name__ == "__main__":
    main()
