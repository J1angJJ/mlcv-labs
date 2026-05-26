from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import ensure_dir, list_images


def _load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it only if you decide the "
            "YOLO workflow is needed, then rerun this command."
        ) from exc
    return YOLO(model_name)


def _find_class_id(names: dict[int, str], class_name: str) -> int:
    normalized = class_name.lower()
    for class_id, name in names.items():
        if str(name).lower() == normalized:
            return int(class_id)
    available = ", ".join(str(name) for name in names.values())
    raise ValueError(f"Class '{class_name}' not found in model names: {available}")


def run_yolo_detection(
    image_dir: Path,
    model_name: str,
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    class_name: str = "bird",
) -> pd.DataFrame:
    images = list_images(image_dir)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    ensure_dir(output_dir)
    model = _load_yolo(model_name)

    rows = []
    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        result = model(image_bgr, conf=conf, iou=iou, verbose=False)[0]
        target_class_id = _find_class_id(result.names, class_name)
        boxes = result.boxes

        target_boxes = []
        for box in boxes:
            class_id = int(box.cls[0].cpu().item())
            if class_id == target_class_id:
                target_boxes.append(box)

        figure_path = output_dir / f"{image_path.stem}_detections.png"
        _save_detection_figure(
            image_bgr=image_bgr,
            boxes=boxes,
            target_class_id=target_class_id,
            class_names=result.names,
            output_path=figure_path,
            title=f"{image_path.name}: {len(target_boxes)} {class_name}(s)",
        )

        rows.append({
            "image": str(image_path.relative_to(image_dir)),
            "target_class": class_name,
            "target_count": len(target_boxes),
            "all_detections": len(boxes),
            "conf": conf,
            "iou": iou,
            "figure": str(figure_path),
        })
        print(f"{image_path.name}: {len(target_boxes)} {class_name}(s), {len(boxes)} total detections")

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "detection_counts.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote detection summary to {summary_path}")
    return summary


def _save_detection_figure(
    image_bgr,
    boxes,
    target_class_id: int,
    class_names: dict[int, str],
    output_path: Path,
    title: str,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image_rgb)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().item())
        confidence = float(box.conf[0].cpu().item())
        is_target = class_id == target_class_id
        color = "lime" if is_target else "deepskyblue"
        label = f"{class_names[class_id]} {confidence:.2f}"

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2 if is_target else 1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 1},
        )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
