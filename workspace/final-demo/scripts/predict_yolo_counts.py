from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
]


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_id(names: dict[int, str] | list[str], class_name: str) -> int:
    normalized = class_name.lower()
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)
    for class_id, name in items:
        if str(name).lower() == normalized:
            return int(class_id)
    available = ", ".join(str(name) for _, name in items)
    raise ValueError(f"Class {class_name!r} not found. Available classes: {available}")


def draw_predictions(
    image_path: Path,
    boxes,
    names: dict[int, str],
    target_class_id: int,
    output_path: Path,
    draw_all_classes: bool,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_path}")

    for box in boxes:
        class_id = int(box.cls[0].cpu().item())
        if not draw_all_classes and class_id != target_class_id:
            continue
        confidence = float(box.conf[0].cpu().item())
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].cpu().tolist()]
        color = (0, 220, 0) if class_id == target_class_id else PALETTE[class_id % len(PALETTE)]
        label = f"{class_id}:{names[class_id]} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_width = min(image.shape[1] - 1, x1 + 8 * len(label) + 8)
        cv2.rectangle(image, (x1, max(0, y1 - 22)), (label_width, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict per-image target-class counts with a YOLO model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to best.pt or another YOLO checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Seabirds.v6i.yolo26"))
    parser.add_argument("--split", default="test", help="Dataset split name, for example valid or test.")
    parser.add_argument("--class-name", default="puffin", help="Target class to count.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--save-visuals", action="store_true", help="Save prediction visualizations.")
    parser.add_argument("--draw-all-classes", action="store_true", help="Draw non-target predictions too.")
    args = parser.parse_args()

    image_dir = args.dataset_root / args.split / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing split image directory: {image_dir}")
    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")

    model = YOLO(str(args.model))
    names = model.names
    target_class_id = find_class_id(names, args.class_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = args.output_dir / "visuals"

    prediction_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []

    for index, image_path in enumerate(images, start=1):
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        target_count = 0
        for box_index, box in enumerate(result.boxes):
            class_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
            if class_id == target_class_id:
                target_count += 1
            detection_rows.append(
                {
                    "split": args.split,
                    "image": image_path.name,
                    "box_index": box_index,
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "is_target": class_id == target_class_id,
                }
            )

        visual_path = ""
        if args.save_visuals:
            visual_path = str(visuals_dir / f"{image_path.stem}_pred.jpg")
            draw_predictions(
                image_path=image_path,
                boxes=result.boxes,
                names=names,
                target_class_id=target_class_id,
                output_path=Path(visual_path),
                draw_all_classes=args.draw_all_classes,
            )

        prediction_rows.append(
            {
                "split": args.split,
                "image": image_path.name,
                "target_class": args.class_name,
                "target_class_id": target_class_id,
                "target_count": target_count,
                "all_detections": len(result.boxes),
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "visual": visual_path,
            }
        )
        print(f"[{index:>4}/{len(images)}] {image_path.name}: {target_count} {args.class_name}")

    predictions_path = args.output_dir / f"{args.split}_predictions.csv"
    detections_path = args.output_dir / f"{args.split}_detections.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with detections_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "image",
            "box_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "is_target",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detection_rows)

    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote detections to {detections_path}")
    if args.save_visuals:
        print(f"Wrote visuals to {visuals_dir}")


if __name__ == "__main__":
    main()
