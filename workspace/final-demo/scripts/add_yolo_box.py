from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml


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


def load_class_names(dataset_root: Path) -> list[str]:
    with (dataset_root / "data.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return [str(name) for name in names]


def class_to_id(value: str, class_names: list[str]) -> int:
    if value.isdigit():
        class_id = int(value)
    else:
        class_id = class_names.index(value)
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"class id out of range: {class_id}")
    return class_id


def pixel_box_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    left = max(0.0, min(float(image_width - 1), min(x1, x2)))
    right = max(0.0, min(float(image_width - 1), max(x1, x2)))
    top = max(0.0, min(float(image_height - 1), min(y1, y2)))
    bottom = max(0.0, min(float(image_height - 1), max(y1, y2)))
    if right <= left or bottom <= top:
        raise ValueError("box must have positive width and height")
    box_width = right - left
    box_height = bottom - top
    x_center = left + box_width / 2
    y_center = top + box_height / 2
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        boxes.append((int(float(parts[0])), *(float(value) for value in parts[1:5])))
    return boxes


def draw_preview(
    image_path: Path,
    label_path: Path,
    class_names: list[str],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_path}")
    image_height, image_width = image.shape[:2]

    for class_id, x_center, y_center, box_width, box_height in read_yolo_boxes(label_path):
        x1 = int(round((x_center - box_width / 2) * image_width))
        y1 = int(round((y_center - box_height / 2) * image_height))
        x2 = int(round((x_center + box_width / 2) * image_width))
        y2 = int(round((y_center + box_height / 2) * image_height))
        color = PALETTE[class_id % len(PALETTE)]
        label = f"{class_id}:{class_names[class_id] if class_id < len(class_names) else 'unknown'}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(image, (x1, max(0, y1 - 22)), (min(image_width - 1, x1 + 8 * len(label) + 8), y1), color, -1)
        cv2.putText(image, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one pixel-defined bounding box to a YOLO label file.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Seabirds.v6i.yolo26"))
    parser.add_argument("--split", required=True, help="Dataset split, for example train, valid, or test.")
    parser.add_argument("--image", required=True, help="Image filename inside the split/images directory.")
    parser.add_argument("--class-name", default="puffin", help="Class name or numeric class id.")
    parser.add_argument("--xyxy", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output-preview", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    class_names = load_class_names(dataset_root)
    class_id = class_to_id(args.class_name, class_names)

    image_path = dataset_root / args.split / "images" / args.image
    label_path = dataset_root / args.split / "labels" / f"{Path(args.image).stem}.txt"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not label_path.exists():
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_path}")
    image_height, image_width = image.shape[:2]
    x_center, y_center, box_width, box_height = pixel_box_to_yolo(*args.xyxy, image_width, image_height)
    new_line = f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"

    if not args.no_backup:
        backup_path = label_path.with_suffix(label_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)

    existing = label_path.read_text(encoding="utf-8").splitlines()
    existing.append(new_line)
    label_path.write_text("\n".join(line for line in existing if line.strip()) + "\n", encoding="utf-8")

    if args.output_preview is not None:
        draw_preview(image_path, label_path, class_names, args.output_preview)

    print(f"Appended: {new_line}")
    print(f"Label file: {label_path}")
    if args.output_preview is not None:
        print(f"Preview: {args.output_preview}")


if __name__ == "__main__":
    main()
