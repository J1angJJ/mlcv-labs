from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import yaml


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
    (188, 189, 34),
    (23, 190, 207),
]


def load_class_names(dataset_root: Path) -> list[str]:
    with (dataset_root / "data.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return [str(name) for name in names]


def discover_splits(dataset_root: Path, requested_split: str) -> list[str]:
    if requested_split != "all":
        return [requested_split]
    return [
        split
        for split in ["train", "valid", "val", "test"]
        if (dataset_root / split / "images").is_dir() and (dataset_root / split / "labels").is_dir()
    ]


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_class_filter(raw_values: list[str], class_names: list[str]) -> set[int] | None:
    if not raw_values:
        return None
    selected: set[int] = set()
    for raw_value in raw_values:
        for token in raw_value.split(","):
            value = token.strip()
            if not value:
                continue
            if value.isdigit():
                class_id = int(value)
            else:
                class_id = class_names.index(value)
            if class_id < 0 or class_id >= len(class_names):
                raise ValueError(f"Class id out of range: {class_id}")
            selected.add(class_id)
    return selected


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
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
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def draw_boxes(
    image_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    class_names: list[str],
    class_filter: set[int] | None,
) -> tuple[object, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_path}")

    height, width = image.shape[:2]
    drawn = 0
    for class_id, x_center, y_center, box_width, box_height in boxes:
        if class_filter is not None and class_id not in class_filter:
            continue
        x1 = int(round((x_center - box_width / 2) * width))
        y1 = int(round((y_center - box_height / 2) * height))
        x2 = int(round((x_center + box_width / 2) * width))
        y2 = int(round((y_center + box_height / 2) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        color = PALETTE[class_id % len(PALETTE)]
        label = f"{class_id}:{class_names[class_id] if class_id < len(class_names) else 'unknown'}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(image, (x1, max(0, y1 - 22)), (min(width - 1, x1 + 8 * len(label) + 8), y1), color, -1)
        cv2.putText(image, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        drawn += 1
    return image, drawn


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly draw YOLO boxes for label-quality inspection.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Seabirds.v6i.yolo26"))
    parser.add_argument("--split", default="valid", help="train, valid, test, or all")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/label_audit"))
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Optional class names or ids to sample and draw, for example: --classes puffin or --classes 4",
    )
    parser.add_argument("--include-empty", action="store_true", help="Allow images with no selected boxes.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    class_names = load_class_names(dataset_root)
    class_filter = parse_class_filter(args.classes, class_names)
    splits = discover_splits(dataset_root, args.split)
    if not splits:
        raise RuntimeError(f"No splits found for {args.split!r} under {dataset_root}")

    candidates: list[tuple[str, Path, Path, int]] = []
    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        for image_path in image_files(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = read_boxes(label_path)
            selected_count = sum(1 for box in boxes if class_filter is None or box[0] in class_filter)
            if selected_count > 0 or args.include_empty:
                candidates.append((split, image_path, label_path, selected_count))

    if not candidates:
        raise RuntimeError("No candidate images matched the requested filters.")

    rng = random.Random(args.seed)
    sample_size = min(args.count, len(candidates))
    sampled = rng.sample(candidates, sample_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for index, (split, image_path, label_path, selected_count) in enumerate(sampled, start=1):
        boxes = read_boxes(label_path)
        rendered, drawn_count = draw_boxes(image_path, boxes, class_names, class_filter)
        output_name = f"{index:03d}_{split}_{image_path.stem}_boxed.jpg"
        output_path = args.output_dir / output_name
        cv2.imwrite(str(output_path), rendered)
        manifest_rows.append(
            {
                "index": index,
                "split": split,
                "image": image_path.name,
                "label": label_path.name,
                "selected_boxes": selected_count,
                "drawn_boxes": drawn_count,
                "output": output_name,
            }
        )

    with (args.output_dir / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "split", "image", "label", "selected_boxes", "drawn_boxes", "output"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Sampled {sample_size} images from {len(candidates)} candidates.")
    print(f"Wrote boxed images and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
