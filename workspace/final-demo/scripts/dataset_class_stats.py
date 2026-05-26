from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_names(dataset_root: Path) -> list[str]:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing data.yaml: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("data.yaml must contain a names list or dict")


def discover_splits(dataset_root: Path) -> list[str]:
    preferred = ["train", "valid", "val", "test"]
    splits = []
    for split in preferred:
        if (dataset_root / split / "images").is_dir() and (dataset_root / split / "labels").is_dir():
            splits.append(split)
    return splits


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_label_file(label_path: Path, num_classes: int) -> tuple[list[tuple[int, float]], list[str]]:
    boxes: list[tuple[int, float]] = []
    errors: list[str] = []
    if not label_path.exists():
        return boxes, errors

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"{label_path}:{line_number}: expected at least 5 columns")
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric YOLO label")
            continue
        if class_id < 0 or class_id >= num_classes:
            errors.append(f"{label_path}:{line_number}: class id {class_id} outside [0, {num_classes - 1}]")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"{label_path}:{line_number}: non-positive box size")
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{label_path}:{line_number}: normalized coordinates outside expected range")
            continue
        boxes.append((class_id, width * height))
    return boxes, errors


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize class statistics for a YOLO-format dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Seabirds.v6i.yolo26"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument(
        "--count-class",
        default="puffin",
        help="Class name or numeric class id used to generate image_counts.csv.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir
    class_names = load_class_names(dataset_root)
    splits = discover_splits(dataset_root)
    if not splits:
        raise RuntimeError(f"No YOLO splits found under {dataset_root}")

    if str(args.count_class).isdigit():
        count_class_id = int(args.count_class)
    else:
        count_class_id = class_names.index(args.count_class)

    split_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    image_count_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []

    print(f"Dataset: {dataset_root}")
    print(f"Classes: {dict(enumerate(class_names))}")
    print()

    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        images = image_files(images_dir)
        image_stems = {image.stem for image in images}
        label_files = sorted(labels_dir.glob("*.txt"))
        label_stems = {label.stem for label in label_files}

        class_box_counts: Counter[int] = Counter()
        class_image_counts: Counter[int] = Counter()
        class_areas: dict[int, list[float]] = defaultdict(list)
        total_boxes = 0
        missing_label_files = 0
        empty_label_files = 0
        images_with_labels = 0

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_label_files += 1
                image_count_rows.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "count": 0,
                        "class_id": count_class_id,
                        "class_name": class_names[count_class_id],
                        "label_status": "missing",
                    }
                )
                continue

            boxes, errors = parse_label_file(label_path, len(class_names))
            for error in errors:
                invalid_rows.append({"split": split, "label_file": label_path.name, "error": error})

            if not boxes:
                empty_label_files += 1
            else:
                images_with_labels += 1

            image_class_ids = {class_id for class_id, _ in boxes}
            for class_id in image_class_ids:
                class_image_counts[class_id] += 1
            for class_id, box_area in boxes:
                class_box_counts[class_id] += 1
                class_areas[class_id].append(box_area)
            total_boxes += len(boxes)

            image_count_rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "count": sum(1 for class_id, _ in boxes if class_id == count_class_id),
                    "class_id": count_class_id,
                    "class_name": class_names[count_class_id],
                    "label_status": "ok",
                }
            )

        orphan_label_files = len(label_stems - image_stems)
        split_rows.append(
            {
                "split": split,
                "image_files": len(images),
                "label_files": len(label_files),
                "images_with_labels": images_with_labels,
                "empty_label_files": empty_label_files,
                "missing_label_files": missing_label_files,
                "orphan_label_files": orphan_label_files,
                "total_boxes": total_boxes,
                "invalid_lines": sum(1 for row in invalid_rows if row["split"] == split),
            }
        )

        for class_id, class_name in enumerate(class_names):
            boxes = class_box_counts[class_id]
            images_with_class = class_image_counts[class_id]
            class_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "boxes": boxes,
                    "images_with_class": images_with_class,
                    "avg_boxes_per_image_with_class": round(boxes / images_with_class, 4)
                    if images_with_class
                    else 0,
                    "mean_normalized_box_area": round(mean(class_areas[class_id]), 8)
                    if class_areas[class_id]
                    else 0,
                }
            )

        print(
            f"{split:>5}: images={len(images)}, labels={len(label_files)}, "
            f"boxes={total_boxes}, missing_labels={missing_label_files}, "
            f"empty_labels={empty_label_files}, invalid_lines={split_rows[-1]['invalid_lines']}"
        )
        for class_id, class_name in enumerate(class_names):
            print(
                f"       {class_id}: {class_name:<10} boxes={class_box_counts[class_id]:>5} "
                f"images={class_image_counts[class_id]:>4}"
            )
        print()

    write_csv(
        output_dir / "split_stats.csv",
        split_rows,
        [
            "split",
            "image_files",
            "label_files",
            "images_with_labels",
            "empty_label_files",
            "missing_label_files",
            "orphan_label_files",
            "total_boxes",
            "invalid_lines",
        ],
    )
    write_csv(
        output_dir / "class_stats.csv",
        class_rows,
        [
            "split",
            "class_id",
            "class_name",
            "boxes",
            "images_with_class",
            "avg_boxes_per_image_with_class",
            "mean_normalized_box_area",
        ],
    )
    write_csv(
        output_dir / "image_counts.csv",
        image_count_rows,
        ["split", "image", "count", "class_id", "class_name", "label_status"],
    )
    write_csv(output_dir / "invalid_labels.csv", invalid_rows, ["split", "label_file", "error"])

    summary = {
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "count_class": {"id": count_class_id, "name": class_names[count_class_id]},
        "splits": split_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote audit files to {output_dir}")


if __name__ == "__main__":
    main()
