from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, list_images


COUNT_COLUMNS = {"image", "count"}
POINT_COLUMNS = {"image", "x", "y"}


def create_image_manifest(image_dir: Path, output_csv: Path) -> pd.DataFrame:
    images = list_images(image_dir)
    rows = [
        {
            "image": str(path.relative_to(image_dir)),
            "file_name": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }
        for path in images
    ]
    frame = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    frame.to_csv(output_csv, index=False)
    print(f"Wrote {len(frame)} image rows to {output_csv}")
    return frame


def validate_count_annotations(image_dir: Path, counts_csv: Path) -> pd.DataFrame:
    counts = pd.read_csv(counts_csv)
    missing = COUNT_COLUMNS - set(counts.columns)
    if missing:
        raise ValueError(f"Count CSV missing columns: {sorted(missing)}")
    if counts["image"].duplicated().any():
        dupes = counts.loc[counts["image"].duplicated(), "image"].tolist()
        raise ValueError(f"Duplicate image rows in count CSV: {dupes[:10]}")
    if (counts["count"] < 0).any():
        raise ValueError("Counts must be non-negative")

    image_set = {str(path.relative_to(image_dir)) for path in list_images(image_dir)}
    annotated = set(counts["image"].astype(str))
    missing_files = sorted(annotated - image_set)
    unannotated_files = sorted(image_set - annotated)

    print(f"Images on disk: {len(image_set)}")
    print(f"Count annotations: {len(annotated)}")
    print(f"Missing image files referenced by CSV: {len(missing_files)}")
    print(f"Images without count annotations: {len(unannotated_files)}")
    if missing_files:
        print("First missing files:", missing_files[:10])
    if unannotated_files:
        print("First unannotated files:", unannotated_files[:10])

    return counts


def validate_point_annotations(image_dir: Path, points_csv: Path) -> pd.DataFrame:
    points = pd.read_csv(points_csv)
    missing = POINT_COLUMNS - set(points.columns)
    if missing:
        raise ValueError(f"Point CSV missing columns: {sorted(missing)}")

    image_set = {str(path.relative_to(image_dir)) for path in list_images(image_dir)}
    annotated = set(points["image"].astype(str))
    missing_files = sorted(annotated - image_set)
    if missing_files:
        raise ValueError(f"Point CSV references missing image files: {missing_files[:10]}")
    if points[["x", "y"]].isna().any().any():
        raise ValueError("Point coordinates must not contain missing values")

    print(f"Point rows: {len(points)}")
    print(f"Images with point annotations: {points['image'].nunique()}")
    return points

