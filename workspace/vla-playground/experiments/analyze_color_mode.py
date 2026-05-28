from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "seabirds_annotation_300_labeled.csv"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "seabirds_annotation_300_labeled_color.csv"
DEFAULT_PROBE_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_probe"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "color_mode_analysis"
DEFAULT_TASKS = ["distance", "contains_puffin", "density", "difficulty", "occlusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate image color mode and analyze probe errors by color mode.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Labeled manifest CSV.")
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST, help="Manifest with color fields.")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR, help="Probe prediction directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Analysis output directory.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Probe tasks to analyze.")
    parser.add_argument("--method", default="linear", choices=["knn", "linear"], help="Probe method to analyze.")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def estimate_color(path: Path) -> tuple[str, float, float]:
    image = Image.open(path).convert("RGB")
    image.thumbnail((256, 256))
    arr = np.asarray(image).astype(np.float32) / 255.0
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    channel_delta = np.mean(np.abs(red - green) + np.abs(red - blue) + np.abs(green - blue)) / 3.0
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = np.mean((max_channel - min_channel) / np.maximum(max_channel, 1e-6))

    # The thresholds are intentionally conservative. "low_color" catches tinted or washed-out images
    # that are not mathematically grayscale but may still weaken color-based bird cues.
    if channel_delta < 0.015 and saturation < 0.08:
        mode = "grayscale"
    elif channel_delta < 0.045 and saturation < 0.18:
        mode = "low_color"
    else:
        mode = "color"
    return mode, float(channel_delta), float(saturation)


def add_color_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        mode, channel_delta, saturation = estimate_color(Path(row["image_path"]))
        item["color_mode_auto"] = mode
        item["color_channel_delta"] = f"{channel_delta:.6f}"
        item["color_saturation"] = f"{saturation:.6f}"
        enriched.append(item)
        if index % 50 == 0:
            print(f"Processed color stats: {index}/{len(rows)}")
    return enriched


def summarize_color(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        mode = row["color_mode_auto"]
        if mode not in summary:
            summary[mode] = {"color_mode_auto": mode, "count": 0, "contains_puffin_yes": 0, "contains_puffin_no": 0}
        summary[mode]["count"] += 1
        if row.get("contains_puffin") == "yes":
            summary[mode]["contains_puffin_yes"] += 1
        elif row.get("contains_puffin") == "no":
            summary[mode]["contains_puffin_no"] += 1
    return [{key: str(value) for key, value in row.items()} for row in summary.values()]


def analyze_probe_by_color(
    enriched_rows: list[dict[str, str]],
    probe_dir: Path,
    tasks: list[str],
    method: str,
) -> list[dict[str, str]]:
    by_id = {row["image_id"]: row for row in enriched_rows}
    output: list[dict[str, str]] = []
    for task in tasks:
        prediction_path = probe_dir / f"{task}_predictions.csv"
        if not prediction_path.exists():
            continue
        predictions = load_csv(prediction_path)
        buckets: dict[tuple[str, str], dict[str, int]] = {}
        for row in predictions:
            if row.get("method") != method:
                continue
            image_id = row.get("image_id", "")
            color_mode = by_id.get(image_id, {}).get("color_mode_auto", "missing")
            key = (task, color_mode)
            if key not in buckets:
                buckets[key] = {"total": 0, "correct": 0}
            buckets[key]["total"] += 1
            if row.get("correct") == "True":
                buckets[key]["correct"] += 1
        for (task_name, color_mode), stats in sorted(buckets.items()):
            total = stats["total"]
            correct = stats["correct"]
            output.append(
                {
                    "task": task_name,
                    "method": method,
                    "color_mode_auto": color_mode,
                    "total": str(total),
                    "correct": str(correct),
                    "accuracy": f"{correct / total:.4f}" if total else "",
                }
            )
    return output


def main() -> None:
    args = parse_args()
    rows = load_csv(args.manifest.resolve())
    enriched = add_color_fields(rows)
    write_csv(enriched, args.output_manifest.resolve())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summarize_color(enriched), output_dir / "color_mode_summary.csv")
    write_csv(analyze_probe_by_color(enriched, args.probe_dir.resolve(), args.tasks, args.method), output_dir / "probe_by_color_mode.csv")

    print(f"Input rows: {len(rows)}")
    print(f"Manifest with color fields: {args.output_manifest.resolve()}")
    print(f"Color summary: {output_dir / 'color_mode_summary.csv'}")
    print(f"Probe analysis: {output_dir / 'probe_by_color_mode.csv'}")


if __name__ == "__main__":
    main()
