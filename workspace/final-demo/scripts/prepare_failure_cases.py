from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd


def read_errors(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image", "ground_truth", "prediction", "error", "abs_error"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame.rename(
        columns={
            "prediction": f"{prefix}_prediction",
            "error": f"{prefix}_error",
            "abs_error": f"{prefix}_abs_error",
            "relative_error": f"{prefix}_relative_error",
        }
    )


def classify_case(row: pd.Series) -> str:
    exp001_error = float(row["exp001_error"])
    exp002_error = float(row["exp002_error"])
    if exp001_error < 0 and exp002_error < 0:
        return "both_under_count"
    if exp001_error > 0 and exp002_error > 0:
        return "both_over_count"
    if exp001_error != 0 and exp002_error == 0:
        return "exp002_fixed_exp001_error"
    if exp001_error == 0 and exp002_error != 0:
        return "exp002_new_error"
    if exp001_error < 0 and exp002_error > 0:
        return "under_to_over"
    if exp001_error > 0 and exp002_error < 0:
        return "over_to_under"
    if exp001_error == 0 and exp002_error == 0:
        return "both_correct"
    return "mixed"


def visual_path(visuals_dir: Path, image_name: str) -> Path:
    return visuals_dir / f"{Path(image_name).stem}_pred.jpg"


def copy_if_exists(source: Path, destination: Path) -> str:
    if not source.exists():
        return "missing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "copied"


def write_notes(path: Path, row: pd.Series) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {path.parent.name}",
                "",
                "## 基本信息",
                "",
                f"- image: `{row['image']}`",
                f"- ground truth: {row['ground_truth']}",
                f"- YOLO26n prediction: {row['exp001_prediction']} (error {row['exp001_error']})",
                f"- YOLO11n prediction: {row['exp002_prediction']} (error {row['exp002_error']})",
                f"- case type: `{row['case_type']}`",
                "",
                "## 人工观察",
                "",
                "- 失败类型：",
                "- 场景特征：",
                "- 可能原因：",
                "- 报告可用结论：",
                "",
                "## 文件",
                "",
                "- `original.jpg`",
                "- `exp001_yolo26n_pred.jpg`",
                "- `exp002_yolo11n_pred.jpg`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare side-by-side failure cases for two experiments.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Seabirds.v6i.yolo26"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--exp001-errors",
        type=Path,
        default=Path("outputs/evaluation/exp001_yolo26n_baseline/test/per_image_errors.csv"),
    )
    parser.add_argument(
        "--exp002-errors",
        type=Path,
        default=Path("outputs/evaluation/exp002_yolo11n_baseline/test/per_image_errors.csv"),
    )
    parser.add_argument(
        "--exp001-visuals",
        type=Path,
        default=Path("outputs/predictions/exp001_yolo26n_baseline/test/visuals"),
    )
    parser.add_argument(
        "--exp002-visuals",
        type=Path,
        default=Path("outputs/predictions/exp002_yolo11n_baseline/test/visuals"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/failure_cases/exp001_vs_exp002"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    exp001 = read_errors(args.exp001_errors, "exp001")
    exp002 = read_errors(args.exp002_errors, "exp002")
    merged = exp001.merge(
        exp002[
            [
                "image",
                "ground_truth",
                "exp002_prediction",
                "exp002_error",
                "exp002_abs_error",
                "exp002_relative_error",
            ]
        ],
        on=["image", "ground_truth"],
        how="outer",
    )
    merged = merged.fillna(
        {
            "exp001_prediction": 0,
            "exp001_error": 0,
            "exp001_abs_error": 0,
            "exp002_prediction": 0,
            "exp002_error": 0,
            "exp002_abs_error": 0,
        }
    )
    merged["case_type"] = merged.apply(classify_case, axis=1)
    merged["max_abs_error"] = merged[["exp001_abs_error", "exp002_abs_error"]].max(axis=1)
    merged["combined_abs_error"] = merged["exp001_abs_error"] + merged["exp002_abs_error"]
    merged = merged.sort_values(
        ["max_abs_error", "combined_abs_error", "image"],
        ascending=[False, False, True],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "comparison_summary.csv"
    merged.to_csv(comparison_path, index=False)

    selected = merged[merged["combined_abs_error"] > 0].head(args.top_k)
    manifest_rows: list[dict[str, object]] = []
    image_dir = args.dataset_root / args.split / "images"

    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        case_dir = args.output_dir / f"case_{index:03d}_{row['case_type']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        image_name = str(row["image"])
        original_status = copy_if_exists(image_dir / image_name, case_dir / "original.jpg")
        exp001_status = copy_if_exists(
            visual_path(args.exp001_visuals, image_name),
            case_dir / "exp001_yolo26n_pred.jpg",
        )
        exp002_status = copy_if_exists(
            visual_path(args.exp002_visuals, image_name),
            case_dir / "exp002_yolo11n_pred.jpg",
        )
        write_notes(case_dir / "notes.md", row)
        manifest_rows.append(
            {
                "case": case_dir.name,
                "image": image_name,
                "ground_truth": row["ground_truth"],
                "exp001_prediction": row["exp001_prediction"],
                "exp001_error": row["exp001_error"],
                "exp002_prediction": row["exp002_prediction"],
                "exp002_error": row["exp002_error"],
                "case_type": row["case_type"],
                "original": original_status,
                "exp001_visual": exp001_status,
                "exp002_visual": exp002_status,
            }
        )

    with (args.output_dir / "case_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Wrote comparison summary to {comparison_path}")
    print(f"Wrote {len(manifest_rows)} case folders to {args.output_dir}")


if __name__ == "__main__":
    main()
