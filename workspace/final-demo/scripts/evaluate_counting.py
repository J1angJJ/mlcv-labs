from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puffin_counting.evaluation import evaluate_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate puffin counting predictions for one dataset split.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="image_counts.csv from dataset audit.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions CSV from predict_yolo_counts.py.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-name", default="puffin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-column", default="target_count")
    args = parser.parse_args()

    ground_truth = pd.read_csv(args.ground_truth)
    predictions = pd.read_csv(args.predictions)

    if "split" in ground_truth.columns:
        ground_truth = ground_truth[ground_truth["split"] == args.split].copy()
    if "class_name" in ground_truth.columns:
        ground_truth = ground_truth[ground_truth["class_name"] == args.class_name].copy()
    if "split" in predictions.columns:
        predictions = predictions[predictions["split"] == args.split].copy()
    if "target_class" in predictions.columns:
        predictions = predictions[predictions["target_class"] == args.class_name].copy()

    if ground_truth.empty:
        raise RuntimeError(f"No ground-truth rows left after filtering split={args.split!r}, class={args.class_name!r}")
    if predictions.empty:
        raise RuntimeError(f"No prediction rows left after filtering split={args.split!r}, class={args.class_name!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_gt = args.output_dir / "ground_truth_filtered.csv"
    filtered_pred = args.output_dir / "predictions_filtered.csv"
    ground_truth.to_csv(filtered_gt, index=False)
    predictions.to_csv(filtered_pred, index=False)

    evaluate_counts(
        ground_truth_csv=filtered_gt,
        predictions_csv=filtered_pred,
        output_dir=args.output_dir,
        prediction_column=args.prediction_column,
    )


if __name__ == "__main__":
    main()
