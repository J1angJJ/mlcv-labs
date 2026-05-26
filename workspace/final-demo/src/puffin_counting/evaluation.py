from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import ensure_dir


def evaluate_counts(
    ground_truth_csv: Path,
    predictions_csv: Path,
    output_dir: Path,
    prediction_column: str = "target_count",
) -> pd.DataFrame:
    gt = pd.read_csv(ground_truth_csv)
    pred = pd.read_csv(predictions_csv)
    required_gt = {"image", "count"}
    required_pred = {"image", prediction_column}
    if missing := required_gt - set(gt.columns):
        raise ValueError(f"Ground truth CSV missing columns: {sorted(missing)}")
    if missing := required_pred - set(pred.columns):
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

    merged = gt.merge(pred, on="image", how="inner", suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("No overlapping image names between ground truth and predictions")

    merged = merged.rename(columns={"count": "ground_truth", prediction_column: "prediction"})
    merged["error"] = merged["prediction"] - merged["ground_truth"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2
    merged["relative_error"] = np.where(
        merged["ground_truth"] > 0,
        merged["abs_error"] / merged["ground_truth"],
        np.nan,
    )

    ensure_dir(output_dir)
    merged_path = output_dir / "per_image_errors.csv"
    metrics_path = output_dir / "count_metrics.csv"
    plot_path = output_dir / "prediction_vs_ground_truth.png"

    metrics = pd.DataFrame([{
        "num_images": len(merged),
        "mae": float(merged["abs_error"].mean()),
        "rmse": float(np.sqrt(merged["squared_error"].mean())),
        "mean_relative_error": float(merged["relative_error"].mean(skipna=True)),
        "bias": float(merged["error"].mean()),
    }])

    merged.sort_values("abs_error", ascending=False).to_csv(merged_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    _save_prediction_plot(merged, plot_path)

    print(metrics.to_string(index=False))
    print(f"Wrote per-image errors to {merged_path}")
    print(f"Wrote count metrics to {metrics_path}")
    print(f"Wrote scatter plot to {plot_path}")
    return metrics


def _save_prediction_plot(frame: pd.DataFrame, output_path: Path) -> None:
    max_value = max(float(frame["ground_truth"].max()), float(frame["prediction"].max()), 1.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["ground_truth"], frame["prediction"], alpha=0.75)
    ax.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Ground truth count")
    ax.set_ylabel("Predicted count")
    ax.set_title("Puffin count prediction vs ground truth")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

