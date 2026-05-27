from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "clip_image_retrieval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_embedding_plot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project CLIP image embeddings to 2D and draw a scatter plot.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with embeddings.npy and metadata.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plot outputs.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="2D projection method.")
    parser.add_argument("--perplexity", type=float, default=3.0, help="t-SNE perplexity. Only used when --method tsne.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def project_embeddings(embeddings: np.ndarray, method: str, perplexity: float, seed: int) -> tuple[np.ndarray, str]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError("At least two embeddings are required for visualization.")

    if method == "pca":
        projector = PCA(n_components=2)
        points = projector.fit_transform(embeddings)
        explained = projector.explained_variance_ratio_
        title = f"PCA of CLIP image embeddings ({explained[0] * 100:.1f}% + {explained[1] * 100:.1f}% variance)"
        return points, title

    max_perplexity = max(1.0, embeddings.shape[0] - 1.0)
    safe_perplexity = min(perplexity, max_perplexity)
    projector = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return projector.fit_transform(embeddings), f"t-SNE of CLIP image embeddings (perplexity={safe_perplexity:g})"


def write_points(metadata: list[dict[str, str]], points: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["image_id", "group", "source", "x", "y", "image_path"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, point in zip(metadata, points):
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "group": row.get("group", ""),
                    "source": row.get("source", ""),
                    "x": f"{point[0]:.8f}",
                    "y": f"{point[1]:.8f}",
                    "image_path": row.get("image_path", ""),
                }
            )


def draw_plot(metadata: list[dict[str, str]], points: np.ndarray, title: str, output_path: Path) -> None:
    source_styles = {
        "original": {"marker": "o", "label": "original"},
        "yolo11n_pred": {"marker": "s", "label": "YOLO11n prediction"},
    }
    group_colors = {
        "case_001_both_under_count": "#1f77b4",
        "case_002_under_to_over": "#17becf",
        "case_003_both_under_count": "#2ca02c",
        "case_004_exp002_new_error": "#ff7f0e",
        "case_005_exp002_fixed_exp001_error": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9.0, 6.5), dpi=160)
    for index, row in enumerate(metadata):
        source = row.get("source", "")
        group = row.get("group", "")
        style = source_styles.get(source, {"marker": "x", "label": source or "unknown"})
        ax.scatter(
            points[index, 0],
            points[index, 1],
            s=86,
            marker=style["marker"],
            color=group_colors.get(group, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
        )
        label = row["image_id"].replace("case_", "c").replace("_original", "_orig").replace("_yolo11n", "_pred")
        ax.annotate(label, (points[index, 0], points[index, 1]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    for group in sorted({row.get("group", "") for row in metadata}):
        group_indices = [idx for idx, row in enumerate(metadata) if row.get("group", "") == group]
        if len(group_indices) == 2:
            first, second = group_indices
            ax.plot(
                [points[first, 0], points[second, 0]],
                [points[first, 1], points[second, 1]],
                color=group_colors.get(group, "#7f7f7f"),
                linewidth=1.1,
                linestyle="--",
                alpha=0.55,
            )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markeredgecolor="black", label="original", markersize=8),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#555555",
            markeredgecolor="black",
            label="YOLO11n prediction",
            markersize=8,
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = input_dir / "embeddings.npy"
    metadata_path = input_dir / "metadata.csv"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings do not exist: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    metadata = load_metadata(metadata_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match embeddings ({embeddings.shape[0]}).")

    points, title = project_embeddings(embeddings, args.method, args.perplexity, args.seed)
    points_csv = output_dir / f"{args.method}_points.csv"
    plot_png = output_dir / f"{args.method}_plot.png"
    write_points(metadata, points, points_csv)
    draw_plot(metadata, points, title, plot_png)

    print(f"Projected {embeddings.shape[0]} embeddings with {args.method}.")
    print(f"Points: {points_csv}")
    print(f"Plot: {plot_png}")


if __name__ == "__main__":
    main()
