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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_plots"
DEFAULT_COLOR_FIELDS = ["distance", "contains_puffin", "density", "difficulty", "occlusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize labeled CLIP embeddings with 2D projections.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with embeddings.npy and metadata.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Projection method.")
    parser.add_argument("--color-fields", nargs="+", default=DEFAULT_COLOR_FIELDS, help="Metadata fields used for coloring plots.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def project(embeddings: np.ndarray, method: str, perplexity: float, seed: int) -> tuple[np.ndarray, str]:
    if method == "pca":
        pca = PCA(n_components=2)
        points = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_
        title = f"PCA ({explained[0] * 100:.1f}% + {explained[1] * 100:.1f}% variance)"
        return points, title

    safe_perplexity = min(perplexity, max(1.0, embeddings.shape[0] - 1.0))
    tsne = TSNE(n_components=2, perplexity=safe_perplexity, init="pca", learning_rate="auto", random_state=seed)
    points = tsne.fit_transform(embeddings)
    return points, f"t-SNE (perplexity={safe_perplexity:g})"


def write_points(rows: list[dict[str, str]], points: np.ndarray, output_path: Path) -> None:
    fieldnames = ["image_id", "x", "y", "split", *DEFAULT_COLOR_FIELDS, "image_path"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row, point in zip(rows, points):
            values = dict(row)
            values["x"] = f"{point[0]:.8f}"
            values["y"] = f"{point[1]:.8f}"
            writer.writerow(values)


def color_map_for(values: list[str]) -> dict[str, str]:
    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#f97316",
        "#0891b2",
        "#4b5563",
        "#ca8a04",
    ]
    ordered = sorted({value or "blank" for value in values})
    return {value: palette[index % len(palette)] for index, value in enumerate(ordered)}


def draw_plot(rows: list[dict[str, str]], points: np.ndarray, field: str, projection_title: str, output_path: Path) -> None:
    values = [row.get(field, "") or "blank" for row in rows]
    colors = color_map_for(values)
    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=160)

    for value in sorted(colors):
        indices = [index for index, item in enumerate(values) if item == value]
        ax.scatter(
            points[indices, 0],
            points[indices, 1],
            s=34,
            alpha=0.78,
            c=colors[value],
            label=f"{value} ({len(indices)})",
            edgecolors="white",
            linewidths=0.35,
        )

    ax.set_title(f"{projection_title}, colored by {field}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(input_dir / "embeddings.npy")
    rows = load_metadata(input_dir / "metadata.csv")
    if len(rows) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(rows)}) do not match embeddings ({embeddings.shape[0]}).")

    points, projection_title = project(embeddings, args.method, args.perplexity, args.seed)
    write_points(rows, points, output_dir / f"{args.method}_points.csv")
    for field in args.color_fields:
        if field not in rows[0]:
            print(f"Skip missing field: {field}")
            continue
        draw_plot(rows, points, field, projection_title, output_dir / f"{args.method}_by_{field}.png")

    print(f"Projected {len(rows)} embeddings with {args.method}.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
