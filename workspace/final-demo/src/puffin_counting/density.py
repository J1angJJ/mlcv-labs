from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .io_utils import ensure_dir


def points_to_density_map(
    image_shape: tuple[int, int],
    points: np.ndarray,
    sigma: float = 15.0,
) -> np.ndarray:
    height, width = image_shape
    density = np.zeros((height, width), dtype=np.float32)

    for x, y in points:
        ix, iy = int(round(float(x))), int(round(float(y)))
        if 0 <= ix < width and 0 <= iy < height:
            density[iy, ix] += 1.0

    if sigma > 0:
        density = gaussian_filter(density, sigma=sigma, mode="constant")

    # Preserve the annotation count after filtering near image borders.
    original_count = float(len(points))
    current_sum = float(density.sum())
    if current_sum > 0:
        density *= original_count / current_sum
    return density


def save_density_visualization(
    image_rgb: np.ndarray,
    points: np.ndarray,
    density: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)
    estimated_count = float(density.sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    if len(points):
        axes[0].scatter(points[:, 0], points[:, 1], c="red", s=16)
    axes[0].set_title(f"Point annotations: {len(points)}")
    axes[0].axis("off")

    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    axes[1].imshow(image_gray, cmap="gray")
    axes[1].imshow(density, cmap="jet", alpha=0.55)
    axes[1].set_title("Density overlay")
    axes[1].axis("off")

    im = axes[2].imshow(density, cmap="jet")
    axes[2].set_title(f"Integrated count: {estimated_count:.2f}")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_density_from_points(
    points_csv: Path,
    image_dir: Path,
    output_dir: Path,
    sigma: float = 15.0,
) -> pd.DataFrame:
    points = pd.read_csv(points_csv)
    required = {"image", "x", "y"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"Point CSV is missing columns: {sorted(missing)}")

    ensure_dir(output_dir)
    rows = []

    for image_name, group in points.groupby("image"):
        image_path = image_dir / image_name
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        point_array = group[["x", "y"]].to_numpy(dtype=np.float32)
        density = points_to_density_map(image_rgb.shape[:2], point_array, sigma=sigma)

        stem = Path(image_name).stem
        density_path = output_dir / f"{stem}_density.npy"
        figure_path = output_dir / f"{stem}_density.png"
        np.save(density_path, density)
        save_density_visualization(
            image_rgb=image_rgb,
            points=point_array,
            density=density,
            output_path=figure_path,
            title=f"{image_name} | sigma={sigma}",
        )

        rows.append({
            "image": image_name,
            "point_count": int(len(point_array)),
            "density_count": float(density.sum()),
            "density_map": str(density_path),
            "figure": str(figure_path),
        })

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "density_counts.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote density summary to {summary_path}")
    return summary


def run_density_demo(
    output_dir: Path,
    num_points: int = 25,
    sigma: float = 15.0,
    seed: int = 42,
) -> None:
    ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    height, width = 320, 480
    image_rgb = np.full((height, width, 3), 225, dtype=np.uint8)

    xs = rng.integers(30, width - 30, size=num_points)
    ys = rng.integers(30, height - 30, size=num_points)
    points = np.column_stack([xs, ys]).astype(np.float32)

    for x, y in points:
        cv2.circle(image_rgb, (int(x), int(y)), 5, (40, 40, 40), -1)

    density = points_to_density_map((height, width), points, sigma=sigma)
    save_density_visualization(
        image_rgb=image_rgb,
        points=points,
        density=density,
        output_path=output_dir / "synthetic_density_demo.png",
        title=f"Synthetic puffin points | GT={num_points}, estimated={density.sum():.2f}",
    )

    pd.DataFrame(points, columns=["x", "y"]).assign(image="synthetic.png").to_csv(
        output_dir / "synthetic_points.csv", index=False
    )
    pd.DataFrame([{
        "image": "synthetic.png",
        "point_count": num_points,
        "density_count": float(density.sum()),
        "sigma": sigma,
        "seed": seed,
    }]).to_csv(output_dir / "synthetic_density_summary.csv", index=False)
    print(f"Synthetic density demo written to {output_dir}")
