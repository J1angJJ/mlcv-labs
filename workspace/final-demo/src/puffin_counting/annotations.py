from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, list_images


def split_image_dataset(
    image_dir: Path,
    output_csv: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    images = list_images(image_dir)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    counts = {name: int(n_total * ratio) for name, ratio in ratios.items()}
    remainder = n_total - sum(counts.values())

    fractional_order = sorted(
        ratios,
        key=lambda name: (n_total * ratios[name]) - int(n_total * ratios[name]),
        reverse=True,
    )
    for name in fractional_order[:remainder]:
        counts[name] += 1

    positive_splits = [name for name, ratio in ratios.items() if ratio > 0]
    if n_total >= len(positive_splits):
        for name in positive_splits:
            if counts[name] == 0:
                donor = max(counts, key=counts.get)
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] += 1

    rows = []
    for index, path in enumerate(shuffled):
        if index < counts["train"]:
            split = "train"
        elif index < counts["train"] + counts["val"]:
            split = "val"
        else:
            split = "test"
        rows.append({"image": str(path.relative_to(image_dir)), "split": split})

    frame = pd.DataFrame(rows).sort_values(["split", "image"])
    ensure_dir(output_csv.parent)
    frame.to_csv(output_csv, index=False)
    print(f"Wrote {len(frame)} split rows to {output_csv}")
    print(frame["split"].value_counts().to_string())
    return frame
