from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings"
DEFAULT_COLOR_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "seabirds_annotation_300_labeled_color.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings_with_color"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy embeddings and replace metadata with color-enriched manifest.")
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--color-manifest", type=Path, default=DEFAULT_COLOR_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    embedding_dir = args.embedding_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embedding_dir / "embeddings.npy")
    rows = load_csv(args.color_manifest.resolve())
    if len(rows) != embeddings.shape[0]:
        raise ValueError(f"Color manifest rows ({len(rows)}) do not match embeddings ({embeddings.shape[0]}).")

    np.save(output_dir / "embeddings.npy", embeddings)
    write_csv(rows, output_dir / "metadata.csv")
    print(f"Rows: {len(rows)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
