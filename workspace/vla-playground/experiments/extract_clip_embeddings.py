from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "seabirds_annotation_300_labeled.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR / "hf_home"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODEL_CACHE_DIR / "torch_home"))

import open_clip
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP image embeddings for a labeled manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV manifest with image_path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained weights tag.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device.")
    parser.add_argument("--batch-size", type=int, default=32, help="Image batch size.")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_manifest(path: Path) -> list[dict[str, str]]:
    manifest = path.resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    if "image_path" not in rows[0]:
        raise ValueError("Manifest must contain image_path.")
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
    return rows


def batched(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def encode_batch(model, preprocess, rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    tensors = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        tensors.append(preprocess(image))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu()


def write_metadata(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()

    parts = []
    for batch_index, batch_rows in enumerate(batched(rows, args.batch_size), start=1):
        parts.append(encode_batch(model, preprocess, batch_rows, device))
        print(f"Encoded batch {batch_index} ({min(batch_index * args.batch_size, len(rows))}/{len(rows)})")

    embeddings = torch.cat(parts, dim=0).numpy()
    np.save(output_dir / "embeddings.npy", embeddings)
    write_metadata(rows, output_dir / "metadata.csv")

    with (output_dir / "run_info.txt").open("w", encoding="utf-8") as file:
        file.write(f"manifest={args.manifest.resolve()}\n")
        file.write(f"model={args.model}\n")
        file.write(f"pretrained={args.pretrained}\n")
        file.write(f"device={device}\n")
        file.write(f"batch_size={args.batch_size}\n")
        file.write(f"num_images={len(rows)}\n")
        file.write(f"embedding_dim={embeddings.shape[1]}\n")

    print(f"Processed {len(rows)} images.")
    print(f"Embeddings: {output_dir / 'embeddings.npy'}")
    print(f"Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
