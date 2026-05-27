from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "final_demo_failure_cases.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_image_retrieval"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR / "hf_home"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODEL_CACHE_DIR / "torch_home"))

import open_clip
import torch
from PIL import Image


@dataclass(frozen=True)
class ImageItem:
    image_id: str
    image_path: Path
    group: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP image embeddings and compute nearest-neighbor retrieval.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV with image_id, group, source, image_path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained weights tag.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--top-k", type=int, default=3, help="Nearest neighbors per image.")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_manifest(path: Path) -> list[ImageItem]:
    manifest = path.resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    items: list[ImageItem] = []
    with manifest.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "image_path"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        for row in reader:
            image_path = Path(row["image_path"].strip())
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image does not exist: {image_path}")
            items.append(
                ImageItem(
                    image_id=row["image_id"].strip() or image_path.name,
                    image_path=image_path,
                    group=row.get("group", "").strip(),
                    source=row.get("source", "").strip(),
                )
            )
    if len(items) < 2:
        raise ValueError("At least two images are required for retrieval.")
    return items


def encode_image(model, preprocess, image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).detach().cpu()


def write_metadata(items: list[ImageItem], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "image_id", "group", "source", "image_path"])
        writer.writeheader()
        for index, item in enumerate(items):
            writer.writerow(
                {
                    "index": index,
                    "image_id": item.image_id,
                    "group": item.group,
                    "source": item.source,
                    "image_path": str(item.image_path),
                }
            )


def write_similarity_matrix(items: list[ImageItem], similarities: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["image_id", *[item.image_id for item in items]]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item, row in zip(items, similarities):
            values = {"image_id": item.image_id}
            values.update({other.image_id: f"{score:.8f}" for other, score in zip(items, row)})
            writer.writerow(values)


def write_nearest_neighbors(items: list[ImageItem], similarities: np.ndarray, top_k: int, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "query_id",
            "query_group",
            "query_source",
            "neighbor_rank",
            "neighbor_id",
            "neighbor_group",
            "neighbor_source",
            "similarity",
            "same_group",
            "same_source",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_index, query in enumerate(items):
            order = np.argsort(-similarities[query_index])
            neighbors = [idx for idx in order if idx != query_index][:top_k]
            for rank, neighbor_index in enumerate(neighbors, start=1):
                neighbor = items[neighbor_index]
                writer.writerow(
                    {
                        "query_id": query.image_id,
                        "query_group": query.group,
                        "query_source": query.source,
                        "neighbor_rank": rank,
                        "neighbor_id": neighbor.image_id,
                        "neighbor_group": neighbor.group,
                        "neighbor_source": neighbor.source,
                        "similarity": f"{similarities[query_index, neighbor_index]:.8f}",
                        "same_group": str(query.group == neighbor.group),
                        "same_source": str(query.source == neighbor.source),
                    }
                )


def main() -> None:
    args = parse_args()
    items = load_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()

    embeddings = torch.stack([encode_image(model, preprocess, item.image_path, device) for item in items])
    similarities = (embeddings @ embeddings.T).numpy()

    np.save(output_dir / "embeddings.npy", embeddings.numpy())
    np.save(output_dir / "similarity_matrix.npy", similarities)
    write_metadata(items, output_dir / "metadata.csv")
    write_similarity_matrix(items, similarities, output_dir / "similarity_matrix.csv")
    write_nearest_neighbors(items, similarities, args.top_k, output_dir / "nearest_neighbors.csv")

    print(f"Processed {len(items)} images.")
    print(f"Embeddings: {output_dir / 'embeddings.npy'}")
    print(f"Nearest neighbors: {output_dir / 'nearest_neighbors.csv'}")
    print(f"Similarity matrix: {output_dir / 'similarity_matrix.csv'}")


if __name__ == "__main__":
    main()
