from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_retrieval"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR / "hf_home"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODEL_CACHE_DIR / "torch_home"))

import open_clip
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve labeled Seabirds images with CLIP text/image queries.")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--text", help="Text query, e.g. 'a black and white image of seabirds'.")
    query.add_argument("--query-image", type=Path, help="External query image path.")
    query.add_argument("--query-id", help="Use an image_id already present in the indexed metadata.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Directory with embeddings.npy and metadata.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of retrieval results.")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata filter in key=value form. Can be repeated, e.g. --filter scene=grass --filter distance=far.",
    )
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device.")
    parser.add_argument("--name", help="Output stem. Defaults to a sanitized query name.")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def safe_stem(text: str) -> str:
    allowed = []
    for char in text.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "_", "-", "."}:
            allowed.append("_")
    stem = "".join(allowed).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem[:80] or "query"


def load_index(index_dir: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    embeddings = np.load(index_dir / "embeddings.npy")
    metadata = load_metadata(index_dir / "metadata.csv")
    if embeddings.shape[0] != len(metadata):
        raise ValueError(f"Embeddings ({embeddings.shape[0]}) and metadata rows ({len(metadata)}) do not match.")
    return embeddings, metadata


def parse_filters(filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in filters:
        if "=" not in item:
            raise ValueError(f"Filter must use key=value form: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Filter must use non-empty key=value form: {item}")
        parsed[key] = value
    return parsed


def row_matches_filters(row: dict[str, str], filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        if row.get(key, "") != expected:
            return False
    return True


def load_model(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def encode_text(model, tokenizer, text: str, device: torch.device) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).detach().cpu().numpy()


def encode_image(model, preprocess, image_path: Path, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).detach().cpu().numpy()


def vector_from_query_id(embeddings: np.ndarray, metadata: list[dict[str, str]], image_id: str) -> np.ndarray:
    for index, row in enumerate(metadata):
        if row.get("image_id") == image_id:
            return embeddings[index]
    raise ValueError(f"query-id not found in metadata: {image_id}")


def retrieve(
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    metadata: list[dict[str, str]],
    top_k: int,
    skip_id: str = "",
    filters: dict[str, str] | None = None,
):
    filters = filters or {}
    similarities = embeddings @ query_vector
    order = np.argsort(-similarities)
    results = []
    for index in order:
        row = metadata[index]
        if skip_id and row.get("image_id") == skip_id:
            continue
        if not row_matches_filters(row, filters):
            continue
        result = dict(row)
        result["rank"] = str(len(results) + 1)
        result["similarity"] = f"{float(similarities[index]):.8f}"
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def write_results(results: list[dict[str, str]], output_path: Path) -> None:
    fieldnames: list[str] = ["rank", "similarity"]
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def file_uri(path_text: str) -> str:
    return Path(path_text).resolve().as_uri()


def write_gallery(results: list[dict[str, str]], title: str, output_path: Path) -> None:
    cards = []
    for row in results:
        image_path = row.get("image_path", "")
        image_uri = file_uri(image_path)
        cards.append(
            f"""
            <article>
              <img src="{html.escape(image_uri)}" alt="{html.escape(row.get('image_id', 'image'))}">
              <div class="body">
                <div class="rank">#{html.escape(row.get('rank', ''))} score={html.escape(row.get('similarity', ''))}</div>
                <div><strong>{html.escape(row.get('image_id', ''))}</strong></div>
                <div>distance={html.escape(row.get('distance', ''))} | density={html.escape(row.get('density', ''))}</div>
                <div>puffin={html.escape(row.get('contains_puffin', ''))} | difficulty={html.escape(row.get('difficulty', ''))}</div>
                <div>occlusion={html.escape(row.get('occlusion', ''))}</div>
                <div class="path">{html.escape(image_path)}</div>
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #f5f7fb; color: #1f2933; }}
    header {{ background: #111827; color: white; padding: 16px 18px; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 18px; }}
    main {{ padding: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
    article {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; overflow: hidden; }}
    img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #e5e7eb; display: block; }}
    .body {{ padding: 10px 12px 12px; font-size: 12px; line-height: 1.45; }}
    .rank {{ color: #1d4ed8; font-weight: 700; margin-bottom: 4px; }}
    .path {{ margin-top: 6px; color: #64748b; word-break: break-all; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1></header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    index_dir = args.index_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings, metadata = load_index(index_dir)
    filters = parse_filters(args.filter)

    skip_id = ""
    if args.query_id:
        query_vector = vector_from_query_id(embeddings, metadata, args.query_id)
        title = f"image-id query: {args.query_id}"
        output_stem = args.name or safe_stem(f"image_id_{args.query_id}")
        skip_id = args.query_id
    else:
        device = resolve_device(args.device)
        model, preprocess, tokenizer = load_model(args.model, args.pretrained, device)
        if args.text:
            query_vector = encode_text(model, tokenizer, args.text, device)
            title = f"text query: {args.text}"
            output_stem = args.name or safe_stem(f"text_{args.text}")
        else:
            query_path = args.query_image.resolve()
            query_vector = encode_image(model, preprocess, query_path, device)
            title = f"image query: {query_path.name}"
            output_stem = args.name or safe_stem(f"image_{query_path.stem}")

    results = retrieve(query_vector, embeddings, metadata, args.top_k, skip_id=skip_id, filters=filters)
    csv_path = output_dir / f"{output_stem}.csv"
    html_path = output_dir / f"{output_stem}.html"
    write_results(results, csv_path)
    write_gallery(results, title, html_path)

    print(f"Query: {title}")
    if filters:
        print(f"Filters: {filters}")
    print(f"Results: {csv_path}")
    print(f"Gallery: {html_path}")
    if results:
        print(f"Top-1: {results[0].get('image_id')} ({results[0].get('similarity')})")


if __name__ == "__main__":
    main()
