from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_IMAGE_DIR = REPO_ROOT / "workspace" / "final-demo" / "report_assets" / "browser_screenshots"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_zero_shot"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models"

# Keep model downloads off the system drive by default. These paths are ignored by Git.
os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR / "hf_home"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODEL_CACHE_DIR / "torch_home"))

import open_clip
import torch
from PIL import Image


DEFAULT_PROMPTS = [
    "a photo of a puffin",
    "a photo of seabirds on a cliff",
    "a drone photo of seabirds",
    "a close-up photo of a puffin",
    "a photo of rocks",
]


@dataclass(frozen=True)
class ImageItem:
    image_id: str
    image_path: Path
    group: str = ""
    source: str = ""


@dataclass(frozen=True)
class ImageScore:
    image_id: str
    image_path: str
    group: str
    source: str
    prompt: str
    score: float
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP zero-shot image-text similarity on a local image folder.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV with at least image_path. Optional columns: image_id, group, source.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV outputs.")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained weights tag.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Text prompt. Can be passed multiple times. Defaults to the built-in prompt set.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 text file containing one prompt per line. Blank lines and lines starting with # are ignored.",
    )
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        prompt_file = args.prompt_file.resolve()
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")
        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                prompts.append(value)
    if args.prompts:
        prompts.extend(args.prompts)
    return prompts or DEFAULT_PROMPTS


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def list_images(image_dir: Path, extensions: list[str]) -> list[Path]:
    suffixes = {item.lower() for item in extensions}
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def load_images(args: argparse.Namespace) -> list[ImageItem]:
    if args.manifest:
        manifest = args.manifest.resolve()
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest}")
        items: list[ImageItem] = []
        with manifest.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "image_path" not in reader.fieldnames:
                raise ValueError("Manifest must contain an image_path column.")
            for row in reader:
                raw_path = row["image_path"].strip()
                if not raw_path:
                    continue
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = (manifest.parent / image_path).resolve()
                if not image_path.exists():
                    raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
                image_id = row.get("image_id", "").strip() or image_path.name
                items.append(
                    ImageItem(
                        image_id=image_id,
                        image_path=image_path,
                        group=row.get("group", "").strip(),
                        source=row.get("source", "").strip(),
                    )
                )
        if not items:
            raise FileNotFoundError(f"No image rows found in {manifest}")
        return items

    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = list_images(image_dir, args.extensions)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return [ImageItem(image_id=path.name, image_path=path, group=image_dir.name, source="image_dir") for path in images]


def encode_text(model, tokenizer, prompts: list[str], device: torch.device) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_image(model, preprocess, image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def write_scores(scores: list[ImageScore], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "image_path", "group", "source", "prompt", "score", "rank"])
        writer.writeheader()
        for item in scores:
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "image_path": item.image_path,
                    "group": item.group,
                    "source": item.source,
                    "prompt": item.prompt,
                    "score": f"{item.score:.8f}",
                    "rank": item.rank,
                }
            )


def write_top_matches(scores: list[ImageScore], output_path: Path) -> None:
    top = [item for item in scores if item.rank == 1]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "image_path", "group", "source", "top_prompt", "score"])
        writer.writeheader()
        for item in top:
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "image_path": item.image_path,
                    "group": item.group,
                    "source": item.source,
                    "top_prompt": item.prompt,
                    "score": f"{item.score:.8f}",
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    prompts = load_prompts(args)

    images = load_images(args)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    text_features = encode_text(model, tokenizer, prompts, device)

    all_scores: list[ImageScore] = []
    for item in images:
        image_features = encode_image(model, preprocess, item.image_path, device)
        similarities = (image_features @ text_features.T).squeeze(0).detach().cpu()
        ranked_indices = torch.argsort(similarities, descending=True).tolist()
        for rank, prompt_index in enumerate(ranked_indices, start=1):
            all_scores.append(
                ImageScore(
                    image_id=item.image_id,
                    image_path=str(item.image_path),
                    group=item.group,
                    source=item.source,
                    prompt=prompts[prompt_index],
                    score=float(similarities[prompt_index]),
                    rank=rank,
                )
            )

    write_scores(all_scores, output_dir / "scores.csv")
    write_top_matches(all_scores, output_dir / "top_matches.csv")

    print(f"Processed {len(images)} images with {len(prompts)} prompts.")
    print(f"Scores: {output_dir / 'scores.csv'}")
    print(f"Top matches: {output_dir / 'top_matches.csv'}")


if __name__ == "__main__":
    main()
