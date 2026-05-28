from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings"
DEFAULT_COLOR_MANIFEST = PROJECT_ROOT / "experiments" / "manifests" / "seabirds_annotation_300_labeled_color.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_probe_by_color"
DEFAULT_TASKS = ["distance", "difficulty", "contains_puffin", "density", "occlusion"]
DEFAULT_SUBSETS = ["all", "color", "grayscale", "low_color"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP embedding probes separately by automatic color mode.")
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR, help="Directory with embeddings.npy and metadata.csv.")
    parser.add_argument("--color-manifest", type=Path, default=DEFAULT_COLOR_MANIFEST, help="Manifest with color_mode_auto.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Metadata fields to probe.")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS, help="Color subsets to evaluate.")
    parser.add_argument("--knn-k", type=int, default=5, help="K for KNN.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def merge_metadata(metadata_rows: list[dict[str, str]], color_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    color_by_id = {row["image_id"]: row for row in color_rows}
    merged = []
    for row in metadata_rows:
        image_id = row["image_id"]
        item = dict(row)
        if image_id in color_by_id:
            item.update(
                {
                    "color_mode_auto": color_by_id[image_id].get("color_mode_auto", ""),
                    "color_channel_delta": color_by_id[image_id].get("color_channel_delta", ""),
                    "color_saturation": color_by_id[image_id].get("color_saturation", ""),
                }
            )
        else:
            item["color_mode_auto"] = "missing"
        merged.append(item)
    return merged


def row_indices_for_subset(rows: list[dict[str, str]], subset: str) -> list[int]:
    if subset == "all":
        return list(range(len(rows)))
    return [index for index, row in enumerate(rows) if row.get("color_mode_auto") == subset]


def task_indices_and_labels(rows: list[dict[str, str]], base_indices: list[int], task: str) -> tuple[list[int], list[str]]:
    indices = []
    labels = []
    for index in base_indices:
        label = rows[index].get(task, "").strip()
        if not label or label in {"unknown", "uncertain"}:
            continue
        indices.append(index)
        labels.append(label)
    return indices, labels


def class_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def can_evaluate(labels: list[str]) -> bool:
    counts = class_counts(labels)
    return len(counts) >= 2 and min(counts.values()) >= 2 and len(labels) >= 8


def evaluate(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    subset: str,
    task: str,
    knn_k: int,
    test_size: float,
    seed: int,
    output_dir: Path,
) -> list[dict[str, str]]:
    subset_indices = row_indices_for_subset(rows, subset)
    indices, labels = task_indices_and_labels(rows, subset_indices, task)
    counts = class_counts(labels)
    if not can_evaluate(labels):
        return [
            {
                "subset": subset,
                "task": task,
                "method": "skipped",
                "accuracy": "",
                "num_samples": str(len(labels)),
                "class_counts": "|".join(f"{key}:{value}" for key, value in sorted(counts.items())),
                "confusion_matrix": "not enough samples/classes",
            }
        ]

    x = embeddings[indices]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(x, y))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    test_rows = [rows[indices[index]] for index in test_index]

    methods = {
        "knn": KNeighborsClassifier(n_neighbors=min(knn_k, len(train_index)), metric="cosine"),
        "linear": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    }

    summaries: list[dict[str, str]] = []
    prediction_rows: list[dict[str, str]] = []
    for method_name, model in methods.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, pred)
        matrix = confusion_matrix(y_test, pred, labels=list(range(len(encoder.classes_))))
        true_labels = encoder.inverse_transform(y_test)
        pred_labels = encoder.inverse_transform(pred)
        summaries.append(
            {
                "subset": subset,
                "task": task,
                "method": method_name,
                "accuracy": f"{accuracy:.4f}",
                "num_samples": str(len(labels)),
                "class_counts": "|".join(f"{key}:{value}" for key, value in sorted(counts.items())),
                "confusion_matrix": np.array2string(matrix, separator=","),
            }
        )
        for row, true_label, pred_label in zip(test_rows, true_labels, pred_labels):
            prediction_rows.append(
                {
                    "subset": subset,
                    "task": task,
                    "method": method_name,
                    "image_id": row.get("image_id", ""),
                    "color_mode_auto": row.get("color_mode_auto", ""),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": str(true_label == pred_label),
                    "image_path": row.get("image_path", ""),
                }
            )

    if prediction_rows:
        prediction_path = output_dir / f"{subset}_{task}_predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = ["subset", "task", "method", "image_id", "color_mode_auto", "true_label", "pred_label", "correct", "image_path"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)
    return summaries


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["subset", "task", "method", "accuracy", "num_samples", "class_counts", "confusion_matrix"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    embedding_dir = args.embedding_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embedding_dir / "embeddings.npy")
    metadata_rows = load_csv(embedding_dir / "metadata.csv")
    color_rows = load_csv(args.color_manifest.resolve())
    if len(metadata_rows) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata_rows)}) do not match embeddings ({embeddings.shape[0]}).")

    rows = merge_metadata(metadata_rows, color_rows)
    summaries: list[dict[str, str]] = []
    for subset in args.subsets:
        for task in args.tasks:
            summaries.extend(evaluate(embeddings, rows, subset, task, args.knn_k, args.test_size, args.seed, output_dir))

    write_summary(summaries, output_dir / "summary_by_color.csv")
    print(f"Evaluated subsets: {', '.join(args.subsets)}")
    print(f"Evaluated tasks: {', '.join(args.tasks)}")
    print(f"Summary: {output_dir / 'summary_by_color.csv'}")


if __name__ == "__main__":
    main()
