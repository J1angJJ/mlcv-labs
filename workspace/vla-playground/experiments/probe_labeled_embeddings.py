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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_labeled_300_probe"
DEFAULT_TASKS = ["distance", "difficulty", "contains_puffin", "density", "occlusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probe classifiers on labeled CLIP embeddings.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with embeddings.npy and metadata.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Metadata fields to probe.")
    parser.add_argument("--knn-k", type=int, default=5, help="K for KNN.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def valid_rows_for_task(rows: list[dict[str, str]], task: str) -> tuple[list[int], list[str]]:
    indices = []
    labels = []
    for index, row in enumerate(rows):
        label = row.get(task, "").strip()
        if not label or label == "unknown" or label == "uncertain":
            continue
        indices.append(index)
        labels.append(label)
    return indices, labels


def can_stratify(labels: list[str]) -> bool:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return len(counts) >= 2 and min(counts.values()) >= 2


def evaluate_task(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    task: str,
    knn_k: int,
    test_size: float,
    seed: int,
    output_dir: Path,
) -> list[dict[str, str]]:
    indices, labels = valid_rows_for_task(rows, task)
    if not can_stratify(labels):
        return [
            {
                "task": task,
                "method": "skipped",
                "accuracy": "",
                "num_samples": str(len(labels)),
                "classes": "|".join(sorted(set(labels))),
                "confusion_matrix": "not enough labeled samples for stratified evaluation",
            }
        ]

    x = embeddings[indices]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(x, y))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    metadata_test = [rows[indices[i]] for i in test_index]

    methods = {
        "knn": KNeighborsClassifier(n_neighbors=min(knn_k, len(train_index)), metric="cosine"),
        "linear": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    }

    summary: list[dict[str, str]] = []
    prediction_rows: list[dict[str, str]] = []
    for method_name, model in methods.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, pred)
        matrix = confusion_matrix(y_test, pred, labels=list(range(len(encoder.classes_))))
        true_labels = encoder.inverse_transform(y_test)
        pred_labels = encoder.inverse_transform(pred)
        summary.append(
            {
                "task": task,
                "method": method_name,
                "accuracy": f"{accuracy:.4f}",
                "num_samples": str(len(labels)),
                "classes": "|".join(encoder.classes_),
                "confusion_matrix": np.array2string(matrix, separator=","),
            }
        )
        for row, true_label, pred_label in zip(metadata_test, true_labels, pred_labels):
            prediction_rows.append(
                {
                    "task": task,
                    "method": method_name,
                    "image_id": row.get("image_id", ""),
                    "split": row.get("split", ""),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": str(true_label == pred_label),
                    "image_path": row.get("image_path", ""),
                }
            )

    with (output_dir / f"{task}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["task", "method", "image_id", "split", "true_label", "pred_label", "correct", "image_path"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    return summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(input_dir / "embeddings.npy")
    rows = load_metadata(input_dir / "metadata.csv")
    if len(rows) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(rows)}) do not match embeddings ({embeddings.shape[0]}).")

    all_summary: list[dict[str, str]] = []
    for task in args.tasks:
        all_summary.extend(evaluate_task(embeddings, rows, task, args.knn_k, args.test_size, args.seed, output_dir))

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["task", "method", "accuracy", "num_samples", "classes", "confusion_matrix"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary)

    print(f"Evaluated tasks: {', '.join(args.tasks)}")
    print(f"Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
