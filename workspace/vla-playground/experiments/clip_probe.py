from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "clip_image_retrieval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small probes on frozen CLIP image embeddings.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with embeddings.npy and metadata.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for probe outputs.")
    parser.add_argument("--knn-k", type=int, default=1, help="K for KNN probe.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def scene_label(row: dict[str, str]) -> str:
    group = row.get("group", "")
    if group.startswith("case_001") or group.startswith("case_002"):
        return "far_scene"
    return "close_scene"


def source_label(row: dict[str, str]) -> str:
    return row.get("source", "") or "unknown"


def leave_one_out_predictions(
    embeddings: np.ndarray,
    labels: list[str],
    method: str,
    knn_k: int,
) -> tuple[list[str], float, list[str], np.ndarray]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = list(encoder.classes_)

    predictions: list[int] = []
    truth: list[int] = []
    splitter = LeaveOneOut()
    for train_index, test_index in splitter.split(embeddings):
        x_train, x_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if method == "knn":
            model = KNeighborsClassifier(n_neighbors=min(knn_k, len(train_index)), metric="cosine")
        elif method == "linear":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        model.fit(x_train, y_train)
        predictions.append(int(model.predict(x_test)[0]))
        truth.append(int(y_test[0]))

    accuracy = accuracy_score(truth, predictions)
    matrix = confusion_matrix(truth, predictions, labels=list(range(len(classes))))
    decoded_predictions = list(encoder.inverse_transform(predictions))
    return decoded_predictions, float(accuracy), classes, matrix


def write_predictions(
    metadata: list[dict[str, str]],
    task_name: str,
    true_labels: list[str],
    method_predictions: dict[str, list[str]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["image_id", "group", "source", "task", "true_label", *[f"{method}_prediction" for method in method_predictions]]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(metadata):
            values = {
                "image_id": row["image_id"],
                "group": row.get("group", ""),
                "source": row.get("source", ""),
                "task": task_name,
                "true_label": true_labels[index],
            }
            for method, predictions in method_predictions.items():
                values[f"{method}_prediction"] = predictions[index]
            writer.writerow(values)


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["task", "method", "accuracy", "classes", "confusion_matrix"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = input_dir / "embeddings.npy"
    metadata_path = input_dir / "metadata.csv"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings do not exist: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    metadata = load_metadata(metadata_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match embeddings ({embeddings.shape[0]}).")

    tasks = {
        "source_style": [source_label(row) for row in metadata],
        "scene_distance": [scene_label(row) for row in metadata],
    }

    summary_rows: list[dict[str, str]] = []
    for task_name, labels in tasks.items():
        method_predictions: dict[str, list[str]] = {}
        for method in ["knn", "linear"]:
            predictions, accuracy, classes, matrix = leave_one_out_predictions(embeddings, labels, method, args.knn_k)
            method_predictions[method] = predictions
            summary_rows.append(
                {
                    "task": task_name,
                    "method": method,
                    "accuracy": f"{accuracy:.4f}",
                    "classes": "|".join(classes),
                    "confusion_matrix": np.array2string(matrix, separator=","),
                }
            )
        write_predictions(metadata, task_name, labels, method_predictions, output_dir / f"{task_name}_predictions.csv")

    write_summary(summary_rows, output_dir / "summary.csv")
    print(f"Processed {embeddings.shape[0]} embeddings.")
    print(f"Summary: {output_dir / 'summary.csv'}")
    print(f"Predictions: {output_dir / '<task>_predictions.csv'}")


if __name__ == "__main__":
    main()
