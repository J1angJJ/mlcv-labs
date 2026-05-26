from __future__ import annotations

import argparse
from pathlib import Path

from src.puffin_counting.annotations import split_image_dataset
from src.puffin_counting.dataset import (
    create_image_manifest,
    validate_count_annotations,
    validate_point_annotations,
)
from src.puffin_counting.density import run_density_demo, run_density_from_points
from src.puffin_counting.detection import run_yolo_detection
from src.puffin_counting.evaluation import evaluate_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Puffin counting utilities for the final computer vision assignment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Count birds using an optional YOLO detector.")
    detect.add_argument("--image-dir", type=Path, required=True, help="Directory containing input images.")
    detect.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path or name.")
    detect.add_argument("--output-dir", type=Path, default=Path("outputs/detection"))
    detect.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    detect.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    detect.add_argument("--class-name", type=str, default="bird", help="Class name to count.")

    density = subparsers.add_parser("density", help="Create density maps from point annotations.")
    density.add_argument("--points-csv", type=Path, required=True)
    density.add_argument("--image-dir", type=Path, required=True)
    density.add_argument("--output-dir", type=Path, default=Path("outputs/density"))
    density.add_argument("--sigma", type=float, default=15.0)

    demo = subparsers.add_parser("density-demo", help="Run a synthetic density-counting demo.")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/density_demo"))
    demo.add_argument("--num-points", type=int, default=25)
    demo.add_argument("--sigma", type=float, default=15.0)
    demo.add_argument("--seed", type=int, default=42)

    split = subparsers.add_parser("split", help="Create train/validation/test splits for image files.")
    split.add_argument("--image-dir", type=Path, required=True)
    split.add_argument("--output", type=Path, default=Path("outputs/splits.csv"))
    split.add_argument("--train", type=float, default=0.7)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)

    manifest = subparsers.add_parser("manifest", help="Create a CSV manifest from an image directory.")
    manifest.add_argument("--image-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, default=Path("outputs/image_manifest.csv"))

    validate = subparsers.add_parser("validate-data", help="Validate image count and point annotations.")
    validate.add_argument("--image-dir", type=Path, required=True)
    validate.add_argument("--counts-csv", type=Path)
    validate.add_argument("--points-csv", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predicted counts against ground truth.")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--prediction-column", type=str, default="target_count")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "detect":
        run_yolo_detection(
            image_dir=args.image_dir,
            model_name=args.model,
            output_dir=args.output_dir,
            conf=args.conf,
            iou=args.iou,
            class_name=args.class_name,
        )
    elif args.command == "density":
        run_density_from_points(
            points_csv=args.points_csv,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            sigma=args.sigma,
        )
    elif args.command == "density-demo":
        run_density_demo(
            output_dir=args.output_dir,
            num_points=args.num_points,
            sigma=args.sigma,
            seed=args.seed,
        )
    elif args.command == "split":
        split_image_dataset(
            image_dir=args.image_dir,
            output_csv=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
    elif args.command == "manifest":
        create_image_manifest(image_dir=args.image_dir, output_csv=args.output)
    elif args.command == "validate-data":
        if args.counts_csv is None and args.points_csv is None:
            raise SystemExit("validate-data requires --counts-csv and/or --points-csv")
        if args.counts_csv is not None:
            validate_count_annotations(image_dir=args.image_dir, counts_csv=args.counts_csv)
        if args.points_csv is not None:
            validate_point_annotations(image_dir=args.image_dir, points_csv=args.points_csv)
    elif args.command == "evaluate":
        evaluate_counts(
            ground_truth_csv=args.ground_truth,
            predictions_csv=args.predictions,
            output_dir=args.output_dir,
            prediction_column=args.prediction_column,
        )


if __name__ == "__main__":
    main()
