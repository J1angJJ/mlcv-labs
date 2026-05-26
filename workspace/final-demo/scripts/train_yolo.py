from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO detection model.")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained model checkpoint or model yaml.")
    parser.add_argument("--data", default="configs/seabirds_v6_local.yaml", help="Dataset yaml path.")
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="exp001_yolo26n_baseline", help="Experiment name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing output directory.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data)),
        project=str(Path(args.project)),
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
