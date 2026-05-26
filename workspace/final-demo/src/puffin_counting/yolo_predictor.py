from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2

from .io_utils import ensure_dir
from .model_interface import BoxPrediction, CountPrediction


PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
]


def _load_yolo(model_path: Path | str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for YOLO inference") from exc
    return YOLO(str(model_path))


def _find_class_id(names: dict[int, str] | list[str], class_name: str) -> int:
    normalized = class_name.lower()
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, name in items:
        if str(name).lower() == normalized:
            return int(class_id)
    raise ValueError(f"Class {class_name!r} was not found in model classes: {names}")


def box_to_dict(box: BoxPrediction) -> dict[str, object]:
    return asdict(box)


class YOLOCountPredictor:
    def __init__(
        self,
        model_path: Path,
        target_class: str = "puffin",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | int | None = None,
        output_dir: Path = Path("outputs/backend_predictions"),
        save_visuals: bool = True,
        draw_all_classes: bool = True,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
        self.model_path = model_path
        self.target_class = target_class
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = None if device in (None, "", "auto") else device
        self.output_dir = ensure_dir(output_dir)
        self.save_visuals = save_visuals
        self.draw_all_classes = draw_all_classes
        self.model = _load_yolo(model_path)
        self.names = self.model.names
        self.target_class_id = _find_class_id(self.names, target_class)

    def predict_image(self, image_path: Path) -> CountPrediction:
        result = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        boxes: list[BoxPrediction] = []
        target_confidences: list[float] = []
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
            is_target = class_id == self.target_class_id
            prediction = BoxPrediction(
                class_id=class_id,
                class_name=str(self.names[class_id]),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                is_target=is_target,
            )
            if is_target:
                boxes.append(prediction)
                target_confidences.append(confidence)

        figure_path: str | None = None
        if self.save_visuals:
            figure = self.output_dir / f"{image_path.stem}_prediction.jpg"
            self._save_visual(image_path=image_path, result_boxes=result.boxes, output_path=figure)
            figure_path = str(figure)

        return CountPrediction(
            image=image_path.name,
            count=len(boxes),
            confidence=(sum(target_confidences) / len(target_confidences)) if target_confidences else None,
            figure_path=figure_path,
            boxes=boxes,
            all_detections=len(result.boxes),
            model_name=str(self.model_path),
            target_class=self.target_class,
            conf_threshold=self.conf,
            iou_threshold=self.iou,
        )

    def _save_visual(self, image_path: Path, result_boxes, output_path: Path) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV failed to read image: {image_path}")

        for box in result_boxes:
            class_id = int(box.cls[0].cpu().item())
            if not self.draw_all_classes and class_id != self.target_class_id:
                continue
            confidence = float(box.conf[0].cpu().item())
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].cpu().tolist()]
            is_target = class_id == self.target_class_id
            color = (0, 220, 0) if is_target else PALETTE[class_id % len(PALETTE)]
            label = f"{class_id}:{self.names[class_id]} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if is_target else 1)
            label_width = min(image.shape[1] - 1, x1 + 8 * len(label) + 8)
            cv2.rectangle(image, (x1, max(0, y1 - 22)), (label_width, y1), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
