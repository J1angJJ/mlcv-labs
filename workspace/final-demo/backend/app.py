from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised only when FastAPI is absent
    raise RuntimeError(
        "FastAPI backend dependencies are not installed. "
        "Install them with: pip install fastapi uvicorn python-multipart"
    ) from exc

from src.puffin_counting.config import load_config, resolve_path
from src.puffin_counting.io_utils import IMAGE_EXTENSIONS, ensure_dir
from src.puffin_counting.model_interface import CountPredictor
from src.puffin_counting.yolo_predictor import YOLOCountPredictor


CONFIG_PATH = Path(os.getenv("PUFFIN_BACKEND_CONFIG", "configs/default.yaml")).resolve()
CONFIG = load_config(CONFIG_PATH)


def _config_value(section: str, key: str, default):
    value = CONFIG.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def _resolve_config_path(section: str, key: str, default: str) -> Path:
    return resolve_path(CONFIG_PATH, _config_value(section, key, default))


UPLOAD_DIR = ensure_dir(_resolve_config_path("paths", "upload_dir", "data/uploads"))
BACKEND_OUTPUT_DIR = ensure_dir(_resolve_config_path("backend", "prediction_output_dir", "outputs/backend_predictions"))
FRONTEND_DIR = _resolve_config_path("backend", "frontend_dir", "frontend")
MODEL_PATH = _resolve_config_path("inference", "model_path", "runs/train/exp002_yolo11n_baseline/weights/best.pt")
TARGET_CLASS = str(_config_value("dataset", "class_name", "puffin"))
CONF = float(_config_value("inference", "conf", 0.25))
IOU = float(_config_value("inference", "iou", 0.7))
IMGSZ = int(_config_value("inference", "image_size", 640))
DEVICE = _config_value("inference", "device", "auto")
SAVE_UPLOADS = bool(_config_value("backend", "save_uploads", True))


def create_predictor() -> CountPredictor:
    return YOLOCountPredictor(
        model_path=MODEL_PATH,
        target_class=TARGET_CLASS,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        output_dir=BACKEND_OUTPUT_DIR,
        save_visuals=True,
        draw_all_classes=True,
    )


predictor = create_predictor()
app = FastAPI(title="Puffin Counting Backend", version="0.2.0")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/prediction-files", StaticFiles(directory=BACKEND_OUTPUT_DIR), name="prediction-files")


def _prediction_url(figure_path: str | None) -> str | None:
    if not figure_path:
        return None
    path = Path(figure_path).resolve()
    try:
        relative = path.relative_to(BACKEND_OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return f"/prediction-files/{relative.as_posix()}"


@app.get("/", include_in_schema=False)
def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend is not configured")
    return FileResponse(index_path)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "target_class": TARGET_CLASS,
    }


@app.get("/model")
def model_info() -> dict[str, object]:
    return {
        "model_path": str(MODEL_PATH),
        "target_class": TARGET_CLASS,
        "conf": CONF,
        "iou": IOU,
        "image_size": IMGSZ,
        "device": DEVICE,
        "upload_dir": str(UPLOAD_DIR),
        "prediction_output_dir": str(BACKEND_OUTPUT_DIR),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image suffix: {suffix}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    upload_name = f"{uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / upload_name
    if SAVE_UPLOADS:
        upload_path.write_bytes(data)
        inference_path = upload_path
    else:
        upload_path.write_bytes(data)
        inference_path = upload_path

    started = perf_counter()
    try:
        prediction = predictor.predict_image(inference_path)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    return {
        "upload_id": upload_path.stem,
        "file_name": file.filename,
        "stored_path": str(upload_path),
        "model_path": prediction.model_name,
        "target_class": prediction.target_class,
        "count": prediction.count,
        "mean_confidence": prediction.confidence,
        "all_detections": prediction.all_detections,
        "conf_threshold": prediction.conf_threshold,
        "iou_threshold": prediction.iou_threshold,
        "elapsed_ms": elapsed_ms,
        "figure_path": prediction.figure_path,
        "figure_url": _prediction_url(prediction.figure_path),
        "boxes": [asdict(box) for box in prediction.boxes],
        "warnings": prediction.warnings,
    }
