from __future__ import annotations

from pathlib import Path
from uuid import uuid4

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except ImportError as exc:  # pragma: no cover - exercised only when FastAPI is absent
    raise RuntimeError(
        "FastAPI backend dependencies are not installed. "
        "Install them later with: pip install fastapi uvicorn python-multipart"
    ) from exc

from src.puffin_counting.io_utils import IMAGE_EXTENSIONS, ensure_dir
from src.puffin_counting.model_interface import NotConfiguredPredictor


UPLOAD_DIR = ensure_dir(Path("data/uploads"))
predictor = NotConfiguredPredictor()
app = FastAPI(title="Puffin Counting Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image suffix: {suffix}")

    upload_name = f"{uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / upload_name
    data = await file.read()
    upload_path.write_bytes(data)

    prediction = predictor.predict_image(upload_path)
    return {
        "upload_id": upload_path.stem,
        "file_name": file.filename,
        "stored_path": str(upload_path),
        "count": prediction.count,
        "confidence": prediction.confidence,
        "figure_path": prediction.figure_path,
        "warnings": prediction.warnings,
    }

