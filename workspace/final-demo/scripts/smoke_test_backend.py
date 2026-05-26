from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the FastAPI prediction endpoint.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "data/Seabirds.v6i.yolo26/test/images/"
            "1127_maine-puffins-1000x644_jpeg.rf.21367ecf0d8ee509f74ef00728fab9a3.jpg"
        ),
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image does not exist: {args.image}")

    client = TestClient(app)
    health = client.get("/health")
    health.raise_for_status()
    print("health:", health.json())

    with args.image.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": (args.image.name, handle, "image/jpeg")},
        )
    response.raise_for_status()
    payload = response.json()
    print("prediction:")
    print(f"  file: {payload['file_name']}")
    print(f"  count: {payload['count']}")
    print(f"  all_detections: {payload['all_detections']}")
    print(f"  elapsed_ms: {payload['elapsed_ms']}")
    print(f"  figure_path: {payload['figure_path']}")


if __name__ == "__main__":
    main()
