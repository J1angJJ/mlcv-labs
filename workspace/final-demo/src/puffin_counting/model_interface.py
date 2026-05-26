from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class CountPrediction:
    image: str
    count: int
    confidence: float | None = None
    figure_path: str | None = None
    warnings: list[str] = field(default_factory=list)


class CountPredictor(Protocol):
    def predict_image(self, image_path: Path) -> CountPrediction:
        """Return a puffin count prediction for one image."""


class NotConfiguredPredictor:
    def predict_image(self, image_path: Path) -> CountPrediction:
        return CountPrediction(
            image=image_path.name,
            count=0,
            confidence=None,
            warnings=["No model predictor is configured yet."],
        )

