from __future__ import annotations

from dataclasses import dataclass

from airscreen.vision.camera import Frame


@dataclass(frozen=True, slots=True)
class GazeEstimate:
    x: float
    y: float
    confidence: float


class GazeTracker:
    def estimate(self, frame: Frame) -> GazeEstimate | None:
        raise NotImplementedError
