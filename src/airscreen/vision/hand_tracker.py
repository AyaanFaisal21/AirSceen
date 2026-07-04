from __future__ import annotations

from collections.abc import Sequence

from airscreen.landmarks import HandLandmarks
from airscreen.vision.camera import Frame


class HandTracker:
    def track(self, frame: Frame) -> Sequence[HandLandmarks]:
        raise NotImplementedError
