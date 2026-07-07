from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.vision.camera import Frame
from airscreen.vision.gaze_tracker import GazeEstimate


class LandmarkSessionRecorder:
    """Writes frame-level landmark samples as JSON Lines."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: TextIO | None = None

    def record(
        self,
        frame_index: int,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        gaze_estimate: GazeEstimate | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "frame_index": frame_index,
            "frame": {
                "width": frame.width,
                "height": frame.height,
            },
            "hands": [self._hand_payload(hand) for hand in hands],
            "gaze": self._gaze_payload(gaze_estimate) if gaze_estimate is not None else None,
        }

        file = self._open()
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _open(self) -> TextIO:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", encoding="utf-8")

        return self._file

    def _hand_payload(self, hand: HandLandmarks) -> dict[str, object]:
        return {
            "wrist": self._landmark_payload(hand.wrist),
            "thumb_tip": self._landmark_payload(hand.thumb_tip),
            "index_tip": self._landmark_payload(hand.index_tip),
            "middle_tip": self._optional_landmark_payload(hand.middle_tip),
            "ring_tip": self._optional_landmark_payload(hand.ring_tip),
            "pinky_tip": self._optional_landmark_payload(hand.pinky_tip),
        }

    def _optional_landmark_payload(self, landmark: Landmark | None) -> dict[str, float] | None:
        if landmark is None:
            return None

        return self._landmark_payload(landmark)

    def _landmark_payload(self, landmark: Landmark) -> dict[str, float]:
        return {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
        }

    def _gaze_payload(self, gaze_estimate: GazeEstimate) -> dict[str, float]:
        return {
            "x": gaze_estimate.x,
            "y": gaze_estimate.y,
            "confidence": gaze_estimate.confidence,
        }
