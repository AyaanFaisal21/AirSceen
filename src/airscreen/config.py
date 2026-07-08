from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AirScreenConfig:
    camera_index: int = 0
    dry_run: bool = False
    debug_preview: bool = False
    pinch_click_threshold: float = 0.055
    pinch_release_threshold: float = 0.075
    pointer_smoothing: float = 0.35
    pointer_enabled: bool = False
    gaze_enabled: bool = False
    landmark_record_path: Path | None = None
    gaze_calibration_profile_path: Path | None = None
