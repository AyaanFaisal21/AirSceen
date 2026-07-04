from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AirScreenConfig:
    camera_index: int = 0
    dry_run: bool = False
    pinch_click_threshold: float = 0.055
    pinch_release_threshold: float = 0.075
    pointer_smoothing: float = 0.35
    pointer_enabled: bool = False
    gaze_enabled: bool = False
