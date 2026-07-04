from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    wrist: Landmark
    thumb_tip: Landmark
    index_tip: Landmark
    middle_tip: Landmark | None = None
    ring_tip: Landmark | None = None
    pinky_tip: Landmark | None = None
