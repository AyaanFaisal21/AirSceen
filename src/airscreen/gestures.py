from __future__ import annotations

from dataclasses import dataclass
from math import dist

from airscreen.landmarks import HandLandmarks, Landmark


@dataclass(frozen=True, slots=True)
class PinchState:
    is_pinching: bool
    distance: float
    clicked: bool


class PinchClickDetector:
    """Stateful hysteresis detector for middle-thumb click gestures."""

    def __init__(self, click_threshold: float, release_threshold: float) -> None:
        if click_threshold <= 0:
            raise ValueError("click_threshold must be greater than zero")
        if release_threshold <= click_threshold:
            raise ValueError("release_threshold must be greater than click_threshold")

        self._click_threshold = click_threshold
        self._release_threshold = release_threshold
        self._pinching = False

    def update(self, hand: HandLandmarks) -> PinchState:
        if hand.middle_tip is None:
            raise ValueError("middle_tip is required for middle-thumb pinch detection")

        pinch_distance = normalized_distance(hand.middle_tip, hand.thumb_tip)
        clicked = False

        if not self._pinching and pinch_distance <= self._click_threshold:
            self._pinching = True
            clicked = True
        elif self._pinching and pinch_distance >= self._release_threshold:
            self._pinching = False

        return PinchState(
            is_pinching=self._pinching,
            distance=pinch_distance,
            clicked=clicked,
        )


def normalized_distance(a: Landmark, b: Landmark) -> float:
    return dist((a.x, a.y, a.z), (b.x, b.y, b.z))
