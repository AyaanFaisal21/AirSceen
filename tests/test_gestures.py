from __future__ import annotations

import pytest

from airscreen.gestures import PinchClickDetector
from airscreen.landmarks import HandLandmarks, Landmark


def hand_with_pinch_distance(distance: float) -> HandLandmarks:
    return HandLandmarks(
        wrist=Landmark(0.0, 0.0),
        thumb_tip=Landmark(0.0, 0.0),
        index_tip=Landmark(0.5, 0.0),
        middle_tip=Landmark(distance, 0.0),
    )


def test_pinch_detector_clicks_once_until_released() -> None:
    detector = PinchClickDetector(click_threshold=0.05, release_threshold=0.08)

    first = detector.update(hand_with_pinch_distance(0.04))
    second = detector.update(hand_with_pinch_distance(0.03))
    released = detector.update(hand_with_pinch_distance(0.09))
    third = detector.update(hand_with_pinch_distance(0.04))

    assert first.clicked is True
    assert second.clicked is False
    assert released.is_pinching is False
    assert third.clicked is True


def test_pinch_detector_uses_release_hysteresis() -> None:
    detector = PinchClickDetector(click_threshold=0.05, release_threshold=0.08)

    detector.update(hand_with_pinch_distance(0.04))
    state = detector.update(hand_with_pinch_distance(0.06))

    assert state.is_pinching is True
    assert state.clicked is False


def test_pinch_detector_ignores_index_thumb_distance() -> None:
    detector = PinchClickDetector(click_threshold=0.05, release_threshold=0.08)
    hand = HandLandmarks(
        wrist=Landmark(0.0, 0.0),
        thumb_tip=Landmark(0.0, 0.0),
        index_tip=Landmark(0.01, 0.0),
        middle_tip=Landmark(0.5, 0.0),
    )

    state = detector.update(hand)

    assert state.clicked is False
    assert state.is_pinching is False
    assert state.distance == 0.5


def test_pinch_detector_requires_middle_finger_landmark() -> None:
    detector = PinchClickDetector(click_threshold=0.05, release_threshold=0.08)
    hand = HandLandmarks(
        wrist=Landmark(0.0, 0.0),
        thumb_tip=Landmark(0.0, 0.0),
        index_tip=Landmark(0.01, 0.0),
    )

    with pytest.raises(ValueError, match="middle_tip is required"):
        detector.update(hand)


def test_pinch_detector_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        PinchClickDetector(click_threshold=0.0, release_threshold=0.08)

    with pytest.raises(ValueError):
        PinchClickDetector(click_threshold=0.05, release_threshold=0.04)
