from __future__ import annotations

import pytest

from airscreen.gaze_calibration import (
    AxisCalibration,
    GazeCalibrationProfile,
    GazeCalibrationSample,
    GazeCalibrationTarget,
    build_gaze_calibration_profile,
    guided_gaze_targets,
)
from airscreen.vision.gaze_tracker import GazeEstimate


def test_guided_gaze_targets_cover_screen_center_and_edges() -> None:
    targets = guided_gaze_targets()

    assert GazeCalibrationTarget(0.5, 0.5) in targets
    assert GazeCalibrationTarget(0.1, 0.1) in targets
    assert GazeCalibrationTarget(0.9, 0.9) in targets
    assert len(targets) == 9


def test_build_gaze_calibration_profile_maps_raw_gaze_to_targets() -> None:
    samples = [
        GazeCalibrationSample(
            target=target,
            raw_gaze=GazeEstimate(
                x=(target.x + 0.10) / 1.25,
                y=(target.y - 0.08) / 0.80,
                confidence=0.9,
            ),
        )
        for target in guided_gaze_targets()
    ]

    profile = build_gaze_calibration_profile(samples)
    calibrated = profile.apply(GazeEstimate(x=(0.7 + 0.10) / 1.25, y=(0.3 - 0.08) / 0.80, confidence=0.8))

    assert calibrated.x == pytest.approx(0.7)
    assert calibrated.y == pytest.approx(0.3)
    assert calibrated.confidence == 0.8


def test_build_gaze_calibration_profile_rejects_low_confidence_samples() -> None:
    samples = [
        GazeCalibrationSample(
            target=GazeCalibrationTarget(0.1, 0.1),
            raw_gaze=GazeEstimate(x=0.1, y=0.1, confidence=0.1),
        ),
        GazeCalibrationSample(
            target=GazeCalibrationTarget(0.9, 0.9),
            raw_gaze=GazeEstimate(x=0.9, y=0.9, confidence=0.1),
        ),
    ]

    with pytest.raises(ValueError, match="confident"):
        build_gaze_calibration_profile(samples)


def test_gaze_calibration_profile_round_trips_json(tmp_path) -> None:
    path = tmp_path / "gaze-profile.json"
    profile = GazeCalibrationProfile(
        x_axis=AxisCalibration(scale=1.25, offset=-0.1),
        y_axis=AxisCalibration(scale=0.8, offset=0.08),
    )

    profile.save(path)
    loaded = GazeCalibrationProfile.load(path)

    assert loaded == profile
    assert loaded.apply(GazeEstimate(x=2.0, y=-1.0, confidence=0.4)) == GazeEstimate(
        x=1.0,
        y=0.0,
        confidence=0.4,
    )
