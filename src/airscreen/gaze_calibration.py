from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from airscreen.vision.gaze_tracker import GazeEstimate


@dataclass(frozen=True, slots=True)
class GazeCalibrationTarget:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class GazeCalibrationSample:
    target: GazeCalibrationTarget
    raw_gaze: GazeEstimate


@dataclass(frozen=True, slots=True)
class AxisCalibration:
    scale: float
    offset: float

    def apply(self, value: float) -> float:
        return value * self.scale + self.offset


@dataclass(frozen=True, slots=True)
class GazeCalibrationProfile:
    x_axis: AxisCalibration
    y_axis: AxisCalibration

    SCHEMA_VERSION = 1

    def apply(self, estimate: GazeEstimate) -> GazeEstimate:
        return GazeEstimate(
            x=clamp_normalized(self.x_axis.apply(estimate.x)),
            y=clamp_normalized(self.y_axis.apply(estimate.y)),
            confidence=estimate.confidence,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "x_axis": {
                "scale": self.x_axis.scale,
                "offset": self.x_axis.offset,
            },
            "y_axis": {
                "scale": self.y_axis.scale,
                "offset": self.y_axis.offset,
            },
        }

    @classmethod
    def load(cls, path: Path) -> GazeCalibrationProfile:
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_payload(cls, payload: object) -> GazeCalibrationProfile:
        if not isinstance(payload, dict):
            raise ValueError("Gaze calibration profile must be a JSON object.")

        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported gaze calibration profile schema version.")

        x_axis = payload.get("x_axis")
        y_axis = payload.get("y_axis")
        if not isinstance(x_axis, dict) or not isinstance(y_axis, dict):
            raise ValueError("Gaze calibration profile must define x_axis and y_axis.")

        return cls(
            x_axis=axis_from_payload(x_axis),
            y_axis=axis_from_payload(y_axis),
        )


def guided_gaze_targets() -> tuple[GazeCalibrationTarget, ...]:
    return (
        GazeCalibrationTarget(0.1, 0.1),
        GazeCalibrationTarget(0.5, 0.1),
        GazeCalibrationTarget(0.9, 0.1),
        GazeCalibrationTarget(0.1, 0.5),
        GazeCalibrationTarget(0.5, 0.5),
        GazeCalibrationTarget(0.9, 0.5),
        GazeCalibrationTarget(0.1, 0.9),
        GazeCalibrationTarget(0.5, 0.9),
        GazeCalibrationTarget(0.9, 0.9),
    )


def build_gaze_calibration_profile(
    samples: Sequence[GazeCalibrationSample],
    min_confidence: float = 0.5,
) -> GazeCalibrationProfile:
    usable_samples = [sample for sample in samples if sample.raw_gaze.confidence >= min_confidence]
    if len(usable_samples) < 2:
        raise ValueError("At least two confident gaze calibration samples are required.")

    return GazeCalibrationProfile(
        x_axis=fit_axis(
            [sample.raw_gaze.x for sample in usable_samples],
            [sample.target.x for sample in usable_samples],
        ),
        y_axis=fit_axis(
            [sample.raw_gaze.y for sample in usable_samples],
            [sample.target.y for sample in usable_samples],
        ),
    )


def fit_axis(raw_values: Sequence[float], target_values: Sequence[float]) -> AxisCalibration:
    if len(raw_values) != len(target_values):
        raise ValueError("Raw and target value counts must match.")

    if len(raw_values) < 2:
        raise ValueError("At least two values are required to fit a calibration axis.")

    raw_mean = sum(raw_values) / len(raw_values)
    target_mean = sum(target_values) / len(target_values)
    variance = sum((value - raw_mean) ** 2 for value in raw_values)
    if variance <= 1e-12:
        raise ValueError("Calibration samples must span more than one raw gaze value.")

    covariance = sum(
        (raw - raw_mean) * (target - target_mean)
        for raw, target in zip(raw_values, target_values, strict=True)
    )
    scale = covariance / variance
    offset = target_mean - (scale * raw_mean)
    return AxisCalibration(scale=scale, offset=offset)


def axis_from_payload(payload: dict[object, object]) -> AxisCalibration:
    scale = payload.get("scale")
    offset = payload.get("offset")
    if not isinstance(scale, int | float) or not isinstance(offset, int | float):
        raise ValueError("Calibration axis must define numeric scale and offset.")

    return AxisCalibration(scale=float(scale), offset=float(offset))


def clamp_normalized(value: float) -> float:
    return min(max(value, 0.0), 1.0)
