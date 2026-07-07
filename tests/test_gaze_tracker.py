from __future__ import annotations

from dataclasses import dataclass

import pytest

from airscreen.vision.camera import Frame
from airscreen.vision.gaze_tracker import MediaPipeGazeTracker


@dataclass(frozen=True, slots=True)
class FakeFaceLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class FakeFace:
    landmark: list[FakeFaceLandmark]


@dataclass(frozen=True, slots=True)
class FakeFaceResults:
    multi_face_landmarks: list[FakeFace] | None


class FakeFaceMesh:
    def __init__(self, results: FakeFaceResults) -> None:
        self.results = results
        self.processed_image: object | None = None
        self.closed = False

    def process(self, image: object) -> FakeFaceResults:
        self.processed_image = image
        return self.results

    def close(self) -> None:
        self.closed = True


class FakeCv2:
    COLOR_BGR2RGB = 42

    def __init__(self) -> None:
        self.converted_image: object | None = None
        self.convert_code: int | None = None
        self.rgb_image = object()

    def cvtColor(self, image: object, code: int) -> object:
        self.converted_image = image
        self.convert_code = code
        return self.rgb_image


def face_with_iris_landmarks() -> FakeFace:
    landmarks = [FakeFaceLandmark(x=0.0, y=0.0, z=0.0) for _ in range(478)]
    for index in MediaPipeGazeTracker.IRIS_LANDMARKS:
        landmarks[index] = FakeFaceLandmark(x=0.4, y=0.6, z=0.0)
    return FakeFace(landmark=landmarks)


def test_mediapipe_gaze_tracker_estimates_from_iris_landmarks() -> None:
    frame_data = object()
    frame = Frame(width=640, height=480, data=frame_data)
    cv2 = FakeCv2()
    face_mesh = FakeFaceMesh(FakeFaceResults(multi_face_landmarks=[face_with_iris_landmarks()]))
    tracker = MediaPipeGazeTracker(face_mesh=face_mesh, cv2_module=cv2)

    estimate = tracker.estimate(frame)

    assert cv2.converted_image is frame_data
    assert cv2.convert_code == FakeCv2.COLOR_BGR2RGB
    assert face_mesh.processed_image is cv2.rgb_image
    assert estimate is not None
    assert estimate.x == pytest.approx(0.4)
    assert estimate.y == pytest.approx(0.6)
    assert estimate.confidence == 0.75


def test_mediapipe_gaze_tracker_returns_none_when_no_face_is_detected() -> None:
    tracker = MediaPipeGazeTracker(
        face_mesh=FakeFaceMesh(FakeFaceResults(multi_face_landmarks=None)),
        cv2_module=FakeCv2(),
    )

    assert tracker.estimate(Frame(width=10, height=10, data=object())) is None


def test_mediapipe_gaze_tracker_closes_underlying_face_mesh() -> None:
    face_mesh = FakeFaceMesh(FakeFaceResults(multi_face_landmarks=[]))
    tracker = MediaPipeGazeTracker(face_mesh=face_mesh, cv2_module=FakeCv2())

    tracker.close()

    assert face_mesh.closed is True
