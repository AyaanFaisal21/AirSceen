from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from airscreen.vision.camera import Frame


@dataclass(frozen=True, slots=True)
class GazeEstimate:
    x: float
    y: float
    confidence: float


class GazeTracker(Protocol):
    def estimate(self, frame: Frame) -> GazeEstimate | None:
        """Return a normalized gaze estimate for the frame, if one is visible."""


class FaceLandmarkLike(Protocol):
    x: float
    y: float
    z: float


class FaceLandmarksLike(Protocol):
    landmark: Sequence[FaceLandmarkLike]


class FaceMeshResultLike(Protocol):
    multi_face_landmarks: Sequence[FaceLandmarksLike] | None


class FaceMeshLike(Protocol):
    def process(self, image: object) -> FaceMeshResultLike: ...

    def close(self) -> None: ...


class Cv2ColorLike(Protocol):
    @property
    def COLOR_BGR2RGB(self) -> int: ...

    def cvtColor(self, image: object, code: int) -> object: ...


class MediaPipeGazeTracker:
    IRIS_LANDMARKS = (468, 469, 470, 471, 472, 473, 474, 475, 476, 477)
    FALLBACK_EYE_LANDMARKS = (33, 263)

    def __init__(
        self,
        face_mesh: FaceMeshLike | None = None,
        cv2_module: Cv2ColorLike | None = None,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._face_mesh = face_mesh
        self._cv2_module = cv2_module
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def estimate(self, frame: Frame) -> GazeEstimate | None:
        face_mesh = self._load_face_mesh()
        cv2 = self._load_cv2()
        rgb_frame = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks is None:
            return None

        face = results.multi_face_landmarks[0]
        landmarks, confidence = self._select_landmarks(face.landmark)
        if not landmarks:
            return None

        x = sum(landmark.x for landmark in landmarks) / len(landmarks)
        y = sum(landmark.y for landmark in landmarks) / len(landmarks)

        return GazeEstimate(
            x=self._clamp(x),
            y=self._clamp(y),
            confidence=confidence,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()

    def _load_face_mesh(self) -> FaceMeshLike:
        if self._face_mesh is None:
            try:
                import mediapipe as mp  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "MediaPipe is not installed. Install the vision extras with "
                    '`python -m pip install -e ".[vision]"`.'
                ) from exc

            self._face_mesh = cast(
                FaceMeshLike,
                mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self._min_detection_confidence,
                    min_tracking_confidence=self._min_tracking_confidence,
                ),
            )

        return self._face_mesh

    def _load_cv2(self) -> Cv2ColorLike:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(Cv2ColorLike, cv2)
        return self._cv2_module

    def _select_landmarks(
        self,
        landmarks: Sequence[FaceLandmarkLike],
    ) -> tuple[Sequence[FaceLandmarkLike], float]:
        if len(landmarks) > max(self.IRIS_LANDMARKS):
            return [landmarks[index] for index in self.IRIS_LANDMARKS], 0.75

        if len(landmarks) > max(self.FALLBACK_EYE_LANDMARKS):
            return [landmarks[index] for index in self.FALLBACK_EYE_LANDMARKS], 0.35

        return [], 0.0

    def _clamp(self, value: float) -> float:
        return min(max(value, 0.0), 1.0)
