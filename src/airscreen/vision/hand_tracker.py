from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, cast

from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.vision.camera import Frame


class HandTracker(Protocol):
    def track(self, frame: Frame) -> Sequence[HandLandmarks]:
        """Return normalized hand landmarks detected in a frame."""


class NormalizedLandmarkLike(Protocol):
    x: float
    y: float
    z: float


class MediaPipeHandLandmarksLike(Protocol):
    landmark: Sequence[NormalizedLandmarkLike]


class MediaPipeHandResultLike(Protocol):
    multi_hand_landmarks: Sequence[MediaPipeHandLandmarksLike] | None


class MediaPipeHandsLike(Protocol):
    def process(self, image: object) -> MediaPipeHandResultLike: ...

    def close(self) -> None: ...


class Cv2ColorLike(Protocol):
    @property
    def COLOR_BGR2RGB(self) -> int: ...

    def cvtColor(self, image: object, code: int) -> object: ...


class MediaPipeHandTracker:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def __init__(
        self,
        hands: MediaPipeHandsLike | None = None,
        cv2_module: Cv2ColorLike | None = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = hands
        self._cv2_module = cv2_module
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def track(self, frame: Frame) -> Sequence[HandLandmarks]:
        hands = self._load_hands()
        cv2 = self._load_cv2()
        rgb_frame = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks is None:
            return []

        return [self._convert_hand(hand) for hand in results.multi_hand_landmarks]

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

    def _load_hands(self) -> MediaPipeHandsLike:
        if self._hands is None:
            try:
                mp = cast(Any, import_module("mediapipe"))
            except ImportError as exc:
                raise RuntimeError(
                    "MediaPipe is not installed. Install the vision extras with "
                    '`python -m pip install -e ".[vision]"`.'
                ) from exc

            self._hands = cast(
                MediaPipeHandsLike,
                mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self._max_num_hands,
                    min_detection_confidence=self._min_detection_confidence,
                    min_tracking_confidence=self._min_tracking_confidence,
                ),
            )

        return self._hands

    def _load_cv2(self) -> Cv2ColorLike:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            cv2 = import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(Cv2ColorLike, cv2)
        return self._cv2_module

    def _convert_hand(self, hand: MediaPipeHandLandmarksLike) -> HandLandmarks:
        landmarks = hand.landmark

        return HandLandmarks(
            wrist=self._landmark_at(landmarks, self.WRIST),
            thumb_tip=self._landmark_at(landmarks, self.THUMB_TIP),
            index_tip=self._landmark_at(landmarks, self.INDEX_TIP),
            middle_tip=self._landmark_at(landmarks, self.MIDDLE_TIP),
            ring_tip=self._landmark_at(landmarks, self.RING_TIP),
            pinky_tip=self._landmark_at(landmarks, self.PINKY_TIP),
        )

    def _landmark_at(
        self,
        landmarks: Sequence[NormalizedLandmarkLike],
        index: int,
    ) -> Landmark:
        landmark = landmarks[index]
        return Landmark(x=landmark.x, y=landmark.y, z=landmark.z)
