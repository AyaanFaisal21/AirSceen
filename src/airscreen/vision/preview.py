from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from airscreen.config import AirScreenConfig
from airscreen.gestures import PinchClickDetector, PinchState
from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.vision.camera import Frame, FrameSource, OpenCVCameraSource
from airscreen.vision.gaze_tracker import GazeEstimate, GazeTracker, MediaPipeGazeTracker
from airscreen.vision.hand_tracker import HandTracker, MediaPipeHandTracker


Color = tuple[int, int, int]
Point = tuple[int, int]


class PreviewCv2Like(Protocol):
    @property
    def COLOR_BGR2RGB(self) -> int: ...

    @property
    def FONT_HERSHEY_SIMPLEX(self) -> int: ...

    @property
    def LINE_AA(self) -> int: ...

    def VideoCapture(self, camera_index: int) -> object: ...

    def cvtColor(self, image: object, code: int) -> object: ...

    def rectangle(
        self,
        image: object,
        point_a: Point,
        point_b: Point,
        color: Color,
        thickness: int,
    ) -> object: ...

    def circle(
        self,
        image: object,
        center: Point,
        radius: int,
        color: Color,
        thickness: int,
    ) -> object: ...

    def line(
        self,
        image: object,
        point_a: Point,
        point_b: Point,
        color: Color,
        thickness: int,
    ) -> object: ...

    def putText(
        self,
        image: object,
        text: str,
        origin: Point,
        font_face: int,
        font_scale: float,
        color: Color,
        thickness: int,
        line_type: int,
    ) -> object: ...

    def imshow(self, window_name: str, image: object) -> None: ...

    def waitKey(self, delay_ms: int) -> int: ...

    def destroyAllWindows(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FingerMarker:
    label: str
    landmark: Landmark
    color: Color


class FingerOverlayRenderer:
    BOX_RADIUS = 12

    def __init__(self, cv2_module: PreviewCv2Like | None = None) -> None:
        self._cv2_module = cv2_module

    def render(
        self,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        pinch_state: PinchState | None,
        gaze_estimate: GazeEstimate | None = None,
    ) -> object:
        cv2 = self._load_cv2()
        image = frame.data

        for hand in hands:
            self._draw_hand(cv2, image, frame, hand)

        if gaze_estimate is not None:
            self._draw_gaze_estimate(cv2, image, frame, gaze_estimate)

        self._draw_pinch_state(cv2, image, pinch_state)
        return image

    def _draw_hand(
        self,
        cv2: PreviewCv2Like,
        image: object,
        frame: Frame,
        hand: HandLandmarks,
    ) -> None:
        thumb = self._to_pixel(frame, hand.thumb_tip)
        index = self._to_pixel(frame, hand.index_tip)
        cv2.line(image, thumb, index, (255, 255, 255), 2)

        for marker in self._finger_markers(hand):
            center = self._to_pixel(frame, marker.landmark)
            self._draw_finger_marker(cv2, image, marker, center)

    def _draw_finger_marker(
        self,
        cv2: PreviewCv2Like,
        image: object,
        marker: FingerMarker,
        center: Point,
    ) -> None:
        x, y = center
        radius = self.BOX_RADIUS
        top_left = (x - radius, y - radius)
        bottom_right = (x + radius, y + radius)

        cv2.rectangle(image, top_left, bottom_right, marker.color, 2)
        cv2.circle(image, center, 4, marker.color, -1)
        cv2.putText(
            image,
            marker.label,
            (x + radius + 4, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            marker.color,
            1,
            cv2.LINE_AA,
        )

    def _draw_pinch_state(
        self,
        cv2: PreviewCv2Like,
        image: object,
        pinch_state: PinchState | None,
    ) -> None:
        active = pinch_state is not None and pinch_state.is_pinching
        clicked = pinch_state is not None and pinch_state.clicked
        color = (0, 220, 0) if active else (40, 40, 220)
        label = "PINCH CLICK" if clicked else "PINCH HOLD" if active else "NO PINCH"

        cv2.rectangle(image, (12, 12), (232, 56), color, -1)
        cv2.putText(
            image,
            label,
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_gaze_estimate(
        self,
        cv2: PreviewCv2Like,
        image: object,
        frame: Frame,
        gaze_estimate: GazeEstimate,
    ) -> None:
        center = self._to_pixel(frame, Landmark(gaze_estimate.x, gaze_estimate.y))
        x, y = center
        color = (255, 80, 80)

        cv2.circle(image, center, 10, color, 2)
        cv2.line(image, (x - 18, y), (x + 18, y), color, 2)
        cv2.line(image, (x, y - 18), (x, y + 18), color, 2)
        cv2.putText(
            image,
            f"GAZE {gaze_estimate.confidence:.2f}",
            (x + 16, y - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _finger_markers(self, hand: HandLandmarks) -> Sequence[FingerMarker]:
        markers = [
            FingerMarker("THUMB", hand.thumb_tip, (0, 128, 255)),
            FingerMarker("INDEX", hand.index_tip, (0, 255, 255)),
        ]

        optional_fingers = (
            ("MIDDLE", hand.middle_tip, (255, 0, 255)),
            ("RING", hand.ring_tip, (255, 128, 0)),
            ("PINKY", hand.pinky_tip, (255, 255, 0)),
        )
        for label, landmark, color in optional_fingers:
            if landmark is not None:
                markers.append(FingerMarker(label, landmark, color))

        return markers

    def _to_pixel(self, frame: Frame, landmark: Landmark) -> Point:
        x = min(max(int(landmark.x * frame.width), 0), frame.width - 1)
        y = min(max(int(landmark.y * frame.height), 0), frame.height - 1)
        return x, y

    def _load_cv2(self) -> PreviewCv2Like:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(PreviewCv2Like, cv2)
        return self._cv2_module


class DebugPreviewRunner:
    WINDOW_NAME = "AirScreen Debug Preview"
    QUIT_KEYS = {ord("q"), 27}

    def __init__(
        self,
        config: AirScreenConfig,
        frame_source: FrameSource | None = None,
        hand_tracker: HandTracker | None = None,
        gaze_tracker: GazeTracker | None = None,
        renderer: FingerOverlayRenderer | None = None,
        cv2_module: PreviewCv2Like | None = None,
    ) -> None:
        self._config = config
        self._frame_source = frame_source
        self._hand_tracker = hand_tracker
        self._gaze_tracker = gaze_tracker
        self._renderer = renderer
        self._cv2_module = cv2_module

    def run(self) -> int:
        cv2 = self._load_cv2()
        frame_source = self._frame_source or OpenCVCameraSource(
            camera_index=self._config.camera_index,
            cv2_module=cv2,
        )
        hand_tracker = self._hand_tracker or MediaPipeHandTracker(cv2_module=cv2)
        gaze_tracker = self._load_gaze_tracker(cv2)
        renderer = self._renderer or FingerOverlayRenderer(cv2_module=cv2)
        pinch_detector = PinchClickDetector(
            click_threshold=self._config.pinch_click_threshold,
            release_threshold=self._config.pinch_release_threshold,
        )

        try:
            for frame in frame_source.frames():
                hands = hand_tracker.track(frame)
                gaze_estimate = gaze_tracker.estimate(frame) if gaze_tracker is not None else None
                pinch_state = pinch_detector.update(hands[0]) if hands else None
                image = renderer.render(frame, hands, pinch_state, gaze_estimate)
                cv2.imshow(self.WINDOW_NAME, image)

                key = cv2.waitKey(1) & 0xFF
                if key in self.QUIT_KEYS:
                    break
        finally:
            close = getattr(hand_tracker, "close", None)
            if callable(close):
                close()
            close = getattr(gaze_tracker, "close", None)
            if callable(close):
                close()
            cv2.destroyAllWindows()

        return 0

    def _load_gaze_tracker(self, cv2: PreviewCv2Like) -> GazeTracker | None:
        if not self._config.gaze_enabled:
            return None

        return self._gaze_tracker or MediaPipeGazeTracker(cv2_module=cv2)

    def _load_cv2(self) -> PreviewCv2Like:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(PreviewCv2Like, cv2)
        return self._cv2_module
