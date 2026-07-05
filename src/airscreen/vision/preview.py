from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin
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

    def flip(self, image: object, flip_code: int) -> object: ...

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


@dataclass(frozen=True, slots=True)
class PinchParticle:
    origin: Point
    angle: float
    age: int = 0


def finger_markers(hand: HandLandmarks) -> Sequence[FingerMarker]:
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


def to_pixel(frame: Frame, landmark: Landmark) -> Point:
    mirrored_x = 1.0 - landmark.x
    x = min(max(int(mirrored_x * frame.width), 0), frame.width - 1)
    y = min(max(int(landmark.y * frame.height), 0), frame.height - 1)
    return x, y


def fade_color(color: Color, strength: float) -> Color:
    clamped = min(max(strength, 0.0), 1.0)
    return tuple(int(channel * clamped) for channel in color)


class VisualEffectsOverlay:
    TRAIL_LENGTH = 18
    PARTICLE_COUNT = 12
    PARTICLE_LIFETIME = 14
    PARTICLE_DISTANCE = 54

    def __init__(self, cv2_module: PreviewCv2Like | None = None) -> None:
        self._cv2_module = cv2_module
        self._index_trail: deque[Point] = deque(maxlen=self.TRAIL_LENGTH)
        self._particles: list[PinchParticle] = []

    def render(
        self,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        pinch_state: PinchState | None,
    ) -> None:
        cv2 = self._load_cv2()
        image = frame.data

        self._update_index_trail(frame, hands, pinch_state)
        self._draw_index_trail(cv2, image)

        if pinch_state is not None and pinch_state.clicked and hands:
            self._spawn_pinch_burst(frame, hands[0].index_tip)

        self._draw_particles(cv2, image)
        self._advance_particles()

    def _update_index_trail(
        self,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        pinch_state: PinchState | None,
    ) -> None:
        if pinch_state is None or not pinch_state.is_pinching or not hands:
            self._index_trail.clear()
            return

        self._index_trail.append(to_pixel(frame, hands[0].index_tip))

    def _draw_index_trail(self, cv2: PreviewCv2Like, image: object) -> None:
        points = list(self._index_trail)
        for index, point in enumerate(points):
            strength = (index + 1) / len(points)
            radius = max(2, int(7 * strength))
            cv2.circle(image, point, radius, fade_color((80, 255, 180), strength), -1)

        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, (60, 180, 255), 2)

    def _spawn_pinch_burst(self, frame: Frame, landmark: Landmark) -> None:
        origin = to_pixel(frame, landmark)
        self._particles.extend(
            PinchParticle(origin=origin, angle=(2 * pi * index) / self.PARTICLE_COUNT)
            for index in range(self.PARTICLE_COUNT)
        )

    def _draw_particles(self, cv2: PreviewCv2Like, image: object) -> None:
        for particle in self._particles:
            progress = particle.age / self.PARTICLE_LIFETIME
            distance = int(self.PARTICLE_DISTANCE * progress)
            x = particle.origin[0] + int(cos(particle.angle) * distance)
            y = particle.origin[1] + int(sin(particle.angle) * distance)
            strength = 1.0 - progress
            color = fade_color((0, 255, 220), strength)
            cv2.circle(image, (x, y), max(2, int(8 * strength)), color, -1)
            cv2.line(image, particle.origin, (x, y), color, 1)

    def _advance_particles(self) -> None:
        self._particles = [
            PinchParticle(
                origin=particle.origin,
                angle=particle.angle,
                age=particle.age + 1,
            )
            for particle in self._particles
            if particle.age + 1 < self.PARTICLE_LIFETIME
        ]

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

        for marker in finger_markers(hand):
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

    def _to_pixel(self, frame: Frame, landmark: Landmark) -> Point:
        return to_pixel(frame, landmark)

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
        effects: VisualEffectsOverlay | None = None,
        cv2_module: PreviewCv2Like | None = None,
    ) -> None:
        self._config = config
        self._frame_source = frame_source
        self._hand_tracker = hand_tracker
        self._gaze_tracker = gaze_tracker
        self._renderer = renderer
        self._effects = effects
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
        effects = self._effects or VisualEffectsOverlay(cv2_module=cv2)
        pinch_detector = PinchClickDetector(
            click_threshold=self._config.pinch_click_threshold,
            release_threshold=self._config.pinch_release_threshold,
        )

        try:
            for frame in frame_source.frames():
                hands = hand_tracker.track(frame)
                gaze_estimate = gaze_tracker.estimate(frame) if gaze_tracker is not None else None
                pinch_state = pinch_detector.update(hands[0]) if hands else None
                display_frame = Frame(
                    width=frame.width,
                    height=frame.height,
                    data=cv2.flip(frame.data, 1),
                )
                effects.render(display_frame, hands, pinch_state)
                image = renderer.render(display_frame, hands, pinch_state, gaze_estimate)
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
