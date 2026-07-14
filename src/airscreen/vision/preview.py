from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import cos, pi, sin
from time import perf_counter
from typing import Protocol, cast

from airscreen.config import AirScreenConfig
from airscreen.gestures import PinchClickDetector, PinchState
from airscreen.gaze_calibration import GazeCalibrationProfile
from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.recording import LandmarkSessionRecorder
from airscreen.red_circles import IndexFingerSample, RedCircleTarget, RedCircleTargetSpawner, TargetKind
from airscreen.vision.camera import Frame, FrameSource, OpenCVCameraSource, VideoCaptureLike
from airscreen.vision.gaze_tracker import GazeEstimate, GazeTracker, MediaPipeGazeTracker
from airscreen.vision.hand_tracker import HandTracker, MediaPipeHandTracker


Color = tuple[int, int, int]
Point = tuple[int, int]
MouseCallback = Callable[[int, int, int, int, object | None], None]


class PreviewCv2Like(Protocol):
    @property
    def COLOR_BGR2RGB(self) -> int: ...

    @property
    def FONT_HERSHEY_SIMPLEX(self) -> int: ...

    @property
    def LINE_AA(self) -> int: ...

    @property
    def EVENT_LBUTTONDOWN(self) -> int: ...

    def VideoCapture(self, camera_index: int) -> VideoCaptureLike: ...

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

    def namedWindow(self, window_name: str) -> None: ...

    def setMouseCallback(self, window_name: str, callback: MouseCallback) -> None: ...

    def imshow(self, window_name: str, image: object) -> None: ...

    def waitKey(self, delay_ms: int) -> int: ...

    def destroyAllWindows(self) -> None: ...


class LandmarkRecorder(Protocol):
    def record(
        self,
        frame_index: int,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        gaze_estimate: GazeEstimate | None = None,
    ) -> None: ...

    def close(self) -> None: ...


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


@dataclass(frozen=True, slots=True)
class TrailPoint:
    point: Point
    age: int = 0


@dataclass(frozen=True, slots=True)
class HudButton:
    label: str
    top_left: Point
    bottom_right: Point


def point_in_button(point: Point, button: HudButton) -> bool:
    return (
        button.top_left[0] <= point[0] <= button.bottom_right[0]
        and button.top_left[1] <= point[1] <= button.bottom_right[1]
    )


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
    return (
        int(color[0] * clamped),
        int(color[1] * clamped),
        int(color[2] * clamped),
    )


class VisualEffectsOverlay:
    TRAIL_COLOR = (0, 0, 255)
    TRAIL_LENGTH = 64
    TRAIL_POINT_LIFETIME = 48
    PARTICLE_COUNT = 12
    PARTICLE_LIFETIME = 14
    PARTICLE_DISTANCE = 54

    def __init__(self, cv2_module: PreviewCv2Like | None = None) -> None:
        self._cv2_module = cv2_module
        self._index_trail: deque[TrailPoint] = deque(maxlen=self.TRAIL_LENGTH)
        self._particles: list[PinchParticle] = []
        self._trail_enabled = False

    def render(
        self,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        pinch_state: PinchState | None,
    ) -> None:
        cv2 = self._load_cv2()
        image = frame.data

        if pinch_state is not None and pinch_state.clicked:
            self._trail_enabled = not self._trail_enabled

        self._update_index_trail(frame, hands)
        self._draw_index_trail(cv2, image)

        if pinch_state is not None and pinch_state.clicked and hands:
            self._spawn_pinch_burst(frame, hands[0].index_tip)

        self._draw_particles(cv2, image)
        self._advance_particles()

    def _update_index_trail(
        self,
        frame: Frame,
        hands: Sequence[HandLandmarks],
    ) -> None:
        self._age_index_trail()

        if self._trail_enabled and hands:
            self._index_trail.append(TrailPoint(to_pixel(frame, hands[0].index_tip)))

    def _age_index_trail(self) -> None:
        self._index_trail = deque(
            (
                TrailPoint(point=trail_point.point, age=trail_point.age + 1)
                for trail_point in self._index_trail
                if trail_point.age + 1 < self.TRAIL_POINT_LIFETIME
            ),
            maxlen=self.TRAIL_LENGTH,
        )

    def _draw_index_trail(self, cv2: PreviewCv2Like, image: object) -> None:
        trail_points = list(self._index_trail)
        for index, trail_point in enumerate(trail_points):
            order_strength = (index + 1) / len(trail_points)
            age_strength = 1.0 - (trail_point.age / self.TRAIL_POINT_LIFETIME)
            strength = order_strength * age_strength
            radius = max(2, int(7 * strength))
            cv2.circle(
                image,
                trail_point.point,
                radius,
                fade_color(self.TRAIL_COLOR, strength),
                -1,
            )

        for start, end in zip(trail_points, trail_points[1:]):
            line_age = max(start.age, end.age)
            strength = 1.0 - (line_age / self.TRAIL_POINT_LIFETIME)
            cv2.line(image, start.point, end.point, fade_color(self.TRAIL_COLOR, strength), 2)

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
            cv2 = import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(PreviewCv2Like, cv2)
        return self._cv2_module


class FingerOverlayRenderer:
    BOX_RADIUS = 12
    GAZE_CURSOR_COLOR = (255, 120, 20)
    GAZE_CURSOR_OUTLINE_COLOR = (255, 255, 255)

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
        color = self.GAZE_CURSOR_COLOR

        cv2.circle(image, center, 13, self.GAZE_CURSOR_OUTLINE_COLOR, 2)
        cv2.circle(image, center, 9, color, -1)
        cv2.circle(image, (x - 3, y - 3), 3, self.GAZE_CURSOR_OUTLINE_COLOR, -1)
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
            cv2 = import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(PreviewCv2Like, cv2)
        return self._cv2_module


class RedCircleTargetOverlay:
    GOOD_TARGET_COLOR = (0, 220, 0)
    BAD_TARGET_COLOR = (0, 0, 255)
    SCORE_COLOR = (255, 255, 255)

    def __init__(
        self,
        spawner: RedCircleTargetSpawner | None = None,
        cv2_module: PreviewCv2Like | None = None,
    ) -> None:
        self._spawner = spawner or RedCircleTargetSpawner()
        self._cv2_module = cv2_module
        self._previous_index_sample: IndexFingerSample | None = None

    def render(
        self,
        frame: Frame,
        now_seconds: float,
        hands: Sequence[HandLandmarks],
        pinch_state: PinchState | None,
    ) -> None:
        index_point = to_pixel(frame, hands[0].index_tip) if hands else None
        if pinch_state is not None and pinch_state.clicked and index_point is not None:
            self._spawner.pop_at(index_point)

        if index_point is not None and self._previous_index_sample is not None:
            self._spawner.slice_between(
                self._previous_index_sample.point,
                index_point,
                now_seconds - self._previous_index_sample.now_seconds,
            )

        cv2 = self._load_cv2()
        for target in self._spawner.update(frame, now_seconds):
            cv2.circle(frame.data, target.center, target.radius, self._target_color(target), -1)
        self._draw_score(cv2, frame.data)

        self._previous_index_sample = (
            IndexFingerSample(index_point, now_seconds) if index_point is not None else None
        )

    def reset(self) -> None:
        self._spawner.reset()
        self._previous_index_sample = None

    def _target_color(self, target: RedCircleTarget) -> Color:
        if target.kind == TargetKind.BAD:
            return self.BAD_TARGET_COLOR

        return self.GOOD_TARGET_COLOR

    def _draw_score(self, cv2: PreviewCv2Like, image: object) -> None:
        cv2.putText(
            image,
            f"SCORE {self._spawner.score}",
            (12, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            self.SCORE_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _load_cv2(self) -> PreviewCv2Like:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            cv2 = import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(PreviewCv2Like, cv2)
        return self._cv2_module


class DebugPreviewHud:
    PANEL_WIDTH = 320
    BUTTON_HEIGHT = 38
    SETTINGS_BUTTON_WIDTH = 132
    MARGIN = 12
    BACKGROUND_COLOR = (30, 30, 30)
    BUTTON_COLOR = (58, 58, 58)
    ACTIVE_COLOR = (0, 128, 255)
    TEXT_COLOR = (255, 255, 255)

    def __init__(
        self,
        eye_tracking_enabled: bool = False,
        circle_gameplay_enabled: bool = False,
        cv2_module: PreviewCv2Like | None = None,
    ) -> None:
        self.eye_tracking_enabled = eye_tracking_enabled
        self.circle_gameplay_enabled = circle_gameplay_enabled
        self._cv2_module = cv2_module
        self._panel_open = False
        self._reset_requested = False
        self._settings_button: HudButton | None = None
        self._option_buttons: list[HudButton] = []

    def render(self, frame: Frame) -> None:
        cv2 = self._load_cv2()
        image = frame.data
        self._settings_button = self._make_settings_button(frame)
        self._draw_button(cv2, image, self._settings_button, self.BUTTON_COLOR)

        if not self._panel_open:
            self._option_buttons = []
            return

        self._option_buttons = self._make_option_buttons(frame)
        panel_top_left = self._option_buttons[0].top_left
        panel_bottom_right = self._option_buttons[-1].bottom_right
        cv2.rectangle(image, panel_top_left, panel_bottom_right, self.BACKGROUND_COLOR, -1)
        for button in self._option_buttons:
            color = self.ACTIVE_COLOR if self._button_is_active(button) else self.BUTTON_COLOR
            self._draw_button(cv2, image, button, color)

    def handle_mouse_event(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object | None,
    ) -> None:
        cv2 = self._load_cv2()
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.handle_click((x, y))

    def handle_click(self, point: Point) -> None:
        if self._settings_button is not None and point_in_button(point, self._settings_button):
            self._panel_open = not self._panel_open
            return

        if not self._panel_open:
            return

        for button in self._option_buttons:
            if not point_in_button(point, button):
                continue

            if button.label.endswith("EYE TRACKING"):
                self.eye_tracking_enabled = not self.eye_tracking_enabled
            elif button.label.endswith("CIRCLE GAMEPLAY"):
                self.circle_gameplay_enabled = not self.circle_gameplay_enabled
            elif button.label == "RESET":
                self._reset_requested = True
            return

    def consume_reset_requested(self) -> bool:
        if not self._reset_requested:
            return False

        self._reset_requested = False
        return True

    def _make_settings_button(self, frame: Frame) -> HudButton:
        right = frame.width - self.MARGIN
        left = max(self.MARGIN, right - self.SETTINGS_BUTTON_WIDTH)
        top = self.MARGIN
        return HudButton("SETTINGS", (left, top), (right, top + self.BUTTON_HEIGHT))

    def _make_option_buttons(self, frame: Frame) -> list[HudButton]:
        right = frame.width - self.MARGIN
        left = max(self.MARGIN, right - self.PANEL_WIDTH)
        top = self.MARGIN + self.BUTTON_HEIGHT + 8
        labels = [
            f"{'DEACTIVATE' if self.eye_tracking_enabled else 'ACTIVATE'} EYE TRACKING",
            f"{'DEACTIVATE' if self.circle_gameplay_enabled else 'ACTIVATE'} CIRCLE GAMEPLAY",
            "RESET",
        ]
        return [
            HudButton(
                label,
                (left, top + (index * self.BUTTON_HEIGHT)),
                (right, top + ((index + 1) * self.BUTTON_HEIGHT)),
            )
            for index, label in enumerate(labels)
        ]

    def _button_is_active(self, button: HudButton) -> bool:
        if button.label.endswith("EYE TRACKING"):
            return self.eye_tracking_enabled

        if button.label.endswith("CIRCLE GAMEPLAY"):
            return self.circle_gameplay_enabled

        return False

    def _draw_button(
        self,
        cv2: PreviewCv2Like,
        image: object,
        button: HudButton,
        color: Color,
    ) -> None:
        cv2.rectangle(image, button.top_left, button.bottom_right, color, -1)
        cv2.rectangle(image, button.top_left, button.bottom_right, self.TEXT_COLOR, 1)
        cv2.putText(
            image,
            button.label,
            (button.top_left[0] + 10, button.top_left[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    def _load_cv2(self) -> PreviewCv2Like:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            cv2 = import_module("cv2")
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
        red_circle_overlay: RedCircleTargetOverlay | None = None,
        hud: DebugPreviewHud | None = None,
        recorder: LandmarkRecorder | None = None,
        cv2_module: PreviewCv2Like | None = None,
    ) -> None:
        self._config = config
        self._frame_source = frame_source
        self._hand_tracker = hand_tracker
        self._gaze_tracker = gaze_tracker
        self._renderer = renderer
        self._effects = effects
        self._red_circle_overlay = red_circle_overlay
        self._hud = hud
        self._recorder = recorder
        self._cv2_module = cv2_module

    def run(self) -> int:
        cv2 = self._load_cv2()
        cv2.namedWindow(self.WINDOW_NAME)
        frame_source = self._frame_source or OpenCVCameraSource(
            camera_index=self._config.camera_index,
            cv2_module=cv2,
        )
        hand_tracker = self._hand_tracker or MediaPipeHandTracker(cv2_module=cv2)
        gaze_tracker: GazeTracker | None = self._gaze_tracker
        renderer = self._renderer or FingerOverlayRenderer(cv2_module=cv2)
        effects = self._effects or VisualEffectsOverlay(cv2_module=cv2)
        red_circle_overlay = self._red_circle_overlay
        hud = self._hud or DebugPreviewHud(
            eye_tracking_enabled=self._config.gaze_enabled,
            circle_gameplay_enabled=self._config.red_circle_targets_enabled,
            cv2_module=cv2,
        )
        cv2.setMouseCallback(self.WINDOW_NAME, hud.handle_mouse_event)
        recorder = self._load_recorder()
        gaze_profile = self._load_gaze_profile()
        pinch_detector = PinchClickDetector(
            click_threshold=self._config.pinch_click_threshold,
            release_threshold=self._config.pinch_release_threshold,
        )

        try:
            for frame_index, frame in enumerate(frame_source.frames()):
                hands = hand_tracker.track(frame)
                if hud.eye_tracking_enabled and gaze_tracker is None:
                    gaze_tracker = self._load_gaze_tracker(cv2)

                gaze_estimate = (
                    gaze_tracker.estimate(frame)
                    if hud.eye_tracking_enabled and gaze_tracker is not None
                    else None
                )
                if gaze_estimate is not None and gaze_profile is not None:
                    gaze_estimate = gaze_profile.apply(gaze_estimate)

                if recorder is not None:
                    recorder.record(frame_index, frame, hands, gaze_estimate)

                pinch_state = pinch_detector.update(hands[0]) if hands else None
                display_frame = Frame(
                    width=frame.width,
                    height=frame.height,
                    data=cv2.flip(frame.data, 1),
                )
                effects.render(display_frame, hands, pinch_state)
                if hud.circle_gameplay_enabled and red_circle_overlay is None:
                    red_circle_overlay = self._load_red_circle_overlay(cv2)

                if hud.consume_reset_requested() and red_circle_overlay is not None:
                    red_circle_overlay.reset()

                if hud.circle_gameplay_enabled and red_circle_overlay is not None:
                    red_circle_overlay.render(display_frame, perf_counter(), hands, pinch_state)

                image = renderer.render(display_frame, hands, pinch_state, gaze_estimate)
                hud.render(display_frame)
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
            if recorder is not None:
                recorder.close()
            cv2.destroyAllWindows()

        return 0

    def _load_gaze_tracker(self, cv2: PreviewCv2Like) -> GazeTracker | None:
        return self._gaze_tracker or MediaPipeGazeTracker(cv2_module=cv2)

    def _load_recorder(self) -> LandmarkRecorder | None:
        if self._recorder is not None:
            return self._recorder

        if self._config.landmark_record_path is None:
            return None

        return LandmarkSessionRecorder(self._config.landmark_record_path)

    def _load_gaze_profile(self) -> GazeCalibrationProfile | None:
        if self._config.gaze_calibration_profile_path is None:
            return None

        return GazeCalibrationProfile.load(self._config.gaze_calibration_profile_path)

    def _load_red_circle_overlay(
        self,
        cv2: PreviewCv2Like,
    ) -> RedCircleTargetOverlay | None:
        return self._red_circle_overlay or RedCircleTargetOverlay(cv2_module=cv2)

    def _load_cv2(self) -> PreviewCv2Like:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            cv2 = import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        self._cv2_module = cast(PreviewCv2Like, cv2)
        return self._cv2_module
