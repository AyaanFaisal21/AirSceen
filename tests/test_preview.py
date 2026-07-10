from __future__ import annotations

from collections.abc import Iterator, Sequence

from airscreen.config import AirScreenConfig
from airscreen.gestures import PinchState
from airscreen.gaze_calibration import AxisCalibration, GazeCalibrationProfile
from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.vision.camera import Frame
from airscreen.vision.gaze_tracker import GazeEstimate
from airscreen.vision.preview import (
    DebugPreviewRunner,
    FingerOverlayRenderer,
    RedCircleTargetOverlay,
    VisualEffectsOverlay,
)


class FakeCv2:
    COLOR_BGR2RGB = 1
    FONT_HERSHEY_SIMPLEX = 2
    LINE_AA = 3

    def __init__(self) -> None:
        self.rectangles: list[tuple[object, tuple[int, int], tuple[int, int], tuple[int, int, int], int]] = []
        self.circles: list[tuple[object, tuple[int, int], int, tuple[int, int, int], int]] = []
        self.lines: list[tuple[object, tuple[int, int], tuple[int, int], tuple[int, int, int], int]] = []
        self.text: list[str] = []
        self.shown_images: list[object] = []
        self.destroyed = False

    def VideoCapture(self, camera_index: int) -> object:
        raise AssertionError(f"Unexpected camera open for index {camera_index}")

    def cvtColor(self, image: object, code: int) -> object:
        return image

    def flip(self, image: object, flip_code: int) -> object:
        return image

    def rectangle(
        self,
        image: object,
        point_a: tuple[int, int],
        point_b: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> object:
        self.rectangles.append((image, point_a, point_b, color, thickness))
        return image

    def circle(
        self,
        image: object,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        thickness: int,
    ) -> object:
        self.circles.append((image, center, radius, color, thickness))
        return image

    def line(
        self,
        image: object,
        point_a: tuple[int, int],
        point_b: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> object:
        self.lines.append((image, point_a, point_b, color, thickness))
        return image

    def putText(
        self,
        image: object,
        text: str,
        origin: tuple[int, int],
        font_face: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
    ) -> object:
        self.text.append(text)
        return image

    def imshow(self, window_name: str, image: object) -> None:
        self.shown_images.append(image)

    def waitKey(self, delay_ms: int) -> int:
        return ord("q")

    def destroyAllWindows(self) -> None:
        self.destroyed = True


class FakeFrameSource:
    def __init__(self, frames: Sequence[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[Frame]:
        yield from self._frames


class FakeHandTracker:
    def __init__(self, hands: Sequence[HandLandmarks]) -> None:
        self.hands = hands
        self.closed = False

    def track(self, frame: Frame) -> Sequence[HandLandmarks]:
        return self.hands

    def close(self) -> None:
        self.closed = True


class FakeGazeTracker:
    def __init__(self, estimate: GazeEstimate | None) -> None:
        self.estimate_result = estimate
        self.closed = False

    def estimate(self, frame: Frame) -> GazeEstimate | None:
        return self.estimate_result

    def close(self) -> None:
        self.closed = True


class FakeRecorder:
    def __init__(self) -> None:
        self.records: list[tuple[int, Frame, Sequence[HandLandmarks], GazeEstimate | None]] = []
        self.closed = False

    def record(
        self,
        frame_index: int,
        frame: Frame,
        hands: Sequence[HandLandmarks],
        gaze_estimate: GazeEstimate | None = None,
    ) -> None:
        self.records.append((frame_index, frame, hands, gaze_estimate))

    def close(self) -> None:
        self.closed = True


class FakeRedCircleOverlay:
    def __init__(self) -> None:
        self.rendered_frames: list[tuple[Frame, float]] = []

    def render(self, frame: Frame, now_seconds: float) -> None:
        self.rendered_frames.append((frame, now_seconds))


def sample_hand() -> HandLandmarks:
    return HandLandmarks(
        wrist=Landmark(0.1, 0.1),
        thumb_tip=Landmark(0.2, 0.3),
        index_tip=Landmark(0.4, 0.5),
        middle_tip=Landmark(0.5, 0.55),
        ring_tip=Landmark(0.6, 0.6),
        pinky_tip=Landmark(0.7, 0.65),
    )


def test_finger_overlay_draws_fingertip_boxes_and_pinch_indicator() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    renderer = FingerOverlayRenderer(cv2_module=cv2)

    rendered = renderer.render(
        frame,
        [sample_hand()],
        PinchState(is_pinching=True, distance=0.02, clicked=True),
    )

    assert rendered is image
    assert len(cv2.circles) == 5
    assert len(cv2.lines) == 1
    assert cv2.circles[0][1] == (80, 60)
    assert "THUMB" in cv2.text
    assert "INDEX" in cv2.text
    assert "PINCH CLICK" in cv2.text
    assert cv2.rectangles[-1][3] == (0, 220, 0)


def test_finger_overlay_draws_gaze_marker_when_estimate_is_present() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    renderer = FingerOverlayRenderer(cv2_module=cv2)

    renderer.render(
        frame,
        [],
        None,
        GazeEstimate(x=0.25, y=0.75, confidence=0.5),
    )

    assert cv2.circles[0][1] == (75, 150)
    assert "GAZE 0.50" in cv2.text


def test_visual_effects_does_not_draw_index_trail_before_toggle_pinch() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    effects = VisualEffectsOverlay(cv2_module=cv2)

    effects.render(frame, [sample_hand()], None)

    assert cv2.circles == []
    assert cv2.lines == []


def test_visual_effects_toggles_red_index_trail_on_with_pinch_click() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    effects = VisualEffectsOverlay(cv2_module=cv2)

    effects.render(frame, [sample_hand()], PinchState(is_pinching=True, distance=0.02, clicked=True))
    effects.render(frame, [sample_hand()], None)

    assert len(cv2.circles) == 3 + (2 * VisualEffectsOverlay.PARTICLE_COUNT)
    assert len(cv2.lines) == 1 + (2 * VisualEffectsOverlay.PARTICLE_COUNT)
    assert cv2.circles[0][1] == (60, 100)
    assert cv2.circles[0][3] == VisualEffectsOverlay.TRAIL_COLOR
    assert cv2.circles[14][2] == 7


def test_visual_effects_toggles_index_trail_off_with_second_pinch_click() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    effects = VisualEffectsOverlay(cv2_module=cv2)

    effects.render(frame, [sample_hand()], PinchState(is_pinching=True, distance=0.02, clicked=True))
    cv2.circles.clear()
    cv2.lines.clear()
    effects.render(frame, [sample_hand()], PinchState(is_pinching=True, distance=0.02, clicked=True))

    assert len(cv2.circles) == 1 + (2 * VisualEffectsOverlay.PARTICLE_COUNT)
    assert cv2.circles[0][3] == (0, 0, 249)


def test_visual_effects_keeps_fading_index_trail_when_tracking_drops() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    effects = VisualEffectsOverlay(cv2_module=cv2)

    effects.render(frame, [sample_hand()], PinchState(is_pinching=True, distance=0.02, clicked=True))
    cv2.circles.clear()
    cv2.lines.clear()

    effects.render(frame, [], None)

    assert len(cv2.circles) == VisualEffectsOverlay.PARTICLE_COUNT + 1
    assert cv2.circles[0][3] == (0, 0, 249)
    assert len(cv2.lines) == VisualEffectsOverlay.PARTICLE_COUNT


def test_visual_effects_expires_index_trail_after_tracking_stays_lost() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    effects = VisualEffectsOverlay(cv2_module=cv2)

    effects.render(frame, [sample_hand()], PinchState(is_pinching=True, distance=0.02, clicked=True))
    for _ in range(VisualEffectsOverlay.TRAIL_POINT_LIFETIME):
        cv2.circles.clear()
        effects.render(frame, [], None)

    assert len(cv2.circles) <= VisualEffectsOverlay.PARTICLE_COUNT


def test_visual_effects_spawns_pinch_particles_on_click_entry() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=200, data=image)
    effects = VisualEffectsOverlay(cv2_module=cv2)

    effects.render(
        frame,
        [sample_hand()],
        PinchState(is_pinching=True, distance=0.02, clicked=True),
    )

    assert len(cv2.circles) == 1 + VisualEffectsOverlay.PARTICLE_COUNT
    assert len(cv2.lines) == VisualEffectsOverlay.PARTICLE_COUNT
    assert cv2.lines[0][1] == (60, 100)


def test_debug_preview_runner_shows_frame_and_closes_resources() -> None:
    cv2 = FakeCv2()
    image = object()
    frame_source = FakeFrameSource([Frame(width=100, height=100, data=image)])
    hand_tracker = FakeHandTracker([sample_hand()])
    renderer = FingerOverlayRenderer(cv2_module=cv2)
    runner = DebugPreviewRunner(
        AirScreenConfig(debug_preview=True),
        frame_source=frame_source,
        hand_tracker=hand_tracker,
        renderer=renderer,
        cv2_module=cv2,
    )

    result = runner.run()

    assert result == 0
    assert cv2.shown_images == [image]
    assert cv2.destroyed is True
    assert hand_tracker.closed is True


def test_debug_preview_runner_uses_gaze_tracker_when_enabled() -> None:
    cv2 = FakeCv2()
    image = object()
    frame_source = FakeFrameSource([Frame(width=100, height=100, data=image)])
    hand_tracker = FakeHandTracker([])
    gaze_tracker = FakeGazeTracker(GazeEstimate(x=0.5, y=0.5, confidence=0.75))
    renderer = FingerOverlayRenderer(cv2_module=cv2)
    runner = DebugPreviewRunner(
        AirScreenConfig(debug_preview=True, gaze_enabled=True),
        frame_source=frame_source,
        hand_tracker=hand_tracker,
        gaze_tracker=gaze_tracker,
        renderer=renderer,
        cv2_module=cv2,
    )

    result = runner.run()

    assert result == 0
    assert "GAZE 0.75" in cv2.text
    assert gaze_tracker.closed is True


def test_debug_preview_runner_records_landmark_frames() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=100, data=image)
    frame_source = FakeFrameSource([frame])
    hand = sample_hand()
    hand_tracker = FakeHandTracker([hand])
    gaze_estimate = GazeEstimate(x=0.5, y=0.5, confidence=0.75)
    gaze_tracker = FakeGazeTracker(gaze_estimate)
    recorder = FakeRecorder()
    renderer = FingerOverlayRenderer(cv2_module=cv2)
    runner = DebugPreviewRunner(
        AirScreenConfig(debug_preview=True, gaze_enabled=True),
        frame_source=frame_source,
        hand_tracker=hand_tracker,
        gaze_tracker=gaze_tracker,
        recorder=recorder,
        renderer=renderer,
        cv2_module=cv2,
    )

    result = runner.run()

    assert result == 0
    assert recorder.records == [(0, frame, [hand], gaze_estimate)]
    assert recorder.closed is True


def test_debug_preview_runner_applies_gaze_calibration_profile(tmp_path) -> None:
    cv2 = FakeCv2()
    image = object()
    frame_source = FakeFrameSource([Frame(width=100, height=100, data=image)])
    hand_tracker = FakeHandTracker([])
    gaze_tracker = FakeGazeTracker(GazeEstimate(x=0.25, y=0.75, confidence=0.75))
    profile_path = tmp_path / "gaze-profile.json"
    GazeCalibrationProfile(
        x_axis=AxisCalibration(scale=2.0, offset=0.0),
        y_axis=AxisCalibration(scale=0.5, offset=0.0),
    ).save(profile_path)
    renderer = FingerOverlayRenderer(cv2_module=cv2)
    runner = DebugPreviewRunner(
        AirScreenConfig(
            debug_preview=True,
            gaze_enabled=True,
            gaze_calibration_profile_path=profile_path,
        ),
        frame_source=frame_source,
        hand_tracker=hand_tracker,
        gaze_tracker=gaze_tracker,
        renderer=renderer,
        cv2_module=cv2,
    )

    result = runner.run()

    assert result == 0
    assert cv2.circles[0][1] == (50, 37)


def test_red_circle_target_overlay_draws_spawned_targets() -> None:
    cv2 = FakeCv2()
    image = object()
    frame = Frame(width=100, height=100, data=image)
    overlay = RedCircleTargetOverlay(cv2_module=cv2)

    overlay.render(frame, now_seconds=0.0)
    overlay.render(frame, now_seconds=10.0)

    assert cv2.circles[-1][0] is image
    assert cv2.circles[-1][2] == 28
    assert cv2.circles[-1][3] == (0, 0, 255)
    assert cv2.circles[-1][4] == -1


def test_debug_preview_runner_renders_red_circle_overlay_when_enabled() -> None:
    cv2 = FakeCv2()
    image = object()
    frame_source = FakeFrameSource([Frame(width=100, height=100, data=image)])
    hand_tracker = FakeHandTracker([])
    red_circle_overlay = FakeRedCircleOverlay()
    renderer = FingerOverlayRenderer(cv2_module=cv2)
    runner = DebugPreviewRunner(
        AirScreenConfig(debug_preview=True, red_circle_targets_enabled=True),
        frame_source=frame_source,
        hand_tracker=hand_tracker,
        renderer=renderer,
        red_circle_overlay=red_circle_overlay,
        cv2_module=cv2,
    )

    result = runner.run()

    assert result == 0
    assert len(red_circle_overlay.rendered_frames) == 1
