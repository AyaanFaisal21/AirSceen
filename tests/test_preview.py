from __future__ import annotations

from collections.abc import Iterator, Sequence

from airscreen.config import AirScreenConfig
from airscreen.gestures import PinchState
from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.vision.camera import Frame
from airscreen.vision.preview import DebugPreviewRunner, FingerOverlayRenderer


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
    assert cv2.circles[0][1] == (20, 60)
    assert "THUMB" in cv2.text
    assert "INDEX" in cv2.text
    assert "PINCH CLICK" in cv2.text
    assert cv2.rectangles[-1][3] == (0, 220, 0)


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
