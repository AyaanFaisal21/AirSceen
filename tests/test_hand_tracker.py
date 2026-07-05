from __future__ import annotations

from dataclasses import dataclass

from airscreen.vision.camera import Frame
from airscreen.vision.hand_tracker import MediaPipeHandTracker


@dataclass(frozen=True, slots=True)
class FakeLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class FakeMediaPipeHand:
    landmark: list[FakeLandmark]


@dataclass(frozen=True, slots=True)
class FakeResults:
    multi_hand_landmarks: list[FakeMediaPipeHand] | None


class FakeHands:
    def __init__(self, results: FakeResults) -> None:
        self.results = results
        self.processed_image: object | None = None
        self.closed = False

    def process(self, image: object) -> FakeResults:
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


def fake_hand() -> FakeMediaPipeHand:
    return FakeMediaPipeHand(
        landmark=[FakeLandmark(x=index / 100, y=index / 200, z=index / 300) for index in range(21)]
    )


def test_mediapipe_hand_tracker_converts_landmark_indices() -> None:
    frame_data = object()
    frame = Frame(width=640, height=480, data=frame_data)
    cv2 = FakeCv2()
    hands = FakeHands(FakeResults(multi_hand_landmarks=[fake_hand()]))
    tracker = MediaPipeHandTracker(hands=hands, cv2_module=cv2)

    tracked_hands = tracker.track(frame)

    assert cv2.converted_image is frame_data
    assert cv2.convert_code == FakeCv2.COLOR_BGR2RGB
    assert hands.processed_image is cv2.rgb_image
    assert len(tracked_hands) == 1

    hand = tracked_hands[0]
    assert hand.wrist.x == 0.0
    assert hand.thumb_tip.x == 0.04
    assert hand.index_tip.x == 0.08
    assert hand.middle_tip.x == 0.12
    assert hand.ring_tip.x == 0.16
    assert hand.pinky_tip.x == 0.2


def test_mediapipe_hand_tracker_returns_empty_list_when_no_hands_are_detected() -> None:
    tracker = MediaPipeHandTracker(
        hands=FakeHands(FakeResults(multi_hand_landmarks=None)),
        cv2_module=FakeCv2(),
    )

    assert tracker.track(Frame(width=10, height=10, data=object())) == []


def test_mediapipe_hand_tracker_closes_underlying_detector() -> None:
    hands = FakeHands(FakeResults(multi_hand_landmarks=[]))
    tracker = MediaPipeHandTracker(hands=hands, cv2_module=FakeCv2())

    tracker.close()

    assert hands.closed is True
