from __future__ import annotations

import pytest

from airscreen.vision.camera import CameraOpenError, OpenCVCameraSource


class FakeImage:
    shape = (480, 640, 3)


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self._reads = [(True, FakeImage()), (False, None)]

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, FakeImage | None]:
        return self._reads.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.camera_index: int | None = None

    def VideoCapture(self, camera_index: int) -> FakeCapture:
        self.camera_index = camera_index
        return self.capture


def test_opencv_camera_source_yields_frames_and_releases_capture() -> None:
    capture = FakeCapture()
    cv2 = FakeCv2(capture)
    source = OpenCVCameraSource(camera_index=2, cv2_module=cv2)

    frames = list(source.frames())

    assert cv2.camera_index == 2
    assert len(frames) == 1
    assert frames[0].width == 640
    assert frames[0].height == 480
    assert frames[0].data is not None
    assert capture.released is True


def test_opencv_camera_source_releases_and_raises_when_camera_cannot_open() -> None:
    capture = FakeCapture(opened=False)
    source = OpenCVCameraSource(camera_index=4, cv2_module=FakeCv2(capture))

    with pytest.raises(CameraOpenError):
        list(source.frames())

    assert capture.released is True
