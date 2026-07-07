from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class Frame:
    width: int
    height: int
    data: object


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]:
        """Yield camera frames until the source is closed."""


class ImageLike(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, ImageLike | None]: ...

    def release(self) -> None: ...


class OpenCVLike(Protocol):
    def VideoCapture(self, camera_index: int) -> VideoCaptureLike: ...


class CameraOpenError(RuntimeError):
    """Raised when a configured camera cannot be opened."""


class OpenCVCameraSource:
    def __init__(self, camera_index: int = 0, cv2_module: OpenCVLike | None = None) -> None:
        self._camera_index = camera_index
        self._cv2_module = cv2_module

    def frames(self) -> Iterator[Frame]:
        cv2 = self._load_cv2()
        capture = cv2.VideoCapture(self._camera_index)

        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(f"Unable to open camera index {self._camera_index}")

        try:
            while True:
                ok, image = capture.read()
                if not ok or image is None:
                    break

                height, width = image.shape[:2]
                yield Frame(width=int(width), height=int(height), data=image)
        finally:
            capture.release()

    def _load_cv2(self) -> OpenCVLike:
        if self._cv2_module is not None:
            return self._cv2_module

        try:
            cv2 = import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install the vision extras with "
                '`python -m pip install -e ".[vision]"`.'
            ) from exc

        return cast(OpenCVLike, cv2)
