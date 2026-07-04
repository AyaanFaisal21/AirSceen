from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Frame:
    width: int
    height: int
    data: object


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]:
        """Yield camera frames until the source is closed."""
