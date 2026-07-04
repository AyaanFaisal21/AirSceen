from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    x: int
    y: int


class PointerController:
    """Adapter boundary for desktop pointer movement and click automation."""

    def move_to(self, point: ScreenPoint) -> None:
        raise NotImplementedError

    def click(self) -> None:
        raise NotImplementedError


class DryRunPointerController(PointerController):
    def __init__(self) -> None:
        self.moves: list[ScreenPoint] = []
        self.clicks = 0

    def move_to(self, point: ScreenPoint) -> None:
        self.moves.append(point)

    def click(self) -> None:
        self.clicks += 1
