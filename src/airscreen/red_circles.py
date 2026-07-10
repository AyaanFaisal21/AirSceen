from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from math import dist
from typing import Protocol

from airscreen.vision.camera import Frame


Point = tuple[int, int]


class RandomLike(Protocol):
    def uniform(self, a: float, b: float) -> float: ...

    def randint(self, a: int, b: int) -> int: ...


@dataclass(frozen=True, slots=True)
class RedCircleTarget:
    center: Point
    radius: int


class RedCircleTargetSpawner:
    MIN_SPAWN_DELAY_SECONDS = 2.0
    MAX_SPAWN_DELAY_SECONDS = 10.0
    MAX_TARGETS = 4
    TARGET_RADIUS = 28
    EDGE_PADDING = 16

    def __init__(self, random_source: RandomLike | None = None) -> None:
        self._random_source = random_source or Random()
        self._targets: list[RedCircleTarget] = []
        self._next_spawn_at: float | None = None

    @property
    def targets(self) -> Sequence[RedCircleTarget]:
        return tuple(self._targets)

    def update(self, frame: Frame, now_seconds: float) -> Sequence[RedCircleTarget]:
        if self._next_spawn_at is None:
            self._schedule_next_spawn(now_seconds)

        if len(self._targets) >= self.MAX_TARGETS:
            return self.targets

        if self._next_spawn_at is not None and now_seconds >= self._next_spawn_at:
            self._targets.append(self._new_target(frame))
            self._schedule_next_spawn(now_seconds)

        return self.targets

    def pop_at(self, point: Point) -> RedCircleTarget | None:
        hit_targets = [
            (index, target)
            for index, target in enumerate(self._targets)
            if dist(point, target.center) <= target.radius
        ]
        if not hit_targets:
            return None

        target_index, target = min(
            hit_targets,
            key=lambda indexed_target: dist(point, indexed_target[1].center),
        )
        del self._targets[target_index]
        return target

    def _schedule_next_spawn(self, now_seconds: float) -> None:
        delay = self._random_source.uniform(
            self.MIN_SPAWN_DELAY_SECONDS,
            self.MAX_SPAWN_DELAY_SECONDS,
        )
        self._next_spawn_at = now_seconds + delay

    def _new_target(self, frame: Frame) -> RedCircleTarget:
        margin = self.TARGET_RADIUS + self.EDGE_PADDING
        min_x = min(margin, max(frame.width - 1, 0))
        max_x = max(frame.width - margin - 1, min_x)
        min_y = min(margin, max(frame.height - 1, 0))
        max_y = max(frame.height - margin - 1, min_y)
        return RedCircleTarget(
            center=(
                self._random_source.randint(min_x, max_x),
                self._random_source.randint(min_y, max_y),
            ),
            radius=self.TARGET_RADIUS,
        )
