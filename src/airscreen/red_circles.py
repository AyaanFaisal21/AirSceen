from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import dist
from random import Random
from typing import Protocol

from airscreen.vision.camera import Frame


Point = tuple[int, int]


class RandomLike(Protocol):
    def random(self) -> float: ...

    def uniform(self, a: float, b: float) -> float: ...

    def randint(self, a: int, b: int) -> int: ...


class TargetKind(StrEnum):
    GOOD = "good"
    BAD = "bad"


@dataclass(frozen=True, slots=True)
class RedCircleTarget:
    center: Point
    radius: int
    kind: TargetKind
    expires_at: float | None = None


@dataclass(frozen=True, slots=True)
class IndexFingerSample:
    point: Point
    now_seconds: float


class RedCircleTargetSpawner:
    MIN_SPAWN_DELAY_SECONDS = 2.0
    MAX_SPAWN_DELAY_SECONDS = 10.0
    MAX_TARGETS = 4
    TARGET_RADIUS = 28
    EDGE_PADDING = 16
    MIN_SLICE_SPEED_PIXELS_PER_SECOND = 450.0
    BAD_TARGET_PROBABILITY = 0.25
    BAD_TARGET_LIFETIME_SECONDS = 5.0

    def __init__(self, random_source: RandomLike | None = None) -> None:
        self._random_source = random_source or Random()
        self._targets: list[RedCircleTarget] = []
        self._next_spawn_at: float | None = None
        self._score = 0

    @property
    def targets(self) -> Sequence[RedCircleTarget]:
        return tuple(self._targets)

    @property
    def score(self) -> int:
        return self._score

    def update(self, frame: Frame, now_seconds: float) -> Sequence[RedCircleTarget]:
        self._expire_bad_targets(now_seconds)

        if self._next_spawn_at is None:
            self._schedule_next_spawn(now_seconds)

        if len(self._targets) >= self.MAX_TARGETS:
            return self.targets

        if self._next_spawn_at is not None and now_seconds >= self._next_spawn_at:
            self._targets.append(self._new_target(frame, now_seconds))
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
        self._apply_score(target)
        return target

    def slice_between(
        self,
        start: Point,
        end: Point,
        elapsed_seconds: float,
    ) -> RedCircleTarget | None:
        if elapsed_seconds <= 0:
            return None

        speed = dist(start, end) / elapsed_seconds
        if speed < self.MIN_SLICE_SPEED_PIXELS_PER_SECOND:
            return None

        hit_targets = [
            (index, target)
            for index, target in enumerate(self._targets)
            if segment_slices_circle(start, end, target)
        ]
        if not hit_targets:
            return None

        target_index, target = min(
            hit_targets,
            key=lambda indexed_target: segment_distance_to_point(
                start,
                end,
                indexed_target[1].center,
            ),
        )
        del self._targets[target_index]
        self._apply_score(target)
        return target

    def _schedule_next_spawn(self, now_seconds: float) -> None:
        delay = self._random_source.uniform(
            self.MIN_SPAWN_DELAY_SECONDS,
            self.MAX_SPAWN_DELAY_SECONDS,
        )
        self._next_spawn_at = now_seconds + delay

    def _new_target(self, frame: Frame, now_seconds: float) -> RedCircleTarget:
        margin = self.TARGET_RADIUS + self.EDGE_PADDING
        min_x = min(margin, max(frame.width - 1, 0))
        max_x = max(frame.width - margin - 1, min_x)
        min_y = min(margin, max(frame.height - 1, 0))
        max_y = max(frame.height - margin - 1, min_y)
        kind = self._new_target_kind()
        return RedCircleTarget(
            center=(
                self._random_source.randint(min_x, max_x),
                self._random_source.randint(min_y, max_y),
            ),
            radius=self.TARGET_RADIUS,
            kind=kind,
            expires_at=(
                now_seconds + self.BAD_TARGET_LIFETIME_SECONDS
                if kind == TargetKind.BAD
                else None
            ),
        )

    def _new_target_kind(self) -> TargetKind:
        if self._random_source.random() < self.BAD_TARGET_PROBABILITY:
            return TargetKind.BAD

        return TargetKind.GOOD

    def _expire_bad_targets(self, now_seconds: float) -> None:
        self._targets = [
            target
            for target in self._targets
            if target.expires_at is None or now_seconds < target.expires_at
        ]

    def _apply_score(self, target: RedCircleTarget) -> None:
        self._score += -1 if target.kind == TargetKind.BAD else 1


def segment_slices_circle(start: Point, end: Point, target: RedCircleTarget) -> bool:
    return (
        dist(start, target.center) > target.radius
        and dist(end, target.center) > target.radius
        and segment_distance_to_point(start, end, target.center) <= target.radius
    )


def segment_distance_to_point(start: Point, end: Point, point: Point) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = (segment_x * segment_x) + (segment_y * segment_y)
    if length_squared == 0:
        return dist(start, point)

    projection = (
        ((point[0] - start[0]) * segment_x) + ((point[1] - start[1]) * segment_y)
    ) / length_squared
    clamped_projection = min(max(projection, 0.0), 1.0)
    closest = (
        start[0] + (segment_x * clamped_projection),
        start[1] + (segment_y * clamped_projection),
    )
    return dist(closest, point)
