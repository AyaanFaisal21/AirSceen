from __future__ import annotations

from airscreen.red_circles import RedCircleTargetSpawner
from airscreen.vision.camera import Frame


class FakeRandom:
    def __init__(self) -> None:
        self.uniform_calls: list[tuple[float, float]] = []
        self.randint_calls: list[tuple[int, int]] = []

    def uniform(self, a: float, b: float) -> float:
        self.uniform_calls.append((a, b))
        return a

    def randint(self, a: int, b: int) -> int:
        self.randint_calls.append((a, b))
        return a


def test_red_circle_spawner_waits_at_least_two_seconds_before_first_spawn() -> None:
    random_source = FakeRandom()
    spawner = RedCircleTargetSpawner(random_source=random_source)
    frame = Frame(width=640, height=480, data=object())

    assert spawner.update(frame, now_seconds=0.0) == ()
    assert spawner.update(frame, now_seconds=1.99) == ()
    assert len(spawner.update(frame, now_seconds=2.0)) == 1
    assert random_source.uniform_calls[0] == (2.0, 10.0)


def test_red_circle_spawner_caps_active_targets_at_four() -> None:
    spawner = RedCircleTargetSpawner(random_source=FakeRandom())
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    for now_seconds in (2.0, 4.0, 6.0, 8.0, 10.0, 20.0):
        targets = spawner.update(frame, now_seconds=now_seconds)

    assert len(targets) == 4


def test_red_circle_spawner_keeps_targets_inside_frame_margins() -> None:
    random_source = FakeRandom()
    spawner = RedCircleTargetSpawner(random_source=random_source)
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    targets = spawner.update(frame, now_seconds=2.0)

    assert targets[0].center == (44, 44)
    assert targets[0].radius == 28
    assert random_source.randint_calls == [(44, 595), (44, 435)]


def test_red_circle_spawner_pops_hit_target() -> None:
    spawner = RedCircleTargetSpawner(random_source=FakeRandom())
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    spawner.update(frame, now_seconds=2.0)

    popped = spawner.pop_at((44, 44))

    assert popped is not None
    assert popped.center == (44, 44)
    assert spawner.targets == ()


def test_red_circle_spawner_keeps_targets_when_pop_misses() -> None:
    spawner = RedCircleTargetSpawner(random_source=FakeRandom())
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    spawner.update(frame, now_seconds=2.0)

    assert spawner.pop_at((200, 200)) is None
    assert len(spawner.targets) == 1


def test_red_circle_spawner_slices_target_with_fast_crossing_motion() -> None:
    spawner = RedCircleTargetSpawner(random_source=FakeRandom())
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    spawner.update(frame, now_seconds=2.0)

    sliced = spawner.slice_between((0, 44), (90, 44), elapsed_seconds=0.1)

    assert sliced is not None
    assert sliced.center == (44, 44)
    assert spawner.targets == ()


def test_red_circle_spawner_does_not_slice_with_slow_motion() -> None:
    spawner = RedCircleTargetSpawner(random_source=FakeRandom())
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    spawner.update(frame, now_seconds=2.0)

    assert spawner.slice_between((0, 44), (90, 44), elapsed_seconds=1.0) is None
    assert len(spawner.targets) == 1


def test_red_circle_spawner_does_not_slice_by_slowly_moving_onto_target() -> None:
    spawner = RedCircleTargetSpawner(random_source=FakeRandom())
    frame = Frame(width=640, height=480, data=object())

    spawner.update(frame, now_seconds=0.0)
    spawner.update(frame, now_seconds=2.0)

    assert spawner.slice_between((0, 44), (44, 44), elapsed_seconds=0.05) is None
    assert len(spawner.targets) == 1
