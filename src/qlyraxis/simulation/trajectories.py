"""Mandatory FSOC beacon trajectory models."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

Point = tuple[float, float]
Size = tuple[float, float]


class Trajectory(Protocol):
    def position_at(self, time_s: float) -> Point: ...


def _validate_world(world_size: Size) -> None:
    if world_size[0] <= 0 or world_size[1] <= 0:
        raise ValueError("world dimensions must be positive")


def _wrap(value: float, limit: float) -> float:
    return value % limit


def _reflect(value: float, limit: float) -> float:
    """Reflect any coordinate into [0, limit] without discontinuities."""

    period = 2.0 * limit
    reflected = value % period
    return reflected if reflected <= limit else period - reflected


@dataclass(frozen=True, slots=True)
class StraightLineTrajectory:
    initial: Point
    speed_px_s: float
    heading_deg: float
    world_size: Size

    def __post_init__(self) -> None:
        _validate_world(self.world_size)
        if self.speed_px_s < 0:
            raise ValueError("speed_px_s cannot be negative")

    def position_at(self, time_s: float) -> Point:
        if time_s < 0:
            raise ValueError("time_s cannot be negative")
        angle = math.radians(self.heading_deg)
        x = self.initial[0] + self.speed_px_s * math.cos(angle) * time_s
        y = self.initial[1] + self.speed_px_s * math.sin(angle) * time_s
        return _wrap(x, self.world_size[0]), _wrap(y, self.world_size[1])


@dataclass(frozen=True, slots=True)
class CircularTrajectory:
    center: Point
    radius_px: float
    period_s: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.radius_px <= 0 or self.period_s <= 0:
            raise ValueError("radius_px and period_s must be positive")

    def position_at(self, time_s: float) -> Point:
        if time_s < 0:
            raise ValueError("time_s cannot be negative")
        angle = self.phase_rad + math.tau * time_s / self.period_s
        return (
            self.center[0] + self.radius_px * math.cos(angle),
            self.center[1] + self.radius_px * math.sin(angle),
        )


@dataclass(frozen=True, slots=True)
class FigureEightTrajectory:
    center: Point
    width_px: float
    height_px: float
    period_s: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0 or self.period_s <= 0:
            raise ValueError("width_px, height_px, and period_s must be positive")

    def position_at(self, time_s: float) -> Point:
        if time_s < 0:
            raise ValueError("time_s cannot be negative")
        angle = self.phase_rad + math.tau * time_s / self.period_s
        return (
            self.center[0] + 0.5 * self.width_px * math.sin(angle),
            self.center[1] + 0.5 * self.height_px * math.sin(2.0 * angle),
        )


@dataclass(slots=True)
class RandomTrajectory:
    """Seeded, continuous piecewise-linear random motion with reflected bounds."""

    initial: Point
    max_speed_px_s: float
    turn_interval_s: float
    world_size: Size
    seed: int
    _points: list[Point] = field(init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_world(self.world_size)
        if self.max_speed_px_s <= 0 or self.turn_interval_s <= 0:
            raise ValueError("random trajectory speed and turn interval must be positive")
        self._points = [self.initial]
        self._rng = random.Random(self.seed)

    def _extend_to(self, segment_index: int) -> None:
        while len(self._points) <= segment_index + 1:
            start_x, start_y = self._points[-1]
            angle = self._rng.uniform(0.0, math.tau)
            speed = self._rng.uniform(0.35, 1.0) * self.max_speed_px_s
            distance = speed * self.turn_interval_s
            next_x = _reflect(start_x + math.cos(angle) * distance, self.world_size[0])
            next_y = _reflect(start_y + math.sin(angle) * distance, self.world_size[1])
            self._points.append((next_x, next_y))

    def position_at(self, time_s: float) -> Point:
        if time_s < 0:
            raise ValueError("time_s cannot be negative")
        segment_index = int(time_s // self.turn_interval_s)
        self._extend_to(segment_index)
        segment_start = segment_index * self.turn_interval_s
        alpha = (time_s - segment_start) / self.turn_interval_s
        start = self._points[segment_index]
        end = self._points[segment_index + 1]
        return (
            start[0] + alpha * (end[0] - start[0]),
            start[1] + alpha * (end[1] - start[1]),
        )


def _random_initial(world_size: Size, seed: int, margin_px: float = 50.0) -> Point:
    rng = random.Random(seed)
    margin_x = min(margin_px, world_size[0] / 4.0)
    margin_y = min(margin_px, world_size[1] / 4.0)
    return (
        rng.uniform(margin_x, world_size[0] - margin_x),
        rng.uniform(margin_y, world_size[1] - margin_y),
    )


def _initial_from_config(value: object, world_size: Size, seed: int) -> Point:
    if value == "random":
        return _random_initial(world_size, seed)
    if isinstance(value, list) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("initial_location must be 'random' or an [x, y] pair")


def build_trajectory(
    target_config: dict[str, object],
    world_size: Size,
    seed: int,
) -> Trajectory:
    """Construct a trajectory from a validated scenario target section."""

    motion = target_config["motion"]
    if not isinstance(motion, dict):
        raise ValueError("target.motion must be an object")
    motion_type = motion["type"]
    initial = _initial_from_config(target_config["initial_location"], world_size, seed)

    if motion_type == "straight_line":
        return StraightLineTrajectory(
            initial=initial,
            speed_px_s=float(motion["speed_px_s"]),
            heading_deg=float(motion["heading_deg"]),
            world_size=world_size,
        )

    rng = random.Random(seed)
    phase = rng.uniform(0.0, math.tau)
    center = (world_size[0] / 2.0, world_size[1] / 2.0)
    if motion_type == "circular":
        radius = min(float(motion["radius_px"]), min(world_size) / 2.0)
        return CircularTrajectory(
            center=center,
            radius_px=radius,
            period_s=float(motion["period_s"]),
            phase_rad=phase,
        )
    if motion_type == "figure_eight":
        return FigureEightTrajectory(
            center=center,
            width_px=min(float(motion["width_px"]), world_size[0]),
            height_px=min(float(motion["height_px"]), world_size[1]),
            period_s=float(motion["period_s"]),
            phase_rad=phase,
        )
    if motion_type == "random":
        return RandomTrajectory(
            initial=initial,
            max_speed_px_s=float(motion["max_speed_px_s"]),
            turn_interval_s=float(motion["turn_interval_s"]),
            world_size=world_size,
            seed=seed,
        )
    raise ValueError(f"unsupported trajectory: {motion_type}")

