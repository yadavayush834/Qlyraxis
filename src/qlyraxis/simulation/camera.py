"""Constrained virtual pan-tilt camera model."""

from __future__ import annotations

from dataclasses import dataclass

from qlyraxis.contracts import CameraCommand

Point = tuple[float, float]
Size = tuple[float, float]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _approach(current: float, target: float, maximum_change: float) -> float:
    return current + _clamp(target - current, -maximum_change, maximum_change)


@dataclass(frozen=True, slots=True)
class CameraState:
    center_world_px: Point
    pan_deg: float
    tilt_deg: float
    pan_rate_deg_s: float
    tilt_rate_deg_s: float


class VirtualCamera:
    """A viewport whose angular rates and accelerations are physically limited."""

    def __init__(
        self,
        world_size_px: Size,
        viewport_px: Size,
        fov_deg: Size,
        initial_center_px: Point,
        max_pan_speed_deg_s: float,
        max_tilt_speed_deg_s: float,
        max_pan_accel_deg_s2: float = 30.0,
        max_tilt_accel_deg_s2: float = 30.0,
    ) -> None:
        self.world_size_px = world_size_px
        self.viewport_px = viewport_px
        self.fov_deg = fov_deg
        self.max_pan_speed_deg_s = max_pan_speed_deg_s
        self.max_tilt_speed_deg_s = max_tilt_speed_deg_s
        self.max_pan_accel_deg_s2 = max_pan_accel_deg_s2
        self.max_tilt_accel_deg_s2 = max_tilt_accel_deg_s2
        self._initial_center_px = self._bounded_center(initial_center_px)
        self._center_x, self._center_y = self._initial_center_px
        self._pan_rate = 0.0
        self._tilt_rate = 0.0

    @property
    def px_per_degree(self) -> Size:
        return (
            self.viewport_px[0] / self.fov_deg[0],
            self.viewport_px[1] / self.fov_deg[1],
        )

    @property
    def state(self) -> CameraState:
        pixels_per_pan_deg, pixels_per_tilt_deg = self.px_per_degree
        world_center_x = self.world_size_px[0] / 2.0
        world_center_y = self.world_size_px[1] / 2.0
        return CameraState(
            center_world_px=(self._center_x, self._center_y),
            pan_deg=(self._center_x - world_center_x) / pixels_per_pan_deg,
            tilt_deg=(world_center_y - self._center_y) / pixels_per_tilt_deg,
            pan_rate_deg_s=self._pan_rate,
            tilt_rate_deg_s=self._tilt_rate,
        )

    @property
    def viewport_bounds(self) -> tuple[int, int, int, int]:
        half_width = self.viewport_px[0] / 2.0
        half_height = self.viewport_px[1] / 2.0
        left = round(self._center_x - half_width)
        top = round(self._center_y - half_height)
        return (
            int(left),
            int(top),
            int(left + self.viewport_px[0]),
            int(top + self.viewport_px[1]),
        )

    def _bounded_center(self, center: Point) -> Point:
        half_width = self.viewport_px[0] / 2.0
        half_height = self.viewport_px[1] / 2.0
        return (
            _clamp(center[0], half_width, self.world_size_px[0] - half_width),
            _clamp(center[1], half_height, self.world_size_px[1] - half_height),
        )

    def world_to_viewport(self, point: Point) -> Point:
        left, top, _, _ = self.viewport_bounds
        return point[0] - left, point[1] - top

    def contains(self, point: Point, margin_px: float = 0.0) -> bool:
        x, y = self.world_to_viewport(point)
        return (
            -margin_px <= x < self.viewport_px[0] + margin_px
            and -margin_px <= y < self.viewport_px[1] + margin_px
        )

    def update(self, command: CameraCommand, dt_s: float) -> CameraState:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        desired_pan = _clamp(
            command.pan_rate_deg_s,
            -self.max_pan_speed_deg_s,
            self.max_pan_speed_deg_s,
        )
        desired_tilt = _clamp(
            command.tilt_rate_deg_s,
            -self.max_tilt_speed_deg_s,
            self.max_tilt_speed_deg_s,
        )
        self._pan_rate = _approach(
            self._pan_rate, desired_pan, self.max_pan_accel_deg_s2 * dt_s
        )
        self._tilt_rate = _approach(
            self._tilt_rate, desired_tilt, self.max_tilt_accel_deg_s2 * dt_s
        )

        pixels_per_pan_deg, pixels_per_tilt_deg = self.px_per_degree
        requested = (
            self._center_x + self._pan_rate * pixels_per_pan_deg * dt_s,
            self._center_y - self._tilt_rate * pixels_per_tilt_deg * dt_s,
        )
        bounded_x, bounded_y = self._bounded_center(requested)
        if bounded_x != requested[0]:
            self._pan_rate = 0.0
        if bounded_y != requested[1]:
            self._tilt_rate = 0.0
        self._center_x, self._center_y = bounded_x, bounded_y
        return self.state

    def reset(self) -> CameraState:
        self._center_x, self._center_y = self._initial_center_px
        self._pan_rate = 0.0
        self._tilt_rate = 0.0
        return self.state

