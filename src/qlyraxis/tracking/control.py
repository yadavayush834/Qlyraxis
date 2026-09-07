"""PID tracking control and bounded raster search commands."""

from __future__ import annotations

from dataclasses import dataclass

from qlyraxis.contracts import CameraCommand, TrackEstimate
from qlyraxis.simulation.camera import CameraState


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(value, limit))


@dataclass(slots=True)
class _PIDAxis:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    derivative_smoothing: float = 0.7
    integral: float = 0.0
    previous_error: float | None = None
    derivative: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.derivative = 0.0

    def update(self, error: float, dt_s: float) -> float:
        self.integral = _clamp(self.integral + error * dt_s, self.integral_limit)
        raw_derivative = (
            0.0 if self.previous_error is None else (error - self.previous_error) / dt_s
        )
        alpha = self.derivative_smoothing
        self.derivative = alpha * self.derivative + (1.0 - alpha) * raw_derivative
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * self.derivative


class PanTiltController:
    def __init__(
        self,
        viewport_px: tuple[float, float],
        fov_deg: tuple[float, float],
        max_pan_speed_deg_s: float,
        max_tilt_speed_deg_s: float,
        kp: float = 3.0,
        ki: float = 0.45,
        kd: float = 0.08,
        integral_limit_deg_s: float = 3.0,
        deadband_px: float = 0.5,
    ) -> None:
        self.viewport_px = viewport_px
        self.fov_deg = fov_deg
        self.max_pan_speed_deg_s = max_pan_speed_deg_s
        self.max_tilt_speed_deg_s = max_tilt_speed_deg_s
        self.deadband_px = deadband_px
        self._pan = _PIDAxis(kp, ki, kd, integral_limit_deg_s)
        self._tilt = _PIDAxis(kp, ki, kd, integral_limit_deg_s)
        self._last_timestamp_s: float | None = None

    def reset(self) -> None:
        self._pan.reset()
        self._tilt.reset()
        self._last_timestamp_s = None

    def command(self, estimate: TrackEstimate, timestamp_s: float) -> CameraCommand:
        dt_s = (
            1.0 / 30.0
            if self._last_timestamp_s is None
            else timestamp_s - self._last_timestamp_s
        )
        if dt_s <= 0:
            raise ValueError("controller timestamps must increase")
        self._last_timestamp_s = timestamp_s
        center_x = self.viewport_px[0] / 2.0
        center_y = self.viewport_px[1] / 2.0
        error_x_px = estimate.x_px - center_x
        error_y_px = estimate.y_px - center_y
        if abs(error_x_px) <= self.deadband_px:
            error_x_px = 0.0
        if abs(error_y_px) <= self.deadband_px:
            error_y_px = 0.0
        error_pan_deg = error_x_px * self.fov_deg[0] / self.viewport_px[0]
        error_tilt_deg = error_y_px * self.fov_deg[1] / self.viewport_px[1]
        pan_rate = _clamp(self._pan.update(error_pan_deg, dt_s), self.max_pan_speed_deg_s)
        tilt_rate = _clamp(
            -self._tilt.update(error_tilt_deg, dt_s),
            self.max_tilt_speed_deg_s,
        )
        return CameraCommand(pan_rate, tilt_rate)


class RasterSearchController:
    """Sweep the available camera field using overlapping horizontal bands."""

    def __init__(
        self,
        max_pan_speed_deg_s: float,
        max_tilt_speed_deg_s: float,
        pan_limits_deg: tuple[float, float],
        tilt_limits_deg: tuple[float, float],
        vertical_speed_fraction: float = 0.35,
        boundary_margin_deg: float = 0.05,
    ) -> None:
        self.max_pan_speed_deg_s = max_pan_speed_deg_s
        self.max_tilt_speed_deg_s = max_tilt_speed_deg_s
        self.pan_limits_deg = pan_limits_deg
        self.tilt_limits_deg = tilt_limits_deg
        self.vertical_speed_fraction = vertical_speed_fraction
        self.boundary_margin_deg = boundary_margin_deg
        self.reset()

    def reset(self) -> None:
        self._pan_direction = -1.0
        self._tilt_direction = -1.0
        self._initial_horizontal_sweep = True

    def command(self, camera_state: CameraState) -> CameraCommand:
        pan_min, pan_max = self.pan_limits_deg
        tilt_min, tilt_max = self.tilt_limits_deg
        if camera_state.pan_deg >= pan_max - self.boundary_margin_deg:
            self._pan_direction = -1.0
        elif camera_state.pan_deg <= pan_min + self.boundary_margin_deg:
            self._pan_direction = 1.0
            self._initial_horizontal_sweep = False
        if camera_state.tilt_deg >= tilt_max - self.boundary_margin_deg:
            self._tilt_direction = -1.0
        elif camera_state.tilt_deg <= tilt_min + self.boundary_margin_deg:
            self._tilt_direction = 1.0
        return CameraCommand(
            self._pan_direction * self.max_pan_speed_deg_s,
            0.0
            if self._initial_horizontal_sweep
            else self._tilt_direction
            * self.max_tilt_speed_deg_s
            * self.vertical_speed_fraction,
        )
