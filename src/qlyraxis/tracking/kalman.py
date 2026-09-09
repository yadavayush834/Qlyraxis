"""Small constant-velocity Kalman filter for image-plane target motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KalmanState:
    x_px: float
    y_px: float
    velocity_x_px_s: float
    velocity_y_px_s: float


class ConstantVelocityKalman:
    def __init__(
        self,
        process_variance: float = 80.0,
        measurement_variance: float = 1.5,
        initial_position_variance: float = 16.0,
        initial_velocity_variance: float = 900.0,
    ) -> None:
        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("Kalman variances must be positive")
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.initial_position_variance = initial_position_variance
        self.initial_velocity_variance = initial_velocity_variance
        self._state = np.zeros((4, 1), dtype=np.float64)
        self._covariance = np.eye(4, dtype=np.float64)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def state(self) -> KalmanState:
        if not self._initialized:
            raise RuntimeError("Kalman filter is not initialized")
        return KalmanState(*(float(value) for value in self._state[:, 0]))

    def reset(self, x_px: float, y_px: float) -> KalmanState:
        self._state[:, 0] = (x_px, y_px, 0.0, 0.0)
        self._covariance = np.diag(
            [
                self.initial_position_variance,
                self.initial_position_variance,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
            ]
        ).astype(np.float64)
        self._initialized = True
        return self.state

    def predict(self, dt_s: float) -> KalmanState:
        if not self._initialized:
            raise RuntimeError("Kalman filter is not initialized")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        transition = np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        process_noise = self.process_variance * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process_noise
        return self.state

    def correct(self, x_px: float, y_px: float) -> KalmanState:
        if not self._initialized:
            return self.reset(x_px, y_px)
        observation = np.array([[x_px], [y_px]], dtype=np.float64)
        measurement = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        noise = np.eye(2, dtype=np.float64) * self.measurement_variance
        innovation = observation - measurement @ self._state
        innovation_covariance = measurement @ self._covariance @ measurement.T + noise
        gain = self._covariance @ measurement.T @ np.linalg.inv(innovation_covariance)
        self._state = self._state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        covariance_update = identity - gain @ measurement
        self._covariance = (
            covariance_update @ self._covariance @ covariance_update.T + gain @ noise @ gain.T
        )
        return self.state

    def blend_velocity(
        self,
        velocity_x_px_s: float,
        velocity_y_px_s: float,
        weight: float = 0.65,
    ) -> KalmanState:
        """Pull velocity toward a measured manoeuvre without resetting position."""
        if not self._initialized:
            raise RuntimeError("Kalman filter is not initialized")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("velocity blend weight must be in [0, 1]")
        self._state[2, 0] = (
            (1.0 - weight) * self._state[2, 0] + weight * velocity_x_px_s
        )
        self._state[3, 0] = (
            (1.0 - weight) * self._state[3, 0] + weight * velocity_y_px_s
        )
        self._covariance[2, 2] = max(self._covariance[2, 2], 225.0)
        self._covariance[3, 3] = max(self._covariance[3, 3], 225.0)
        return self.state
