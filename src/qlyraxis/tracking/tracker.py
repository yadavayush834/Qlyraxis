"""Kalman-backed tracking state machine and re-acquisition policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from qlyraxis.contracts import Detection, TrackEstimate, TrackingState
from qlyraxis.tracking.kalman import ConstantVelocityKalman


class SearchScope(StrEnum):
    NONE = "none"
    LOCAL = "local"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class BeaconTrackerConfig:
    acquisition_frames: int = 3
    reacquisition_frames: int = 2
    # At the 7 deg/s survey rate the scene moves about 37 px per 30 Hz frame;
    # leave enough room for the specified ±20 px jitter as well.
    acquisition_radius_px: float = 65.0
    # The gate must tolerate command reversals plus frame jitter while the
    # image-space velocity estimate settles after acquisition.
    tracking_gate_px: float = 160.0
    local_reacquisition_radius_px: float = 180.0
    coast_frames: int = 5
    global_search_after_frames: int = 15
    minimum_confidence: float = 0.45
    adaptive_maneuvers: bool = True
    turn_speed_threshold_px_s: float = 35.0
    velocity_residual_threshold_px_s: float = 180.0

    def __post_init__(self) -> None:
        if self.acquisition_frames < 1 or self.reacquisition_frames < 1:
            raise ValueError("acquisition frame counts must be positive")
        if min(
            self.acquisition_radius_px,
            self.tracking_gate_px,
            self.local_reacquisition_radius_px,
        ) <= 0:
            raise ValueError("association radii must be positive")
        if self.coast_frames < 0 or self.global_search_after_frames < 1:
            raise ValueError("invalid loss-handling frame counts")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.turn_speed_threshold_px_s <= 0:
            raise ValueError("turn speed threshold must be positive")
        if self.velocity_residual_threshold_px_s <= 0:
            raise ValueError("velocity residual threshold must be positive")


class BeaconTracker:
    """Turn unordered detections into a continuous filtered target estimate."""

    def __init__(
        self,
        config: BeaconTrackerConfig | None = None,
        kalman: ConstantVelocityKalman | None = None,
    ) -> None:
        self.config = config or BeaconTrackerConfig()
        self.kalman = kalman or ConstantVelocityKalman()
        self.reset()

    def reset(self) -> None:
        self.state = TrackingState.SEARCH
        self.search_scope = SearchScope.GLOBAL
        self.selected_detection: Detection | None = None
        self.estimate: TrackEstimate | None = None
        self._last_timestamp_s: float | None = None
        self._confirmation_count = 0
        self._missed_frames = 0
        self._reacquisition_age = 0
        self._reacquiring = False
        self._last_confidence = 0.0
        self._last_detection: Detection | None = None
        self._last_detection_timestamp_s: float | None = None
        self._measured_velocity: tuple[float, float] | None = None
        self.turn_detected = False

    def _correct_with_maneuver_detection(
        self,
        detection: Detection,
        timestamp_s: float,
    ) -> None:
        self.turn_detected = False
        measured_velocity = None
        if (
            self._last_detection is not None
            and self._last_detection_timestamp_s is not None
        ):
            dt_s = timestamp_s - self._last_detection_timestamp_s
            if dt_s > 0:
                raw_velocity = (
                    (detection.x_px - self._last_detection.x_px) / dt_s,
                    (detection.y_px - self._last_detection.y_px) / dt_s,
                )
                if self._measured_velocity is None:
                    measured_velocity = raw_velocity
                else:
                    alpha = 0.45
                    measured_velocity = (
                        alpha * raw_velocity[0]
                        + (1.0 - alpha) * self._measured_velocity[0],
                        alpha * raw_velocity[1]
                        + (1.0 - alpha) * self._measured_velocity[1],
                    )
                self._measured_velocity = measured_velocity

        predicted = self.kalman.state
        self.kalman.correct(detection.x_px, detection.y_px)
        if self.config.adaptive_maneuvers and measured_velocity is not None:
            measured_x, measured_y = measured_velocity
            predicted_speed = math.hypot(
                predicted.velocity_x_px_s,
                predicted.velocity_y_px_s,
            )
            measured_speed = math.hypot(measured_x, measured_y)
            dot_product = (
                predicted.velocity_x_px_s * measured_x
                + predicted.velocity_y_px_s * measured_y
            )
            velocity_residual = math.hypot(
                measured_x - predicted.velocity_x_px_s,
                measured_y - predicted.velocity_y_px_s,
            )
            reversed_direction = (
                predicted_speed >= self.config.turn_speed_threshold_px_s
                and measured_speed >= self.config.turn_speed_threshold_px_s
                and dot_product < 0.0
            )
            self.turn_detected = (
                reversed_direction
                or velocity_residual
                >= self.config.velocity_residual_threshold_px_s
            )
            if self.turn_detected:
                self.kalman.blend_velocity(measured_x, measured_y)

        self._last_detection = detection
        self._last_detection_timestamp_s = timestamp_s

    @staticmethod
    def _distance(detection: Detection, x_px: float, y_px: float) -> float:
        return math.hypot(detection.x_px - x_px, detection.y_px - y_px)

    def _valid(self, detections: Sequence[Detection]) -> list[Detection]:
        return [
            detection
            for detection in detections
            if detection.confidence >= self.config.minimum_confidence
        ]

    @staticmethod
    def _best(detections: Sequence[Detection]) -> Detection | None:
        return max(detections, key=lambda item: item.confidence, default=None)

    def _within(
        self,
        detections: Sequence[Detection],
        x_px: float,
        y_px: float,
        radius_px: float,
    ) -> Detection | None:
        nearby = [
            detection
            for detection in detections
            if self._distance(detection, x_px, y_px) <= radius_px
        ]
        return min(
            nearby,
            key=lambda item: self._distance(item, x_px, y_px),
            default=None,
        )

    def _select(self, detections: Sequence[Detection]) -> Detection | None:
        valid = self._valid(detections)
        if not valid:
            return None
        if self.state == TrackingState.SEARCH:
            self.search_scope = SearchScope.GLOBAL
            return self._best(valid)
        if self.state == TrackingState.ACQUIRE and self.selected_detection is not None:
            return self._within(
                valid,
                self.selected_detection.x_px,
                self.selected_detection.y_px,
                self.config.acquisition_radius_px,
            )
        if self.kalman.initialized:
            predicted = self.kalman.state
            if self.state == TrackingState.REACQUIRE:
                if self._reacquisition_age <= self.config.global_search_after_frames:
                    self.search_scope = SearchScope.LOCAL
                    return self._within(
                        valid,
                        predicted.x_px,
                        predicted.y_px,
                        self.config.local_reacquisition_radius_px,
                    )
                self.search_scope = SearchScope.GLOBAL
                return self._best(valid)
            self.search_scope = SearchScope.NONE
            return self._within(
                valid,
                predicted.x_px,
                predicted.y_px,
                self.config.tracking_gate_px,
            )
        return self._best(valid)

    def _as_estimate(self, confidence: float) -> TrackEstimate | None:
        if not self.kalman.initialized:
            return None
        filtered = self.kalman.state
        return TrackEstimate(
            x_px=filtered.x_px,
            y_px=filtered.y_px,
            velocity_x_px_s=filtered.velocity_x_px_s,
            velocity_y_px_s=filtered.velocity_y_px_s,
            confidence=max(0.0, min(confidence, 1.0)),
            state=self.state,
        )

    def update(
        self,
        detections: Sequence[Detection],
        timestamp_s: float,
    ) -> TrackEstimate | None:
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("tracker timestamps must increase")
        if self.kalman.initialized and self._last_timestamp_s is not None:
            self.kalman.predict(timestamp_s - self._last_timestamp_s)
        self._last_timestamp_s = timestamp_s
        if self.state == TrackingState.REACQUIRE:
            self._reacquisition_age += 1

        selected = self._select(detections)
        self.selected_detection = selected
        if selected is not None:
            self._missed_frames = 0
            self._last_confidence = selected.confidence
            if self.state == TrackingState.SEARCH:
                self.kalman.reset(selected.x_px, selected.y_px)
                self.state = TrackingState.ACQUIRE
                self.search_scope = SearchScope.NONE
                self._confirmation_count = 1
                self._reacquiring = False
                self._last_detection = selected
                self._last_detection_timestamp_s = timestamp_s
                self._measured_velocity = None
            elif self.state == TrackingState.REACQUIRE:
                self.kalman.reset(selected.x_px, selected.y_px)
                self.state = TrackingState.ACQUIRE
                self.search_scope = SearchScope.NONE
                self._confirmation_count = 1
                self._reacquiring = True
                self._last_detection = selected
                self._last_detection_timestamp_s = timestamp_s
                self._measured_velocity = None
            else:
                self._correct_with_maneuver_detection(selected, timestamp_s)
                if self.state == TrackingState.ACQUIRE:
                    self._confirmation_count += 1
                    needed = (
                        self.config.reacquisition_frames
                        if self._reacquiring
                        else self.config.acquisition_frames
                    )
                    if self._confirmation_count >= needed:
                        self.state = TrackingState.TRACK
                        self._reacquisition_age = 0
                        self._reacquiring = False
                else:
                    self.state = TrackingState.TRACK
            self.estimate = self._as_estimate(self._last_confidence)
            return self.estimate

        self._missed_frames += 1
        if self.state == TrackingState.ACQUIRE:
            if self._missed_frames > 1:
                self.state = TrackingState.SEARCH
                self.search_scope = SearchScope.GLOBAL
                self._confirmation_count = 0
                self.estimate = None
                return None
        elif self.state == TrackingState.TRACK:
            self.state = TrackingState.COAST
        elif self.state == TrackingState.COAST:
            if self._missed_frames > self.config.coast_frames:
                self.state = TrackingState.REACQUIRE
                self.search_scope = SearchScope.LOCAL
                self._reacquisition_age = 0
        elif self.state == TrackingState.REACQUIRE:
            self.search_scope = (
                SearchScope.LOCAL
                if self._reacquisition_age <= self.config.global_search_after_frames
                else SearchScope.GLOBAL
            )

        confidence = self._last_confidence * (0.75 ** self._missed_frames)
        self.estimate = self._as_estimate(confidence)
        return self.estimate
