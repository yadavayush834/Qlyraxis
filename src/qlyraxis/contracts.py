"""Stable data contracts shared by future Qlyraxis modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, Sequence


class TrackingState(StrEnum):
    SEARCH = "search"
    ACQUIRE = "acquire"
    TRACK = "track"
    COAST = "coast"
    REACQUIRE = "reacquire"


@dataclass(frozen=True, slots=True)
class FramePacket:
    index: int
    timestamp_s: float
    image: Any


@dataclass(frozen=True, slots=True)
class Detection:
    x_px: float
    y_px: float
    confidence: float
    width_px: float
    height_px: float


@dataclass(frozen=True, slots=True)
class TrackEstimate:
    x_px: float
    y_px: float
    velocity_x_px_s: float
    velocity_y_px_s: float
    confidence: float
    state: TrackingState


@dataclass(frozen=True, slots=True)
class CameraCommand:
    pan_rate_deg_s: float
    tilt_rate_deg_s: float


class FrameSource(Protocol):
    def read(self) -> FramePacket | None: ...


class Detector(Protocol):
    def detect(self, frame: FramePacket) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def update(
        self, detections: Sequence[Detection], timestamp_s: float
    ) -> TrackEstimate | None: ...


class Controller(Protocol):
    def command(self, estimate: TrackEstimate, timestamp_s: float) -> CameraCommand: ...


class MetricsSink(Protocol):
    def record(
        self,
        frame: FramePacket,
        estimate: TrackEstimate | None,
        command: CameraCommand | None,
        ground_truth: tuple[float, float] | None = None,
    ) -> None: ...
