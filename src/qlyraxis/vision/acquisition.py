"""Temporal confirmation gate for initial beacon acquisition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from qlyraxis.contracts import Detection


class AcquisitionState(StrEnum):
    SEARCHING = "searching"
    VERIFYING = "verifying"
    ACQUIRED = "acquired"


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    state: AcquisitionState
    selected: Detection | None
    confirmation_count: int
    missed_frames: int


class AcquisitionGate:
    """Confirm a spatially consistent beacon across consecutive frames."""

    def __init__(
        self,
        confirmation_frames: int = 3,
        association_radius_px: float = 18.0,
        minimum_confidence: float = 0.45,
        missed_frame_tolerance: int = 1,
    ) -> None:
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least one")
        if association_radius_px <= 0:
            raise ValueError("association_radius_px must be positive")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if missed_frame_tolerance < 0:
            raise ValueError("missed_frame_tolerance cannot be negative")
        self.confirmation_frames = confirmation_frames
        self.association_radius_px = association_radius_px
        self.minimum_confidence = minimum_confidence
        self.missed_frame_tolerance = missed_frame_tolerance
        self.reset()

    def reset(self) -> AcquisitionResult:
        self._state = AcquisitionState.SEARCHING
        self._selected: Detection | None = None
        self._confirmation_count = 0
        self._missed_frames = 0
        return self.result

    @property
    def result(self) -> AcquisitionResult:
        return AcquisitionResult(
            state=self._state,
            selected=self._selected,
            confirmation_count=self._confirmation_count,
            missed_frames=self._missed_frames,
        )

    def _nearest_valid(self, detections: Sequence[Detection]) -> Detection | None:
        valid = [
            detection
            for detection in detections
            if detection.confidence >= self.minimum_confidence
        ]
        if not valid:
            return None
        if self._selected is None:
            return max(valid, key=lambda detection: detection.confidence)
        nearby = [
            detection
            for detection in valid
            if math.hypot(
                detection.x_px - self._selected.x_px,
                detection.y_px - self._selected.y_px,
            )
            <= self.association_radius_px
        ]
        return max(nearby, key=lambda detection: detection.confidence, default=None)

    def update(self, detections: Sequence[Detection]) -> AcquisitionResult:
        candidate = self._nearest_valid(detections)
        if candidate is None:
            self._missed_frames += 1
            if self._missed_frames > self.missed_frame_tolerance:
                return self.reset()
            return self.result

        self._missed_frames = 0
        if self._selected is None:
            self._confirmation_count = 1
        else:
            self._confirmation_count += 1
        self._selected = candidate
        if self._confirmation_count >= self.confirmation_frames:
            self._state = AcquisitionState.ACQUIRED
        else:
            self._state = AcquisitionState.VERIFYING
        return self.result

