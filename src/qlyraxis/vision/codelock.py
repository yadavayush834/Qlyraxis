"""Temporal optical-code identity verification for FSOC beacon candidates."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from qlyraxis.contracts import Detection, Detector, FramePacket


@dataclass(slots=True)
class _CandidateHistory:
    track_id: int
    x_px: float
    y_px: float
    last_frame: int
    samples: deque[tuple[int, float]] = field(default_factory=deque)
    correlation: float | None = None


class CodeLockDetector:
    """Allow only candidates carrying the configured repeating intensity code."""

    def __init__(
        self,
        detector: Detector,
        pattern: str,
        identity: str,
        *,
        symbol_frames: int = 1,
        minimum_correlation: float = 0.70,
        association_radius_px: float = 100.0,
        history_cycles: int = 2,
    ) -> None:
        if len(pattern) < 7 or set(pattern) != {"0", "1"}:
            raise ValueError("CodeLock pattern must contain both 0 and 1 and be at least 7 bits")
        if not identity.strip():
            raise ValueError("CodeLock identity cannot be empty")
        if symbol_frames < 1:
            raise ValueError("CodeLock symbol_frames must be positive")
        if not 0.0 < minimum_correlation <= 1.0:
            raise ValueError("CodeLock correlation threshold must be in (0, 1]")
        if association_radius_px <= 0 or history_cycles < 1:
            raise ValueError("CodeLock association radius and history cycles must be positive")
        self.detector = detector
        self.pattern = pattern
        self.identity = identity
        self.symbol_frames = symbol_frames
        self.minimum_correlation = minimum_correlation
        self.association_radius_px = association_radius_px
        self._expanded_period = tuple(
            int(bit) for bit in pattern for _ in range(symbol_frames)
        )
        self._required_samples = len(self._expanded_period)
        self._history_limit = self._required_samples * history_cycles
        self._histories: list[_CandidateHistory] = []
        self._next_track_id = 1
        self.identity_status = "SEARCHING"
        self.best_correlation: float | None = None
        self.sample_count = 0
        self.verified_track_id: int | None = None

    @property
    def required_samples(self) -> int:
        return self._required_samples

    def reset(self) -> None:
        self._histories.clear()
        self._next_track_id = 1
        self.identity_status = "SEARCHING"
        self.best_correlation = None
        self.sample_count = 0
        self.verified_track_id = None

    def detect(self, frame: FramePacket) -> tuple[Detection, ...]:
        detections = tuple(self.detector.detect(frame))
        self._histories = [
            history
            for history in self._histories
            if frame.index - history.last_frame <= self._required_samples
        ]
        unused = set(range(len(self._histories)))
        matched: list[tuple[Detection, _CandidateHistory]] = []
        for detection in detections:
            history_index = self._nearest_history(detection, unused)
            if history_index is None:
                history = _CandidateHistory(
                    self._next_track_id,
                    detection.x_px,
                    detection.y_px,
                    frame.index,
                    deque(maxlen=self._history_limit),
                )
                self._next_track_id += 1
                self._histories.append(history)
            else:
                unused.remove(history_index)
                history = self._histories[history_index]
            history.x_px = detection.x_px
            history.y_px = detection.y_px
            history.last_frame = frame.index
            history.samples.append(
                (frame.index, self._candidate_signal(frame.image, detection))
            )
            history.correlation = self._correlation(history.samples)
            matched.append((detection, history))

        ranked = sorted(
            (history for _, history in matched),
            key=lambda history: (
                history.correlation if history.correlation is not None else -1.0,
                len(history.samples),
            ),
            reverse=True,
        )
        best = ranked[0] if ranked else None
        self.best_correlation = None if best is None else best.correlation
        self.sample_count = 0 if best is None else len(best.samples)
        verified = [
            (detection, history)
            for detection, history in matched
            if len(history.samples) >= self._required_samples
            and history.correlation is not None
            and history.correlation >= self.minimum_correlation
        ]
        if verified:
            self.identity_status = "VERIFIED"
            self.verified_track_id = max(
                verified,
                key=lambda item: item[1].correlation or -1.0,
            )[1].track_id
        elif best is None:
            self.identity_status = "SEARCHING"
            self.verified_track_id = None
        elif len(best.samples) < self._required_samples:
            self.identity_status = "COLLECTING"
            self.verified_track_id = None
        else:
            self.identity_status = "REJECTED"
            self.verified_track_id = None

        return tuple(
            Detection(
                detection.x_px,
                detection.y_px,
                min(
                    1.0,
                    0.55 * detection.confidence
                    + 0.45 * max(history.correlation or 0.0, 0.0),
                ),
                detection.width_px,
                detection.height_px,
            )
            for detection, history in sorted(
                verified,
                key=lambda item: item[1].correlation or -1.0,
                reverse=True,
            )
        )

    def _nearest_history(
        self, detection: Detection, candidates: set[int]
    ) -> int | None:
        eligible = [
            (
                math.hypot(
                    detection.x_px - self._histories[index].x_px,
                    detection.y_px - self._histories[index].y_px,
                ),
                index,
            )
            for index in candidates
        ]
        distance, index = min(eligible, default=(math.inf, -1))
        return index if distance <= self.association_radius_px else None

    def _correlation(
        self, samples: Sequence[tuple[int, float]]
    ) -> float | None:
        if len(samples) < self._required_samples:
            return None
        recent = tuple(samples)[-self._history_limit :]
        observed = np.asarray([sample[1] for sample in recent], dtype=np.float64)
        observed -= observed.mean()
        observed_norm = float(np.linalg.norm(observed))
        if observed_norm < 1e-6:
            return None
        best = -1.0
        period = len(self._expanded_period)
        for phase in range(period):
            expected = np.asarray(
                [self._expanded_period[(frame_index + phase) % period] for frame_index, _ in recent],
                dtype=np.float64,
            )
            expected -= expected.mean()
            denominator = observed_norm * float(np.linalg.norm(expected))
            if denominator > 1e-6:
                best = max(best, float(np.dot(observed, expected) / denominator))
        return max(-1.0, min(best, 1.0))

    @staticmethod
    def _candidate_signal(image, detection: Detection) -> float:
        if image.ndim == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            grayscale = image
        else:
            raise ValueError("CodeLock input must be grayscale or BGR")
        center_x = round(detection.x_px)
        center_y = round(detection.y_px)
        radius = max(2, min(7, round(max(detection.width_px, detection.height_px) / 2)))
        padding = radius + 4
        y0, y1 = max(0, center_y - padding), min(grayscale.shape[0], center_y + padding + 1)
        x0, x1 = max(0, center_x - padding), min(grayscale.shape[1], center_x + padding + 1)
        patch = grayscale[y0:y1, x0:x1].astype(np.float32)
        inner_y0, inner_y1 = max(0, center_y - radius - y0), min(patch.shape[0], center_y + radius + 1 - y0)
        inner_x0, inner_x1 = max(0, center_x - radius - x0), min(patch.shape[1], center_x + radius + 1 - x0)
        inner = patch[inner_y0:inner_y1, inner_x0:inner_x1]
        ring_mask = np.ones(patch.shape, dtype=bool)
        ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
        background = patch[ring_mask]
        background_level = float(np.median(background)) if background.size else 0.0
        return max(0.0, float(np.percentile(inner, 85)) - background_level)
