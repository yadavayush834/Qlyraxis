"""Tracking composition for recorded media without simulator ground truth."""

from __future__ import annotations

from dataclasses import dataclass

from qlyraxis.contracts import Detection, Detector, FramePacket, TrackEstimate, TrackingState
from qlyraxis.tracking import BeaconTracker, SearchScope


@dataclass(frozen=True, slots=True)
class RecordedTrackingStep:
    frame: FramePacket
    detections: tuple[Detection, ...]
    estimate: TrackEstimate | None
    state: TrackingState
    search_scope: SearchScope


class RecordedTrackingSystem:
    """Run detector and tracker on any FrameSource packet."""

    def __init__(self, detector: Detector, tracker: BeaconTracker | None = None) -> None:
        self.detector = detector
        self.tracker = tracker or BeaconTracker()

    def step(self, frame: FramePacket) -> RecordedTrackingStep:
        detections = tuple(self.detector.detect(frame))
        estimate = self.tracker.update(detections, frame.timestamp_s)
        return RecordedTrackingStep(
            frame=frame,
            detections=detections,
            estimate=estimate,
            state=self.tracker.state,
            search_scope=self.tracker.search_scope,
        )

    def reset(self) -> None:
        self.tracker.reset()
