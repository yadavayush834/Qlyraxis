"""Tracking and control diagnostics drawn over the camera feed."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from qlyraxis.contracts import CameraCommand, Detection, TrackEstimate, TrackingState
from qlyraxis.tracking.tracker import SearchScope
from qlyraxis.vision.visualization import annotate_detections


def annotate_tracking(
    image: NDArray,
    detections: Sequence[Detection],
    estimate: TrackEstimate | None,
    state: TrackingState,
    search_scope: SearchScope,
    command: CameraCommand | None,
) -> NDArray[np.uint8]:
    canvas = annotate_detections(image, detections)
    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    cv2.drawMarker(
        canvas,
        center,
        (255, 180, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
    )
    if estimate is not None:
        predicted = (round(estimate.x_px), round(estimate.y_px))
        cv2.circle(canvas, predicted, 9, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, center, predicted, (100, 80, 100), 1, cv2.LINE_AA)
    state_color = (0, 255, 0) if state == TrackingState.TRACK else (0, 190, 255)
    cv2.putText(
        canvas,
        f"STATE: {state.upper()}  SEARCH: {search_scope.upper()}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        state_color,
        1,
        cv2.LINE_AA,
    )
    control_text = (
        "RECORDED INPUT"
        if command is None
        else (
            f"CMD pan {command.pan_rate_deg_s:+.2f}  "
            f"tilt {command.tilt_rate_deg_s:+.2f} deg/s"
        )
    )
    cv2.putText(
        canvas,
        control_text,
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas
