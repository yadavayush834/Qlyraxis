"""Small diagnostic overlays for acquisition verification."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from qlyraxis.contracts import Detection
from qlyraxis.vision.acquisition import AcquisitionResult


def annotate_detections(
    image: NDArray,
    detections: Sequence[Detection],
    acquisition: AcquisitionResult | None = None,
) -> NDArray[np.uint8]:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    selected = acquisition.selected if acquisition is not None else None
    for detection in detections:
        half_width = round(detection.width_px / 2)
        half_height = round(detection.height_px / 2)
        center = (round(detection.x_px), round(detection.y_px))
        is_selected = detection is selected or detection == selected
        color = (0, 255, 0) if is_selected else (0, 190, 255)
        cv2.rectangle(
            canvas,
            (center[0] - half_width, center[1] - half_height),
            (center[0] + half_width, center[1] + half_height),
            color,
            1,
        )
        cv2.drawMarker(
            canvas,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=11,
            thickness=1,
        )
        cv2.putText(
            canvas,
            f"CONF {detection.confidence:.2f}",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    if acquisition is not None:
        cv2.putText(
            canvas,
            f"ACQUISITION: {acquisition.state.upper()}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if acquisition.state == "acquired" else (0, 190, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
