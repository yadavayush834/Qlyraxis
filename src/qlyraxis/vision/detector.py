"""Hybrid small-beacon detector with subpixel centroiding."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from qlyraxis.contracts import Detection, FramePacket
from qlyraxis.vision.preprocess import FramePreprocessor, PreprocessConfig, PreprocessedFrame


@dataclass(frozen=True, slots=True)
class BeaconDetectorConfig:
    min_diameter_px: int = 4
    max_diameter_px: int = 28
    min_area_px: int = 8
    max_area_px: int = 650
    minimum_threshold: int = 32
    mad_scale: float = 7.0
    dog_threshold: int = 28
    minimum_confidence: float = 0.45

    def __post_init__(self) -> None:
        if self.min_diameter_px <= 0 or self.max_diameter_px < self.min_diameter_px:
            raise ValueError("invalid beacon diameter limits")
        if self.min_area_px <= 0 or self.max_area_px < self.min_area_px:
            raise ValueError("invalid beacon area limits")
        if not 0 <= self.minimum_threshold <= 255:
            raise ValueError("minimum_threshold must be in [0, 255]")
        if self.mad_scale <= 0:
            raise ValueError("mad_scale must be positive")
        if not 0 <= self.dog_threshold <= 255:
            raise ValueError("dog_threshold must be in [0, 255]")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DetectionDebug:
    preprocessed: PreprocessedFrame
    intensity_mask: NDArray[np.uint8]
    multiscale_mask: NDArray[np.uint8]
    candidate_mask: NDArray[np.uint8]
    detections: tuple[Detection, ...]
    intensity_threshold: float


class BeaconDetector:
    """Find bright compact targets without simulator state or ground truth."""

    def __init__(
        self,
        config: BeaconDetectorConfig | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self.config = config or BeaconDetectorConfig()
        self.preprocessor = FramePreprocessor(preprocess_config)

    def detect(self, frame: FramePacket) -> tuple[Detection, ...]:
        return self.detect_debug(frame).detections

    def detect_image(self, image: NDArray, index: int = 0) -> tuple[Detection, ...]:
        return self.detect(FramePacket(index=index, timestamp_s=0.0, image=image))

    @staticmethod
    def _robust_threshold(image: NDArray[np.uint8], minimum: int, scale: float) -> float:
        pixels = image.astype(np.float32)
        median = float(np.median(pixels))
        mad = float(np.median(np.abs(pixels - median)))
        robust_sigma = 1.4826 * mad
        percentile = float(np.percentile(pixels, 99.5))
        threshold = max(float(minimum), median + scale * max(robust_sigma, 1.0))
        if percentile > median + 5.0:
            threshold = min(threshold, percentile)
        return min(threshold, 254.0)

    def _candidate_masks(
        self, processed: PreprocessedFrame
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], float]:
        threshold = self._robust_threshold(
            processed.enhanced,
            self.config.minimum_threshold,
            self.config.mad_scale,
        )
        _, intensity_mask = cv2.threshold(
            processed.enhanced, threshold, 255, cv2.THRESH_BINARY
        )

        small_scale = cv2.GaussianBlur(processed.enhanced, (0, 0), 0.8)
        large_scale = cv2.GaussianBlur(processed.enhanced, (0, 0), 2.4)
        dog = cv2.subtract(small_scale, large_scale)
        dog_threshold = self._robust_threshold(
            dog,
            self.config.dog_threshold,
            self.config.mad_scale,
        )
        _, multiscale_mask = cv2.threshold(
            dog,
            dog_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        candidate_mask = cv2.bitwise_or(intensity_mask, multiscale_mask)
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return intensity_mask, multiscale_mask, candidate_mask, threshold

    @staticmethod
    def _local_background(
        image: NDArray[np.uint8], x: int, y: int, width: int, height: int
    ) -> float:
        padding = 4
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + width + padding)
        y1 = min(image.shape[0], y + height + padding)
        neighborhood = image[y0:y1, x0:x1]
        inner_x0, inner_y0 = x - x0, y - y0
        ring_mask = np.ones(neighborhood.shape, dtype=bool)
        ring_mask[
            inner_y0 : inner_y0 + height,
            inner_x0 : inner_x0 + width,
        ] = False
        background = neighborhood[ring_mask]
        return float(np.median(background)) if background.size else 0.0

    def _measure_component(
        self,
        label: int,
        labels: NDArray[np.int32],
        stats: NDArray[np.int32],
        grayscale: NDArray[np.uint8],
    ) -> Detection | None:
        x, y, width, height, area = (int(value) for value in stats[label])
        if not self.config.min_area_px <= area <= self.config.max_area_px:
            return None
        if not (
            self.config.min_diameter_px <= width <= self.config.max_diameter_px
            and self.config.min_diameter_px <= height <= self.config.max_diameter_px
        ):
            return None
        aspect_ratio = width / height
        if not 0.45 <= aspect_ratio <= 2.2:
            return None

        component_mask = labels[y : y + height, x : x + width] == label
        grayscale_roi = grayscale[y : y + height, x : x + width].astype(np.float64)
        background = self._local_background(grayscale, x, y, width, height)
        weights = np.where(component_mask, np.maximum(grayscale_roi - background, 0.0), 0.0)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return None
        rows, columns = np.indices(weights.shape, dtype=np.float64)
        centroid_x = x + float((weights * columns).sum() / total_weight)
        centroid_y = y + float((weights * rows).sum() / total_weight)

        foreground = grayscale_roi[component_mask]
        mean_brightness = float(foreground.mean())
        peak_brightness = float(foreground.max())
        contrast_score = np.clip((mean_brightness - background) / 180.0, 0.0, 1.0)
        peak_score = np.clip((peak_brightness - background) / 220.0, 0.0, 1.0)
        compactness = np.clip(area / float(width * height), 0.0, 1.0)
        size_score = 1.0 if 5 <= width <= 22 and 5 <= height <= 22 else 0.65
        confidence = float(
            0.38 * contrast_score
            + 0.27 * peak_score
            + 0.20 * compactness
            + 0.15 * size_score
        )
        if confidence < self.config.minimum_confidence:
            return None
        return Detection(
            x_px=centroid_x,
            y_px=centroid_y,
            confidence=min(confidence, 1.0),
            width_px=float(width),
            height_px=float(height),
        )

    def detect_debug(self, frame: FramePacket) -> DetectionDebug:
        processed = self.preprocessor.process(frame.image)
        intensity, multiscale, candidates, threshold = self._candidate_masks(processed)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )
        detections = []
        for label in range(1, component_count):
            detection = self._measure_component(
                label,
                labels,
                stats,
                processed.grayscale,
            )
            if detection is not None:
                detections.append(detection)
        detections.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return DetectionDebug(
            preprocessed=processed,
            intensity_mask=intensity,
            multiscale_mask=multiscale,
            candidate_mask=candidates,
            detections=tuple(detections),
            intensity_threshold=threshold,
        )
