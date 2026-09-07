"""Monochrome conversion and local contrast enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    median_kernel: int = 3

    def __post_init__(self) -> None:
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if self.clahe_tile_size <= 0:
            raise ValueError("clahe_tile_size must be positive")
        if self.median_kernel < 1 or self.median_kernel % 2 == 0:
            raise ValueError("median_kernel must be an odd positive integer")


@dataclass(frozen=True, slots=True)
class PreprocessedFrame:
    grayscale: NDArray[np.uint8]
    denoised: NDArray[np.uint8]
    enhanced: NDArray[np.uint8]


class FramePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        tile = self.config.clahe_tile_size
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(tile, tile),
        )

    @staticmethod
    def to_grayscale(image: NDArray) -> NDArray[np.uint8]:
        if not isinstance(image, np.ndarray):
            raise TypeError("frame image must be a NumPy array")
        if image.size == 0:
            raise ValueError("frame image cannot be empty")
        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("frame image must be grayscale, BGR, or BGRA")
        if grayscale.dtype == np.uint8:
            return np.ascontiguousarray(grayscale)
        if np.issubdtype(grayscale.dtype, np.floating):
            maximum = float(np.nanmax(grayscale))
            scale = 255.0 if maximum <= 1.0 else 1.0
            return np.clip(np.nan_to_num(grayscale) * scale, 0, 255).astype(np.uint8)
        return np.clip(grayscale, 0, 255).astype(np.uint8)

    def process(self, image: NDArray) -> PreprocessedFrame:
        grayscale = self.to_grayscale(image)
        denoised = (
            cv2.medianBlur(grayscale, self.config.median_kernel)
            if self.config.median_kernel > 1
            else grayscale.copy()
        )
        enhanced = self._clahe.apply(denoised)
        return PreprocessedFrame(
            grayscale=grayscale,
            denoised=denoised,
            enhanced=enhanced,
        )

