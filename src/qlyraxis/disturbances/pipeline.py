"""Configurable, frame-deterministic virtual camera disturbances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class DisturbanceMetadata:
    camera_jitter_px: Point
    platform_offset_px: Point
    total_shift_px: Point
    beacon_dropped: bool
    atmosphere: str
    atmosphere_strength: float
    turbulence_strength_px: float


@dataclass(frozen=True, slots=True)
class DisturbanceResult:
    image: NDArray[np.uint8]
    transformed_points: tuple[Point | None, ...]
    metadata: DisturbanceMetadata


class DisturbancePipeline:
    """Apply configured disturbances without mutable random-number state."""

    _ATMOSPHERE_DEFAULTS = {
        "clear": 0.0,
        "haze": 0.35,
        "fog": 0.55,
        "rain": 0.45,
        "low_light": 0.65,
    }

    def __init__(self, config: dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self._phase_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 91]))
        self._phases = tuple(float(value) for value in self._phase_rng.uniform(0, math.tau, 8))

    @staticmethod
    def _rng(seed: int, frame_index: int, stream: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([seed, frame_index, stream]))

    def _dropout_active(self, time_s: float) -> bool:
        dropout = self.config.get("dropout", {})
        if not isinstance(dropout, dict) or not dropout.get("enabled", False):
            return False
        start_s = float(dropout.get("start_s", 0.0))
        duration_s = float(dropout.get("duration_s", 0.0))
        return start_s <= time_s < start_s + duration_s

    def _camera_jitter(self, frame_index: int) -> Point:
        maximum = float(self.config.get("camera_jitter_max_px_frame", 0.0))
        if maximum <= 0:
            return 0.0, 0.0
        rng = self._rng(self.seed, frame_index, 1)
        jitter = rng.uniform(-maximum, maximum, size=2)
        return float(jitter[0]), float(jitter[1])

    def _platform_offset(self, frame_index: int, update_hz: float) -> Point:
        amplitude = float(self.config.get("platform_motion_max_px_frame", 0.0))
        motion = str(self.config.get("platform_motion", "none"))
        if amplitude <= 0 or motion == "none":
            return 0.0, 0.0
        time_s = frame_index / update_hz
        p0, p1, p2, p3 = self._phases[:4]
        if motion == "linear":
            return (
                amplitude * math.sin(math.tau * time_s / 4.0 + p0),
                0.35 * amplitude * math.sin(math.tau * time_s / 5.0 + p1),
            )
        if motion == "circular":
            angle = math.tau * time_s / 5.0 + p0
            return amplitude * math.cos(angle), amplitude * math.sin(angle)
        if motion == "figure_eight":
            angle = math.tau * time_s / 5.0 + p0
            return amplitude * math.sin(angle), amplitude * math.sin(2.0 * angle)
        if motion == "spiral":
            period_s = 6.0
            progress = (time_s % period_s) / period_s
            angle = math.tau * 2.0 * progress + p0
            radius = amplitude * progress
            return radius * math.cos(angle), radius * math.sin(angle)
        if motion == "random":
            return (
                amplitude
                * (0.62 * math.sin(1.7 * time_s + p0) + 0.38 * math.sin(3.1 * time_s + p1)),
                amplitude
                * (0.58 * math.sin(1.3 * time_s + p2) + 0.42 * math.sin(2.7 * time_s + p3)),
            )
        raise ValueError(f"unsupported platform motion: {motion}")

    @staticmethod
    def _shift_image(image: NDArray[np.uint8], shift: Point) -> NDArray[np.uint8]:
        if shift == (0.0, 0.0):
            return image.copy()
        matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
        return cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _turbulence(
        self,
        image: NDArray[np.uint8],
        points: Sequence[Point | None],
        frame_index: int,
        strength: float,
    ) -> tuple[NDArray[np.uint8], tuple[Point | None, ...]]:
        if strength <= 0:
            return image, tuple(points)
        height, width = image.shape[:2]
        phase = 0.19 * frame_index + self._phases[4]
        y_axis = np.arange(height, dtype=np.float32)
        x_axis = np.arange(width, dtype=np.float32)
        horizontal_shift = strength * np.sin(y_axis / 21.0 + phase)
        vertical_shift = strength * np.sin(x_axis / 25.0 + 0.83 * phase + self._phases[5])
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)
        map_x = grid_x - horizontal_shift[:, None]
        map_y = grid_y - vertical_shift[None, :]
        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        transformed = []
        for point in points:
            if point is None:
                transformed.append(None)
                continue
            x_px, y_px = point
            row = int(np.clip(round(y_px), 0, height - 1))
            column = int(np.clip(round(x_px), 0, width - 1))
            transformed.append(
                (
                    x_px + float(horizontal_shift[row]),
                    y_px + float(vertical_shift[column]),
                )
            )
        return warped, tuple(transformed)

    @staticmethod
    def _odd_kernel(value: float, maximum: int = 31) -> int:
        rounded = max(1, min(int(round(value)), maximum))
        return rounded if rounded % 2 == 1 else rounded + 1

    def _apply_blur(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        defocus = float(self.config.get("defocus_blur_px", 0.0))
        motion = float(self.config.get("motion_blur_px", 0.0))
        output = image
        if defocus > 0:
            kernel = self._odd_kernel(2.0 * defocus + 1.0)
            output = cv2.GaussianBlur(output, (kernel, kernel), sigmaX=max(defocus / 2.0, 0.1))
        if motion > 0:
            kernel_size = self._odd_kernel(motion)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            output = cv2.filter2D(output, -1, kernel)
        return output

    def _apply_atmosphere(
        self,
        image: NDArray[np.uint8],
        atmosphere: str,
        strength: float,
        frame_index: int,
    ) -> NDArray[np.uint8]:
        if atmosphere == "clear" or strength <= 0:
            return image
        strength = float(np.clip(strength, 0.0, 1.0))
        if atmosphere == "low_light":
            return cv2.convertScaleAbs(image, alpha=1.0 - 0.82 * strength, beta=0)
        if atmosphere == "haze":
            return cv2.convertScaleAbs(
                image,
                alpha=1.0 - 0.48 * strength,
                beta=55.0 * strength,
            )
        if atmosphere == "fog":
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0 + 2.5 * strength)
            return cv2.convertScaleAbs(
                blurred,
                alpha=1.0 - 0.68 * strength,
                beta=145.0 * strength,
            )
        if atmosphere == "rain":
            output = cv2.convertScaleAbs(
                image,
                alpha=1.0 - 0.25 * strength,
                beta=15.0 * strength,
            )
            rng = self._rng(self.seed, frame_index, 2)
            streak_count = int(40 + 100 * strength)
            for _ in range(streak_count):
                x_px = int(rng.integers(0, image.shape[1]))
                y_px = int(rng.integers(-15, image.shape[0]))
                length = int(rng.integers(5, 14))
                level = int(rng.integers(90, 180))
                cv2.line(output, (x_px, y_px), (x_px - 2, y_px + length), level, 1)
            return output
        raise ValueError(f"unsupported atmosphere: {atmosphere}")

    def _apply_noise(self, image: NDArray[np.uint8], frame_index: int) -> NDArray[np.uint8]:
        output = image.astype(np.float32)
        noise_types = self.config.get("noise", [])
        if not isinstance(noise_types, list):
            raise ValueError("disturbance noise must be a list")
        if "gaussian" in noise_types:
            standard_deviation = float(self.config.get("noise_std_px", 0.0))
            rng = self._rng(self.seed, frame_index, 3)
            output += rng.normal(0.0, standard_deviation, output.shape)
        output = np.clip(output, 0, 255)
        if "poisson" in noise_types:
            rng = self._rng(self.seed, frame_index, 4)
            levels = 48.0
            output = rng.poisson(output / 255.0 * levels) / levels * 255.0
        output_u8 = np.clip(output, 0, 255).astype(np.uint8)
        if "salt_pepper" in noise_types:
            fraction = float(self.config.get("salt_pepper_fraction", 0.0))
            if fraction > 0:
                rng = self._rng(self.seed, frame_index, 5)
                count = int(output_u8.size * fraction)
                indices = rng.choice(output_u8.size, count, replace=False)
                split = count // 2
                flat = output_u8.reshape(-1)
                flat[indices[:split]] = 0
                flat[indices[split:]] = 255
        return output_u8

    def apply(
        self,
        image: NDArray[np.uint8],
        points: Sequence[Point | None],
        frame_index: int,
        time_s: float,
        update_hz: float,
    ) -> DisturbanceResult:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("disturbance input must be an 8-bit monochrome image")
        jitter = self._camera_jitter(frame_index)
        platform = self._platform_offset(frame_index, update_hz)
        total_shift = (jitter[0] + platform[0], jitter[1] + platform[1])
        transformed_points = tuple(
            None if point is None else (point[0] + total_shift[0], point[1] + total_shift[1])
            for point in points
        )
        dropped = self._dropout_active(time_s)
        working = np.zeros_like(image) if dropped else self._shift_image(image, total_shift)
        turbulence_strength = float(self.config.get("turbulence_strength_px", 0.0))
        working, transformed_points = self._turbulence(
            working,
            transformed_points,
            frame_index,
            turbulence_strength,
        )
        working = self._apply_blur(working)
        atmosphere = str(self.config.get("atmosphere", "clear"))
        atmosphere_strength = float(
            self.config.get(
                "atmosphere_strength",
                self._ATMOSPHERE_DEFAULTS.get(atmosphere, 0.0),
            )
        )
        working = self._apply_atmosphere(
            working,
            atmosphere,
            atmosphere_strength,
            frame_index,
        )
        working = self._apply_noise(working, frame_index)
        return DisturbanceResult(
            image=working,
            transformed_points=transformed_points,
            metadata=DisturbanceMetadata(
                camera_jitter_px=jitter,
                platform_offset_px=platform,
                total_shift_px=total_shift,
                beacon_dropped=dropped,
                atmosphere=atmosphere,
                atmosphere_strength=atmosphere_strength,
                turbulence_strength_px=turbulence_strength,
            ),
        )

