"""Deterministic synthetic candidate patches for beacon verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PatchDataset:
    images: NDArray[np.float32]
    labels: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1] != 1:
            raise ValueError("dataset images must have shape [N, 1, H, W]")
        if self.labels.shape != (self.images.shape[0],):
            raise ValueError("dataset labels must have shape [N]")

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, images=self.images, labels=self.labels)

    @classmethod
    def load(cls, path: str | Path) -> "PatchDataset":
        with np.load(path) as archive:
            return cls(
                archive["images"].astype(np.float32),
                archive["labels"].astype(np.float32),
            )

    def shuffled(self, seed: int) -> "PatchDataset":
        order = np.random.default_rng(seed).permutation(len(self.labels))
        return PatchDataset(self.images[order], self.labels[order])


def _background(rng: np.random.Generator, size: int) -> NDArray[np.float32]:
    level = float(rng.uniform(0, 95))
    noise = rng.normal(0, rng.uniform(2, 20), (size, size)).astype(np.float32)
    y_axis, x_axis = np.mgrid[:size, :size]
    gradient = rng.uniform(-25, 25) * (x_axis / max(size - 1, 1) - 0.5)
    gradient += rng.uniform(-25, 25) * (y_axis / max(size - 1, 1) - 0.5)
    return np.clip(level + noise + gradient, 0, 255).astype(np.float32)


def _positive_patch(rng: np.random.Generator, size: int) -> NDArray[np.float32]:
    image = _background(rng, size)
    center = (
        int(round(size / 2 + rng.uniform(-2, 2))),
        int(round(size / 2 + rng.uniform(-2, 2))),
    )
    diameter = int(rng.integers(4, 11))
    intensity = float(rng.uniform(175, 255))
    if rng.random() < 0.5:
        cv2.circle(image, center, max(2, diameter // 2), intensity, -1)
    else:
        half = max(2, diameter // 2)
        cv2.rectangle(
            image,
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] + half),
            intensity,
            -1,
        )
    sigma = float(rng.uniform(0, 1.4))
    if sigma > 0.15:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    attenuation = float(rng.uniform(0.65, 1.0))
    haze = float(rng.uniform(0, 45))
    return np.clip(image * attenuation + haze, 0, 255).astype(np.float32)


def _negative_patch(rng: np.random.Generator, size: int) -> NDArray[np.float32]:
    image = _background(rng, size)
    center = (size // 2, size // 2)
    artifact = int(rng.integers(0, 6))
    brightness = float(rng.uniform(160, 255))
    if artifact == 0:
        image[center[1], center[0]] = brightness
    elif artifact == 1:
        angle = float(rng.uniform(0, np.pi))
        length = int(rng.integers(size // 3, size - 3))
        offset = (int(np.cos(angle) * length), int(np.sin(angle) * length))
        cv2.line(
            image,
            (center[0] - offset[0] // 2, center[1] - offset[1] // 2),
            (center[0] + offset[0] // 2, center[1] + offset[1] // 2),
            brightness,
            int(rng.integers(1, 3)),
        )
    elif artifact == 2:
        radius = int(rng.integers(7, 13))
        cv2.circle(image, center, radius, brightness, -1)
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(2.5, 4.5))
    elif artifact == 3:
        axes = (int(rng.integers(8, 14)), int(rng.integers(2, 4)))
        cv2.ellipse(image, center, axes, rng.uniform(0, 180), 0, 360, brightness, -1)
    elif artifact == 4:
        for _ in range(int(rng.integers(3, 9))):
            x_px = int(rng.integers(0, size))
            y_px = int(rng.integers(0, size))
            image[y_px, x_px] = brightness
    else:
        off_center = (
            int(rng.choice([rng.integers(1, 8), rng.integers(size - 8, size - 1)])),
            int(rng.integers(2, size - 2)),
        )
        cv2.circle(image, off_center, int(rng.integers(2, 5)), brightness, -1)
    return np.clip(image, 0, 255).astype(np.float32)


def generate_synthetic_dataset(
    samples_per_class: int = 300,
    patch_size: int = 32,
    seed: int = 26169,
) -> PatchDataset:
    """Generate balanced beacon/non-beacon patches without simulator leakage."""

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if patch_size < 16 or patch_size % 2:
        raise ValueError("patch_size must be an even integer of at least 16")
    rng = np.random.default_rng(seed)
    positives = [_positive_patch(rng, patch_size) for _ in range(samples_per_class)]
    negatives = [_negative_patch(rng, patch_size) for _ in range(samples_per_class)]
    images = np.stack([*positives, *negatives]).astype(np.float32)[:, None] / 255.0
    labels = np.concatenate(
        [
            np.ones(samples_per_class, dtype=np.float32),
            np.zeros(samples_per_class, dtype=np.float32),
        ]
    )
    return PatchDataset(images, labels).shuffled(seed + 1)
