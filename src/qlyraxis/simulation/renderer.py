"""Monochrome beacon renderer and color overview renderer."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from qlyraxis.simulation.camera import VirtualCamera

Point = tuple[float, float]


class SceneRenderer:
    def __init__(
        self,
        world_size_px: tuple[int, int],
        viewport_px: tuple[int, int],
        target_size_px: tuple[int, int],
        target_shape: str,
        beacon_intensity: int = 255,
    ) -> None:
        self.world_size_px = world_size_px
        self.viewport_px = viewport_px
        self.target_size_px = target_size_px
        self.target_shape = target_shape
        self.beacon_intensity = int(np.clip(beacon_intensity, 0, 255))

    def _draw_beacon(
        self,
        image: NDArray[np.uint8],
        center: Point,
        color: int | tuple[int, int, int],
    ) -> None:
        center_px = (round(center[0]), round(center[1]))
        width, height = self.target_size_px
        if self.target_shape == "circle":
            radius = max(1, round(min(width, height) / 2.0))
            cv2.circle(image, center_px, radius, color, thickness=-1, lineType=cv2.LINE_AA)
            return
        half_width = width // 2
        half_height = height // 2
        cv2.rectangle(
            image,
            (center_px[0] - half_width, center_px[1] - half_height),
            (center_px[0] + half_width, center_px[1] + half_height),
            color,
            thickness=-1,
        )

    def render_camera(
        self, target_positions: Sequence[Point], camera: VirtualCamera
    ) -> NDArray[np.uint8]:
        width, height = self.viewport_px
        image = np.zeros((height, width), dtype=np.uint8)
        for target_position in target_positions:
            viewport_position = camera.world_to_viewport(target_position)
            if camera.contains(target_position, margin_px=max(self.target_size_px)):
                self._draw_beacon(image, viewport_position, self.beacon_intensity)
        return image

    def render_overview(
        self, target_positions: Sequence[Point], camera: VirtualCamera
    ) -> NDArray[np.uint8]:
        width, height = self.world_size_px
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for target_position in target_positions:
            self._draw_beacon(image, target_position, (255, 255, 255))
        left, top, right, bottom = camera.viewport_bounds
        cv2.rectangle(image, (left, top), (right - 1, bottom - 1), (255, 180, 0), 3)
        center = camera.state.center_world_px
        cv2.drawMarker(
            image,
            (round(center[0]), round(center[1])),
            (0, 220, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )
        return image

