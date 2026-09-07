"""Recorded-video and image-sequence frame sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from qlyraxis.contracts import FramePacket


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class VideoFileSource:
    """Read a video through OpenCV while preserving media timestamps."""

    def __init__(self, path: str | Path, fallback_fps: float = 30.0) -> None:
        self.path = Path(path)
        if fallback_fps <= 0:
            raise ValueError("fallback_fps must be positive")
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"could not open video: {self.path}")
        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if reported_fps > 0 else float(fallback_fps)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._index = 0
        self._last_timestamp_s: float | None = None

    def read(self) -> FramePacket | None:
        success, image = self._capture.read()
        if not success:
            return None
        timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        timestamp_s = (
            timestamp_ms / 1000.0 if timestamp_ms > 0 else self._index / self.fps
        )
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            timestamp_s = self._last_timestamp_s + 1.0 / self.fps
        packet = FramePacket(self._index, timestamp_s, image)
        self._last_timestamp_s = timestamp_s
        self._index += 1
        return packet

    def reset(self) -> None:
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            self._capture.release()
            self._capture = cv2.VideoCapture(str(self.path))
            if not self._capture.isOpened():
                raise ValueError(f"could not reopen video: {self.path}")
        self._index = 0
        self._last_timestamp_s = None

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoFileSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ImageSequenceSource:
    """Read a deterministic, naturally sorted list of still images."""

    def __init__(self, paths: Iterable[str | Path], fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.paths = tuple(sorted((Path(path) for path in paths), key=_natural_key))
        if not self.paths:
            raise ValueError("image sequence is empty")
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"image file not found: {missing[0]}")
        unsupported = [path for path in self.paths if path.suffix.lower() not in IMAGE_EXTENSIONS]
        if unsupported:
            raise ValueError(f"unsupported image type: {unsupported[0].suffix}")
        self.fps = float(fps)
        self.frame_count = len(self.paths)
        sample = cv2.imread(str(self.paths[0]), cv2.IMREAD_UNCHANGED)
        if sample is None:
            raise ValueError(f"could not decode image: {self.paths[0]}")
        self.height, self.width = sample.shape[:2]
        self._index = 0

    @classmethod
    def from_directory(cls, directory: str | Path, fps: float = 30.0):
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"image directory not found: {root}")
        paths = [path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        return cls(paths, fps)

    def read(self) -> FramePacket | None:
        if self._index >= len(self.paths):
            return None
        path = self.paths[self._index]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"could not decode image: {path}")
        packet = FramePacket(self._index, self._index / self.fps, image)
        self._index += 1
        return packet

    def reset(self) -> None:
        self._index = 0

    def close(self) -> None:
        pass

    def __enter__(self) -> "ImageSequenceSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _natural_key(path: Path) -> tuple[tuple[int, object], ...]:
    import re

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


def open_frame_source(path: str | Path, fps: float = 30.0):
    """Open a directory of images, one image, or a recorded video."""

    source_path = Path(path)
    if source_path.is_dir():
        return ImageSequenceSource.from_directory(source_path, fps)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageSequenceSource([source_path], fps)
    return VideoFileSource(source_path, fallback_fps=fps)
