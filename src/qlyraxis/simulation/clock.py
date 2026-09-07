"""Frame-index based simulation time."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationClock:
    """A deterministic clock that avoids accumulated floating-point drift."""

    update_hz: float
    frame_index: int = 0

    def __post_init__(self) -> None:
        if self.update_hz <= 0:
            raise ValueError("update_hz must be positive")
        if self.frame_index < 0:
            raise ValueError("frame_index cannot be negative")

    @property
    def time_s(self) -> float:
        return self.frame_index / self.update_hz

    @property
    def dt_s(self) -> float:
        return 1.0 / self.update_hz

    def advance(self) -> float:
        self.frame_index += 1
        return self.time_s

    def reset(self) -> None:
        self.frame_index = 0

