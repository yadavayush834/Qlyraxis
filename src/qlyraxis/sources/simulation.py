"""Adapter exposing the virtual environment through the FrameSource contract."""

from __future__ import annotations

from qlyraxis.contracts import CameraCommand, FramePacket
from qlyraxis.simulation import SimulationEngine, SimulationSnapshot


class SimulationFrameSource:
    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self.last_snapshot: SimulationSnapshot | None = None
        self._command = CameraCommand(0.0, 0.0)

    def set_camera_command(self, command: CameraCommand) -> None:
        self._command = command

    def read(self) -> FramePacket:
        self.last_snapshot = self.engine.step(self._command)
        return self.last_snapshot.frame

    def reset(self) -> None:
        self.engine.reset()
        self.last_snapshot = None
        self._command = CameraCommand(0.0, 0.0)

    def close(self) -> None:
        pass
