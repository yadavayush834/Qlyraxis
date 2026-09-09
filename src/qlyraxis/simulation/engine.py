"""Composition root for the Phase 2 virtual environment."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from qlyraxis.config import Scenario
from qlyraxis.contracts import CameraCommand, FramePacket
from qlyraxis.disturbances import DisturbanceMetadata, DisturbancePipeline
from qlyraxis.simulation.camera import CameraState, VirtualCamera
from qlyraxis.simulation.clock import SimulationClock
from qlyraxis.simulation.renderer import SceneRenderer
from qlyraxis.simulation.trajectories import Point, Trajectory, build_trajectory


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    """Full simulation output; only `frame` may be passed to vision modules."""

    frame: FramePacket
    clean_frame: FramePacket
    camera_state: CameraState
    target_world_positions: tuple[Point, ...]
    target_viewport_positions: tuple[Point | None, ...]
    target_sensor_positions: tuple[Point | None, ...]
    disturbances: DisturbanceMetadata


class SimulationEngine:
    def __init__(
        self,
        clock: SimulationClock,
        camera: VirtualCamera,
        trajectories: list[Trajectory],
        renderer: SceneRenderer,
        disturbances: DisturbancePipeline,
        code_lock_config: dict[str, object] | None = None,
    ) -> None:
        self.clock = clock
        self.camera = camera
        self.trajectories = trajectories
        self.renderer = renderer
        self.disturbances = disturbances
        self.code_lock_config = code_lock_config

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "SimulationEngine":
        camera_config = scenario.camera
        world_size = tuple(float(v) for v in camera_config["world_size_px"])
        viewport = tuple(float(v) for v in camera_config["viewport_px"])
        fov = tuple(float(v) for v in camera_config["fov_deg"])
        camera = VirtualCamera(
            world_size_px=world_size,
            viewport_px=viewport,
            fov_deg=fov,
            initial_center_px=tuple(float(v) for v in camera_config["initial_position_px"]),
            max_pan_speed_deg_s=float(camera_config["max_pan_speed_deg_s"]),
            max_tilt_speed_deg_s=float(camera_config["max_tilt_speed_deg_s"]),
        )
        seed = int(scenario.evaluation["random_seed"])
        trajectories = [
            build_trajectory(
                scenario.target,
                world_size,
                seed + index * 10_007,
                tracking_margin_px=(viewport[0] / 2.0, viewport[1] / 2.0),
            )
            for index in range(int(scenario.target["count"]))
        ]
        renderer = SceneRenderer(
            world_size_px=tuple(int(v) for v in world_size),
            viewport_px=tuple(int(v) for v in viewport),
            target_size_px=tuple(int(v) for v in scenario.target["size_px"]),
            target_shape=str(scenario.target["shape"]),
        )
        return cls(
            clock=SimulationClock(float(camera_config["update_hz"])),
            camera=camera,
            trajectories=trajectories,
            renderer=renderer,
            disturbances=DisturbancePipeline(scenario.disturbances, seed),
            code_lock_config=scenario.target.get("beacon_code"),
        )

    def _beacon_intensities(self, frame_index: int) -> tuple[int, ...] | None:
        if self.code_lock_config is None:
            return None
        pattern = str(self.code_lock_config["pattern"])
        symbol_frames = int(self.code_lock_config["symbol_frames"])
        low = round(float(self.code_lock_config["low_intensity"]))
        decoys = tuple(
            str(value) for value in self.code_lock_config.get("decoy_patterns", [])
        )
        patterns = (pattern,) + tuple(
            decoys[(index - 1) % len(decoys)] if decoys else pattern[::-1]
            for index in range(1, len(self.trajectories))
        )
        symbol = frame_index // symbol_frames
        return tuple(
            255 if candidate[symbol % len(candidate)] == "1" else low
            for candidate in patterns
        )

    def step(self, command: CameraCommand | None = None) -> SimulationSnapshot:
        if self.clock.frame_index > 0:
            self.camera.update(command or CameraCommand(0.0, 0.0), self.clock.dt_s)
        time_s = self.clock.time_s
        world_positions = tuple(
            trajectory.position_at(time_s) for trajectory in self.trajectories
        )
        viewport_positions = tuple(
            self.camera.world_to_viewport(position)
            if self.camera.contains(position)
            else None
            for position in world_positions
        )
        clean_image: NDArray = self.renderer.render_camera(
            world_positions,
            self.camera,
            self._beacon_intensities(self.clock.frame_index),
        )
        disturbed = self.disturbances.apply(
            clean_image,
            viewport_positions,
            frame_index=self.clock.frame_index,
            time_s=time_s,
            update_hz=self.clock.update_hz,
        )
        snapshot = SimulationSnapshot(
            frame=FramePacket(
                index=self.clock.frame_index,
                timestamp_s=time_s,
                image=disturbed.image,
            ),
            clean_frame=FramePacket(
                index=self.clock.frame_index,
                timestamp_s=time_s,
                image=clean_image,
            ),
            camera_state=self.camera.state,
            target_world_positions=world_positions,
            target_viewport_positions=viewport_positions,
            target_sensor_positions=disturbed.transformed_points,
            disturbances=disturbed.metadata,
        )
        self.clock.advance()
        return snapshot

    def overview(self, snapshot: SimulationSnapshot) -> NDArray:
        return self.renderer.render_overview(
            snapshot.target_world_positions,
            self.camera,
        )

    def reset(self) -> None:
        self.clock.reset()
        self.camera.reset()
