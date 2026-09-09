"""Reusable Phase 4 detection-tracking-control composition."""

from __future__ import annotations

from dataclasses import dataclass

from qlyraxis.ai import OnnxCandidateVerifier, VerifiedBeaconDetector
from qlyraxis.config import Scenario
from qlyraxis.contracts import (
    CameraCommand,
    Detection,
    Detector,
    TrackEstimate,
    TrackingState,
)
from qlyraxis.resources import resource_path
from qlyraxis.simulation import SimulationEngine, SimulationSnapshot
from qlyraxis.tracking.control import PanTiltController, RasterSearchController
from qlyraxis.tracking.tracker import BeaconTracker, BeaconTrackerConfig, SearchScope
from qlyraxis.vision import BeaconDetector


@dataclass(frozen=True, slots=True)
class ClosedLoopStep:
    simulation: SimulationSnapshot
    detections: tuple[Detection, ...]
    estimate: TrackEstimate | None
    next_command: CameraCommand
    state: TrackingState
    search_scope: SearchScope


class ClosedLoopSystem:
    def __init__(
        self,
        engine: SimulationEngine,
        detector: Detector,
        tracker: BeaconTracker,
        controller: PanTiltController,
        search_controller: RasterSearchController,
        detector_backend: str = "classical",
        profile: str = "improved",
    ) -> None:
        self.engine = engine
        self.detector = detector
        self.tracker = tracker
        self.controller = controller
        self.search_controller = search_controller
        self.detector_backend = detector_backend
        self.profile = profile
        self.command = CameraCommand(0.0, 0.0)
        self._using_prediction = False

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        *,
        use_ai: bool = True,
        predictive_control: bool = True,
        adaptive_maneuvers: bool = True,
    ) -> "ClosedLoopSystem":
        engine = SimulationEngine.from_scenario(scenario)
        camera_config = scenario.camera
        detector: Detector = BeaconDetector()
        detector_backend = "classical"
        if use_ai:
            model_path = resource_path("models/beacon_verifier.onnx")
            if model_path.is_file():
                # Use the verifier as soft evidence on simulated disturbances;
                # temporal association remains the final lock decision.
                verifier = OnnxCandidateVerifier(
                    model_path,
                    threshold=0.0,
                    backend="opencv",
                )
                detector = VerifiedBeaconDetector(
                    verifier,
                    detector,
                    ai_weight=0.2,
                    verification_interval_frames=3,
                )
                detector_backend = f"AI verified/{verifier.backend} at 10 Hz"
        controller = PanTiltController(
            viewport_px=tuple(float(value) for value in camera_config["viewport_px"]),
            fov_deg=tuple(float(value) for value in camera_config["fov_deg"]),
            max_pan_speed_deg_s=float(camera_config["max_pan_speed_deg_s"]),
            max_tilt_speed_deg_s=float(camera_config["max_tilt_speed_deg_s"]),
            feedforward_gain=0.80 if predictive_control else 0.0,
        )
        pan_limits, tilt_limits = engine.camera.angular_limits_deg
        search = RasterSearchController(
            max_pan_speed_deg_s=float(camera_config["max_pan_speed_deg_s"]),
            max_tilt_speed_deg_s=float(camera_config["max_tilt_speed_deg_s"]),
            pan_limits_deg=pan_limits,
            tilt_limits_deg=tilt_limits,
        )
        return cls(
            engine=engine,
            detector=detector,
            tracker=BeaconTracker(
                BeaconTrackerConfig(adaptive_maneuvers=adaptive_maneuvers)
            ),
            controller=controller,
            search_controller=search,
            detector_backend=detector_backend,
            profile=(
                "improved"
                if use_ai and predictive_control and adaptive_maneuvers
                else "baseline"
            ),
        )

    def step(self) -> ClosedLoopStep:
        simulation = self.engine.step(self.command)
        detections = self.detector.detect(simulation.frame)
        estimate = self.tracker.update(detections, simulation.frame.timestamp_s)
        use_prediction = estimate is not None and (
            self.tracker.state
            in {TrackingState.ACQUIRE, TrackingState.TRACK, TrackingState.COAST}
            or (
                self.tracker.state == TrackingState.REACQUIRE
                and self.tracker.search_scope == SearchScope.LOCAL
            )
        )
        if use_prediction:
            if not self._using_prediction:
                self.controller.reset()
            self.command = self.controller.command(
                estimate,
                simulation.frame.timestamp_s,
                (
                    simulation.camera_state.pan_rate_deg_s,
                    simulation.camera_state.tilt_rate_deg_s,
                ),
            )
        else:
            if self._using_prediction:
                self.controller.reset()
            self.command = self.search_controller.command(simulation.camera_state)
        self._using_prediction = use_prediction
        return ClosedLoopStep(
            simulation=simulation,
            detections=detections,
            estimate=estimate,
            next_command=self.command,
            state=self.tracker.state,
            search_scope=self.tracker.search_scope,
        )

    def reset(self) -> None:
        self.engine.reset()
        self.tracker.reset()
        self.controller.reset()
        self.search_controller.reset()
        self.command = CameraCommand(0.0, 0.0)
        self._using_prediction = False
