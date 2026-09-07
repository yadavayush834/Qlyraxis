import math
import unittest

from qlyraxis.config import load_scenario
from qlyraxis.contracts import Detection, TrackEstimate, TrackingState
from qlyraxis.simulation.camera import CameraState
from qlyraxis.tracking import (
    BeaconTracker,
    BeaconTrackerConfig,
    ClosedLoopSystem,
    ConstantVelocityKalman,
    PanTiltController,
    RasterSearchController,
    SearchScope,
)


def detection(x: float, y: float, confidence: float = 0.95) -> Detection:
    return Detection(x, y, confidence, 8, 8)


class KalmanTests(unittest.TestCase):
    def test_constant_velocity_is_estimated(self) -> None:
        kalman = ConstantVelocityKalman()
        kalman.reset(100, 200)
        dt_s = 0.1
        for index in range(1, 101):
            kalman.predict(dt_s)
            kalman.correct(100 + index * 10 * dt_s, 200 - index * 5 * dt_s)
        state = kalman.state
        self.assertAlmostEqual(state.x_px, 200, delta=0.2)
        self.assertAlmostEqual(state.y_px, 150, delta=0.2)
        self.assertAlmostEqual(state.velocity_x_px_s, 10, delta=0.3)
        self.assertAlmostEqual(state.velocity_y_px_s, -5, delta=0.3)

    def test_predict_before_initialization_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            ConstantVelocityKalman().predict(1 / 30)


class TrackerStateMachineTests(unittest.TestCase):
    def acquire(self, tracker: BeaconTracker) -> None:
        tracker.update([detection(100, 100)], 0.0)
        tracker.update([detection(104, 102)], 1 / 30)
        tracker.update([detection(108, 104)], 2 / 30)
        self.assertEqual(tracker.state, TrackingState.TRACK)

    def test_search_acquire_track_sequence(self) -> None:
        tracker = BeaconTracker()
        estimate = tracker.update([detection(100, 100)], 0.0)
        self.assertEqual(tracker.state, TrackingState.ACQUIRE)
        self.assertIsNotNone(estimate)
        tracker.update([detection(104, 102)], 1 / 30)
        tracker.update([detection(108, 104)], 2 / 30)
        self.assertEqual(tracker.state, TrackingState.TRACK)

    def test_track_coast_reacquire_sequence(self) -> None:
        tracker = BeaconTracker(BeaconTrackerConfig(coast_frames=2))
        self.acquire(tracker)
        tracker.update([], 3 / 30)
        self.assertEqual(tracker.state, TrackingState.COAST)
        tracker.update([], 4 / 30)
        tracker.update([], 5 / 30)
        self.assertEqual(tracker.state, TrackingState.REACQUIRE)
        self.assertEqual(tracker.search_scope, SearchScope.LOCAL)

    def test_local_reacquisition_prefers_predicted_target(self) -> None:
        tracker = BeaconTracker(BeaconTrackerConfig(coast_frames=0))
        self.acquire(tracker)
        tracker.update([], 3 / 30)
        tracker.update([], 4 / 30)
        self.assertEqual(tracker.state, TrackingState.REACQUIRE)
        estimate = tracker.estimate
        assert estimate is not None
        near = detection(estimate.x_px + 5, estimate.y_px + 4, 0.8)
        far = detection(400, 400, 0.99)
        tracker.update([far, near], 5 / 30)
        self.assertEqual(tracker.state, TrackingState.ACQUIRE)
        self.assertEqual(tracker.selected_detection, near)

    def test_global_reacquisition_activates_after_local_timeout(self) -> None:
        config = BeaconTrackerConfig(coast_frames=0, global_search_after_frames=1)
        tracker = BeaconTracker(config)
        self.acquire(tracker)
        tracker.update([], 3 / 30)
        tracker.update([], 4 / 30)
        tracker.update([], 5 / 30)
        self.assertEqual(tracker.search_scope, SearchScope.LOCAL)
        remote = detection(500, 400)
        tracker.update([remote], 6 / 30)
        self.assertEqual(tracker.search_scope, SearchScope.NONE)
        self.assertEqual(tracker.selected_detection, remote)
        self.assertEqual(tracker.state, TrackingState.ACQUIRE)

    def test_reacquisition_needs_two_confirmations(self) -> None:
        tracker = BeaconTracker(BeaconTrackerConfig(coast_frames=0))
        self.acquire(tracker)
        tracker.update([], 3 / 30)
        tracker.update([], 4 / 30)
        estimate = tracker.estimate
        assert estimate is not None
        tracker.update([detection(estimate.x_px, estimate.y_px)], 5 / 30)
        self.assertEqual(tracker.state, TrackingState.ACQUIRE)
        tracker.update([detection(estimate.x_px + 1, estimate.y_px)], 6 / 30)
        self.assertEqual(tracker.state, TrackingState.TRACK)


class ControllerTests(unittest.TestCase):
    @staticmethod
    def estimate(x: float, y: float) -> TrackEstimate:
        return TrackEstimate(x, y, 0, 0, 0.9, TrackingState.TRACK)

    def test_pid_command_has_correct_pan_and_tilt_signs(self) -> None:
        controller = PanTiltController((640, 480), (4, 3), 5, 5)
        command = controller.command(self.estimate(420, 340), 0.0)
        self.assertGreater(command.pan_rate_deg_s, 0)
        self.assertLess(command.tilt_rate_deg_s, 0)

    def test_pid_output_is_speed_limited(self) -> None:
        controller = PanTiltController((640, 480), (4, 3), 5, 5)
        command = controller.command(self.estimate(10_000, 10_000), 0.0)
        self.assertLessEqual(abs(command.pan_rate_deg_s), 5)
        self.assertLessEqual(abs(command.tilt_rate_deg_s), 5)

    def test_raster_search_begins_with_horizontal_sweep(self) -> None:
        search = RasterSearchController(5, 5, (-4.25, 4.25), (-4.75, 4.75))
        initial = CameraState((1000, 1000), 0, 0, 0, 0)
        first = search.command(initial)
        self.assertLess(first.pan_rate_deg_s, 0)
        self.assertEqual(first.tilt_rate_deg_s, 0)
        left_edge = CameraState((320, 1000), -4.25, 0, 0, 0)
        second = search.command(left_edge)
        self.assertGreater(second.pan_rate_deg_s, 0)
        self.assertLess(second.tilt_rate_deg_s, 0)


class ClosedLoopIntegrationTests(unittest.TestCase):
    def test_bundled_motion_paths_acquire_within_two_seconds(self) -> None:
        paths = [
            "configs/scenarios/clear_straight.json",
            "configs/scenarios/noisy_circle.json",
            "configs/scenarios/fog_figure_eight.json",
            "configs/scenarios/jitter_random.json",
        ]
        for path in paths:
            with self.subTest(path=path):
                system = ClosedLoopSystem.from_scenario(load_scenario(path))
                acquired_at = None
                for _ in range(75):
                    result = system.step()
                    if result.state == TrackingState.TRACK:
                        acquired_at = result.simulation.frame.timestamp_s
                        break
                self.assertIsNotNone(acquired_at)
                self.assertLessEqual(acquired_at, 2.0)

    def test_clear_closed_loop_retains_lock_and_centroid_accuracy(self) -> None:
        scenario = load_scenario("configs/scenarios/clear_straight.json")
        system = ClosedLoopSystem.from_scenario(scenario)
        acquired = False
        post_acquisition_states = []
        centroid_errors = []
        for _ in range(240):
            result = system.step()
            if result.state == TrackingState.TRACK:
                acquired = True
            if acquired:
                post_acquisition_states.append(result.state)
            truth = result.simulation.target_viewport_positions[0]
            if truth is not None and system.tracker.selected_detection is not None:
                measured = system.tracker.selected_detection
                centroid_errors.append(math.dist((measured.x_px, measured.y_px), truth))
            self.assertLessEqual(
                abs(result.next_command.pan_rate_deg_s),
                float(scenario.camera["max_pan_speed_deg_s"]),
            )
            self.assertLessEqual(
                abs(result.next_command.tilt_rate_deg_s),
                float(scenario.camera["max_tilt_speed_deg_s"]),
            )
        self.assertTrue(acquired)
        self.assertTrue(
            all(
                state in {TrackingState.TRACK, TrackingState.COAST}
                for state in post_acquisition_states
            )
        )
        self.assertLess(max(centroid_errors), 10.0)


if __name__ == "__main__":
    unittest.main()
