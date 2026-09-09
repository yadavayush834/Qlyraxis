import math
import unittest

import numpy as np

from qlyraxis.config import load_scenario
from qlyraxis.contracts import CameraCommand
from qlyraxis.simulation.camera import VirtualCamera
from qlyraxis.simulation.clock import SimulationClock
from qlyraxis.simulation.engine import SimulationEngine
from qlyraxis.simulation.renderer import SceneRenderer
from qlyraxis.simulation.trajectories import (
    CircularTrajectory,
    FigureEightTrajectory,
    RandomTrajectory,
    StraightLineTrajectory,
)


class SimulationClockTests(unittest.TestCase):
    def test_time_is_derived_from_frame_index(self) -> None:
        clock = SimulationClock(30)
        for _ in range(300):
            clock.advance()
        self.assertEqual(clock.frame_index, 300)
        self.assertEqual(clock.time_s, 10.0)


class TrajectoryTests(unittest.TestCase):
    def test_straight_line_reflects_continuously_at_world_edge(self) -> None:
        trajectory = StraightLineTrajectory((90, 50), 20, 0, (100, 100))
        self.assertEqual(trajectory.position_at(0.5), (100.0, 50.0))
        self.assertEqual(trajectory.position_at(1.0), (90.0, 50.0))
        before = trajectory.position_at(0.5 - 1e-8)
        after = trajectory.position_at(0.5 + 1e-8)
        self.assertLess(math.dist(before, after), 0.001)

    def test_circle_is_periodic(self) -> None:
        trajectory = CircularTrajectory((500, 500), 100, 8)
        start = trajectory.position_at(0)
        end = trajectory.position_at(8)
        self.assertAlmostEqual(start[0], end[0], places=9)
        self.assertAlmostEqual(start[1], end[1], places=9)

    def test_figure_eight_crosses_center(self) -> None:
        trajectory = FigureEightTrajectory((500, 500), 400, 200, 10)
        self.assertEqual(trajectory.position_at(0), (500.0, 500.0))
        midpoint = trajectory.position_at(5)
        self.assertAlmostEqual(midpoint[0], 500, places=9)
        self.assertAlmostEqual(midpoint[1], 500, places=9)

    def test_random_motion_is_seeded_continuous_and_bounded(self) -> None:
        first = RandomTrajectory((500, 500), 100, 1.5, (1000, 1000), seed=42)
        second = RandomTrajectory((500, 500), 100, 1.5, (1000, 1000), seed=42)
        positions_a = [first.position_at(index / 10) for index in range(200)]
        positions_b = [second.position_at(index / 10) for index in range(200)]
        self.assertEqual(positions_a, positions_b)
        self.assertTrue(all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in positions_a))
        before = first.position_at(1.5 - 1e-8)
        after = first.position_at(1.5 + 1e-8)
        self.assertLess(math.dist(before, after), 0.001)


class VirtualCameraTests(unittest.TestCase):
    def make_camera(self, acceleration: float = 10_000) -> VirtualCamera:
        return VirtualCamera(
            world_size_px=(2000, 2000),
            viewport_px=(640, 480),
            fov_deg=(4, 3),
            initial_center_px=(1000, 1000),
            max_pan_speed_deg_s=5,
            max_tilt_speed_deg_s=5,
            max_pan_accel_deg_s2=acceleration,
            max_tilt_accel_deg_s2=acceleration,
        )

    def test_requested_rates_are_clamped(self) -> None:
        camera = self.make_camera()
        state = camera.update(CameraCommand(100, -100), 0.1)
        self.assertEqual(state.pan_rate_deg_s, 5)
        self.assertEqual(state.tilt_rate_deg_s, -5)

    def test_acceleration_is_limited(self) -> None:
        camera = self.make_camera(acceleration=2)
        state = camera.update(CameraCommand(5, 5), 0.1)
        self.assertAlmostEqual(state.pan_rate_deg_s, 0.2)
        self.assertAlmostEqual(state.tilt_rate_deg_s, 0.2)

    def test_camera_never_leaves_world(self) -> None:
        camera = self.make_camera()
        for _ in range(200):
            camera.update(CameraCommand(5, 5), 1 / 30)
        x, y = camera.state.center_world_px
        self.assertGreaterEqual(x, 320)
        self.assertLessEqual(x, 1680)
        self.assertGreaterEqual(y, 240)
        self.assertLessEqual(y, 1760)


class RendererAndEngineTests(unittest.TestCase):
    def test_beacon_is_rendered_in_camera_frame(self) -> None:
        camera = VirtualCamera(
            (2000, 2000), (640, 480), (4, 3), (1000, 1000), 5, 5
        )
        renderer = SceneRenderer((2000, 2000), (640, 480), (10, 10), "square")
        image = renderer.render_camera([(1000, 1000)], camera)
        self.assertEqual(image.shape, (480, 640))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(image), 0)
        self.assertGreater(image[240, 320], 0)

    def test_engines_from_same_scenario_are_frame_identical(self) -> None:
        scenario = load_scenario("configs/scenarios/noisy_circle.json")
        first = SimulationEngine.from_scenario(scenario)
        second = SimulationEngine.from_scenario(scenario)
        for _ in range(60):
            frame_a = first.step()
            frame_b = second.step()
            self.assertEqual(frame_a.target_world_positions, frame_b.target_world_positions)
            self.assertTrue(np.array_equal(frame_a.frame.image, frame_b.frame.image))

    def test_engine_uses_required_default_dimensions(self) -> None:
        engine = SimulationEngine.from_scenario(
            load_scenario("configs/scenarios/clear_straight.json")
        )
        snapshot = engine.step()
        self.assertEqual(snapshot.frame.image.shape, (480, 640))
        self.assertEqual(engine.renderer.world_size_px, (2000, 2000))
        self.assertEqual(snapshot.target_viewport_positions[0], (320.0, 240.0))

    def test_moving_target_stays_inside_camera_pointable_world_area(self) -> None:
        engine = SimulationEngine.from_scenario(
            load_scenario("configs/scenarios/clear_straight.json")
        )
        positions = [engine.step().target_world_positions[0] for _ in range(2400)]
        self.assertTrue(
            all(320 <= x <= 1680 and 240 <= y <= 1760 for x, y in positions)
        )


if __name__ == "__main__":
    unittest.main()
