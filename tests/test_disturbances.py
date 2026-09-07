import unittest

import numpy as np

from qlyraxis.config import load_scenario
from qlyraxis.disturbances import DisturbancePipeline
from qlyraxis.simulation import SimulationEngine


def disturbance_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "noise": [],
        "noise_std_px": 0,
        "salt_pepper_fraction": 0,
        "camera_jitter_max_px_frame": 0,
        "atmosphere": "clear",
        "atmosphere_strength": 0,
        "turbulence_strength_px": 0,
        "defocus_blur_px": 0,
        "motion_blur_px": 0,
        "platform_motion": "none",
        "platform_motion_max_px_frame": 0,
        "dropout": {"enabled": False, "start_s": 0, "duration_s": 0},
    }
    config.update(overrides)
    return config


def beacon_image() -> np.ndarray:
    image = np.zeros((80, 100), dtype=np.uint8)
    image[37:44, 47:54] = 255
    return image


class DisturbancePipelineTests(unittest.TestCase):
    def apply(
        self,
        config: dict[str, object],
        *,
        frame_index: int = 12,
        time_s: float = 0.4,
    ):
        return DisturbancePipeline(config, seed=26169).apply(
            beacon_image(),
            [(50.0, 40.0)],
            frame_index=frame_index,
            time_s=time_s,
            update_hz=30,
        )

    def test_clear_profile_preserves_frame_and_ground_truth(self) -> None:
        result = self.apply(disturbance_config())
        np.testing.assert_array_equal(result.image, beacon_image())
        self.assertEqual(result.transformed_points, ((50.0, 40.0),))
        self.assertEqual(result.metadata.total_shift_px, (0.0, 0.0))

    def test_same_seed_and_frame_are_bitwise_reproducible(self) -> None:
        config = disturbance_config(
            noise=["gaussian", "poisson", "salt_pepper"],
            noise_std_px=8,
            salt_pepper_fraction=0.02,
            camera_jitter_max_px_frame=10,
            atmosphere="rain",
            atmosphere_strength=0.5,
            turbulence_strength_px=2,
            platform_motion="random",
            platform_motion_max_px_frame=5,
        )
        first = self.apply(config, frame_index=41)
        second = self.apply(config, frame_index=41)
        np.testing.assert_array_equal(first.image, second.image)
        self.assertEqual(first.transformed_points, second.transformed_points)
        self.assertEqual(first.metadata, second.metadata)

    def test_frame_index_changes_stochastic_disturbances(self) -> None:
        config = disturbance_config(
            noise=["gaussian"],
            noise_std_px=8,
            camera_jitter_max_px_frame=10,
        )
        first = self.apply(config, frame_index=5)
        second = self.apply(config, frame_index=6)
        self.assertFalse(np.array_equal(first.image, second.image))
        self.assertNotEqual(first.metadata.camera_jitter_px, second.metadata.camera_jitter_px)

    def test_jitter_and_platform_motion_transform_truth_consistently(self) -> None:
        config = disturbance_config(
            camera_jitter_max_px_frame=10,
            platform_motion="circular",
            platform_motion_max_px_frame=4,
        )
        result = self.apply(config)
        shift_x, shift_y = result.metadata.total_shift_px
        transformed = result.transformed_points[0]
        assert transformed is not None
        self.assertAlmostEqual(transformed[0], 50.0 + shift_x)
        self.assertAlmostEqual(transformed[1], 40.0 + shift_y)
        self.assertNotEqual(result.metadata.total_shift_px, (0.0, 0.0))

    def test_turbulence_warps_frame_and_point(self) -> None:
        result = self.apply(disturbance_config(turbulence_strength_px=3))
        self.assertFalse(np.array_equal(result.image, beacon_image()))
        self.assertNotEqual(result.transformed_points[0], (50.0, 40.0))

    def test_blur_spreads_beacon_energy(self) -> None:
        result = self.apply(
            disturbance_config(defocus_blur_px=3, motion_blur_px=7)
        )
        self.assertGreater(np.count_nonzero(result.image), np.count_nonzero(beacon_image()))
        self.assertLess(int(result.image.max()), 255)

    def test_atmosphere_profiles_have_expected_brightness_effects(self) -> None:
        haze = self.apply(
            disturbance_config(atmosphere="haze", atmosphere_strength=0.6)
        ).image
        low_light = self.apply(
            disturbance_config(atmosphere="low_light", atmosphere_strength=0.6)
        ).image
        self.assertGreater(int(haze[0, 0]), 0)
        self.assertLess(int(low_light[40, 50]), int(beacon_image()[40, 50]))

    def test_dropout_removes_beacon_only_inside_window(self) -> None:
        config = disturbance_config(
            dropout={"enabled": True, "start_s": 1.0, "duration_s": 0.5}
        )
        before = self.apply(config, time_s=0.9)
        during = self.apply(config, time_s=1.2)
        after = self.apply(config, time_s=1.5)
        self.assertFalse(before.metadata.beacon_dropped)
        self.assertTrue(during.metadata.beacon_dropped)
        self.assertEqual(int(during.image.max()), 0)
        self.assertFalse(after.metadata.beacon_dropped)
        self.assertGreater(int(after.image.max()), 0)


class DisturbedSimulationTests(unittest.TestCase):
    def test_engine_exposes_clean_and_disturbed_frames(self) -> None:
        scenario = load_scenario("configs/scenarios/noisy_circle.json")
        first_engine = SimulationEngine.from_scenario(scenario)
        second_engine = SimulationEngine.from_scenario(scenario)
        first = first_engine.step()
        second = second_engine.step()
        self.assertFalse(np.array_equal(first.clean_frame.image, first.frame.image))
        np.testing.assert_array_equal(first.frame.image, second.frame.image)
        self.assertEqual(first.disturbances, second.disturbances)


if __name__ == "__main__":
    unittest.main()
