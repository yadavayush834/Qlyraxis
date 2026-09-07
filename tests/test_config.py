import json
import unittest
from copy import deepcopy
from pathlib import Path

from qlyraxis.config import ConfigError, load_scenario, validate_scenario


SCENARIO_DIR = Path(__file__).parents[1] / "configs" / "scenarios"


class ScenarioConfigTests(unittest.TestCase):
    def test_all_bundled_scenarios_validate(self) -> None:
        paths = sorted(SCENARIO_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(load_scenario(path).name, path.stem)

    def test_pan_speed_outside_official_range_is_rejected(self) -> None:
        data = json.loads((SCENARIO_DIR / "clear_straight.json").read_text())
        invalid = deepcopy(data)
        invalid["camera"]["max_pan_speed_deg_s"] = 11
        with self.assertRaisesRegex(ConfigError, "between 5 and 10"):
            validate_scenario(invalid)

    def test_target_smaller_than_five_pixels_is_rejected(self) -> None:
        data = json.loads((SCENARIO_DIR / "clear_straight.json").read_text())
        invalid = deepcopy(data)
        invalid["target"]["size_px"] = [4, 10]
        with self.assertRaisesRegex(ConfigError, "between 5 and 20"):
            validate_scenario(invalid)

    def test_low_camera_rate_is_rejected(self) -> None:
        data = json.loads((SCENARIO_DIR / "clear_straight.json").read_text())
        invalid = deepcopy(data)
        invalid["camera"]["update_hz"] = 20
        with self.assertRaisesRegex(ConfigError, "at least 30"):
            validate_scenario(invalid)

    def test_unknown_platform_motion_is_rejected(self) -> None:
        data = json.loads((SCENARIO_DIR / "clear_straight.json").read_text())
        data["disturbances"]["platform_motion"] = "teleport"
        with self.assertRaisesRegex(ConfigError, "platform_motion"):
            validate_scenario(data)

    def test_atmosphere_strength_above_one_is_rejected(self) -> None:
        data = json.loads((SCENARIO_DIR / "clear_straight.json").read_text())
        data["disturbances"]["atmosphere_strength"] = 1.1
        with self.assertRaisesRegex(ConfigError, "atmosphere_strength"):
            validate_scenario(data)

    def test_invalid_dropout_contract_is_rejected(self) -> None:
        data = json.loads((SCENARIO_DIR / "clear_straight.json").read_text())
        data["disturbances"]["dropout"]["enabled"] = "yes"
        with self.assertRaisesRegex(ConfigError, "must be boolean"):
            validate_scenario(data)


if __name__ == "__main__":
    unittest.main()
