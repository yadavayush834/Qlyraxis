import math
import unittest

import cv2
import numpy as np

from qlyraxis.config import load_scenario
from qlyraxis.contracts import Detection, FramePacket, TrackingState
from qlyraxis.simulation import SimulationEngine
from qlyraxis.tracking import ClosedLoopSystem
from qlyraxis.vision import CodeLockDetector


class _TwoCandidateDetector:
    def detect(self, _frame: FramePacket):
        return (
            Detection(100, 100, 0.8, 9, 9),
            Detection(200, 100, 0.8, 9, 9),
        )


class CodeLockTests(unittest.TestCase):
    pattern = "0000011001010"
    decoy = "1111100110101"

    @staticmethod
    def frame(index: int, first: int, second: int) -> FramePacket:
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(image, (100, 100), 4, first, -1)
        cv2.circle(image, (200, 100), 4, second, -1)
        return FramePacket(index, index / 30, image)

    def test_temporal_code_rejects_brighter_complementary_decoy(self) -> None:
        detector = CodeLockDetector(
            _TwoCandidateDetector(), self.pattern, "QLX-07"
        )
        result = ()
        for index in range(len(self.pattern)):
            result = detector.detect(
                self.frame(
                    index,
                    255 if self.pattern[index] == "1" else 90,
                    255 if self.decoy[index] == "1" else 90,
                )
            )
        self.assertEqual(detector.identity_status, "VERIFIED")
        self.assertAlmostEqual(detector.best_correlation or 0, 1.0, places=6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].x_px, 100)

    def test_simulator_modulates_primary_and_decoy_with_opposite_codes(self) -> None:
        engine = SimulationEngine.from_scenario(
            load_scenario("configs/scenarios/codelock_decoy.json")
        )
        first = engine.step()
        first_points = first.target_viewport_positions
        assert first_points[0] is not None and first_points[1] is not None
        primary_0 = first.clean_frame.image[
            round(first_points[0][1]), round(first_points[0][0])
        ]
        decoy_0 = first.clean_frame.image[
            round(first_points[1][1]), round(first_points[1][0])
        ]
        self.assertLess(primary_0, decoy_0)
        sixth = None
        for _ in range(5):
            sixth = engine.step()
        assert sixth is not None
        sixth_points = sixth.target_viewport_positions
        assert sixth_points[0] is not None and sixth_points[1] is not None
        primary_5 = sixth.clean_frame.image[
            round(sixth_points[0][1]), round(sixth_points[0][0])
        ]
        decoy_5 = sixth.clean_frame.image[
            round(sixth_points[1][1]), round(sixth_points[1][0])
        ]
        self.assertGreater(primary_5, decoy_5)

    def test_closed_loop_locks_registered_identity_not_visual_decoy(self) -> None:
        scenario = load_scenario("configs/scenarios/codelock_decoy.json")
        ordinary = ClosedLoopSystem.from_scenario(
            scenario, use_ai=False, use_code_lock=False
        )
        ordinary_result = None
        for _ in range(3):
            ordinary_result = ordinary.step()
        assert ordinary_result is not None
        ordinary_selected = ordinary.tracker.selected_detection
        decoy_truth = ordinary_result.simulation.target_sensor_positions[1]
        assert ordinary_selected is not None and decoy_truth is not None
        self.assertLess(
            math.dist((ordinary_selected.x_px, ordinary_selected.y_px), decoy_truth),
            3,
        )

        protected = ClosedLoopSystem.from_scenario(scenario, use_ai=False)
        protected_result = None
        for _ in range(25):
            protected_result = protected.step()
        assert protected_result is not None
        selected = protected.tracker.selected_detection
        primary_truth = protected_result.simulation.target_sensor_positions[0]
        assert selected is not None and primary_truth is not None
        self.assertEqual(protected_result.state, TrackingState.TRACK)
        self.assertEqual(protected.code_lock.identity_status, "VERIFIED")
        self.assertLess(math.dist((selected.x_px, selected.y_px), primary_truth), 3)

    def test_closed_loop_dwells_while_collecting_identity(self) -> None:
        scenario = load_scenario("configs/scenarios/codelock_decoy.json")
        protected = ClosedLoopSystem.from_scenario(scenario, use_ai=False)

        first = protected.step()

        self.assertEqual(protected.code_lock.identity_status, "COLLECTING")
        self.assertEqual(first.next_command.pan_rate_deg_s, 0.0)
        self.assertEqual(first.next_command.tilt_rate_deg_s, 0.0)


if __name__ == "__main__":
    unittest.main()
