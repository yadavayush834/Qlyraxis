import math
import unittest

import cv2
import numpy as np

from qlyraxis.config import load_scenario
from qlyraxis.contracts import Detection
from qlyraxis.simulation import SimulationEngine
from qlyraxis.vision import AcquisitionGate, AcquisitionState, BeaconDetector
from qlyraxis.vision.preprocess import FramePreprocessor


def square_beacon(
    center: tuple[int, int] = (320, 240),
    size: int = 7,
    background: np.ndarray | None = None,
) -> np.ndarray:
    image = (
        np.zeros((480, 640), dtype=np.uint8)
        if background is None
        else background.copy()
    )
    half = size // 2
    image[
        center[1] - half : center[1] + half + 1,
        center[0] - half : center[0] + half + 1,
    ] = 255
    return image


class PreprocessorTests(unittest.TestCase):
    def test_bgr_input_is_converted_to_monochrome(self) -> None:
        grayscale = square_beacon()
        bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        processed = FramePreprocessor().process(bgr)
        self.assertEqual(processed.grayscale.shape, (480, 640))
        self.assertEqual(processed.grayscale.dtype, np.uint8)

    def test_invalid_image_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "grayscale, BGR, or BGRA"):
            FramePreprocessor().process(np.zeros((10, 10, 2), dtype=np.uint8))


class BeaconDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = BeaconDetector()

    def test_blank_frame_has_no_detection(self) -> None:
        self.assertEqual(
            self.detector.detect_image(np.zeros((480, 640), dtype=np.uint8)),
            (),
        )

    def test_square_centroid_is_subpixel_accurate(self) -> None:
        detections = self.detector.detect_image(square_beacon((271, 183), 7))
        self.assertEqual(len(detections), 1)
        self.assertLess(math.hypot(detections[0].x_px - 271, detections[0].y_px - 183), 0.1)
        self.assertGreater(detections[0].confidence, 0.9)

    def test_small_circular_beacon_is_detected(self) -> None:
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(image, (211, 301), 3, 255, thickness=-1, lineType=cv2.LINE_AA)
        detections = self.detector.detect_image(image)
        self.assertEqual(len(detections), 1)
        self.assertLess(math.hypot(detections[0].x_px - 211, detections[0].y_px - 301), 0.2)

    def test_two_separated_beacons_are_detected(self) -> None:
        image = square_beacon((150, 150), 7)
        image = square_beacon((480, 330), 9, image)
        detections = self.detector.detect_image(image)
        self.assertEqual(len(detections), 2)

    def test_isolated_salt_noise_is_rejected(self) -> None:
        rng = np.random.default_rng(26169)
        image = np.zeros((480, 640), dtype=np.uint8)
        indices = rng.choice(image.size, int(image.size * 0.01), replace=False)
        image.flat[indices] = 255
        image = square_beacon((320, 240), 7, image)
        detections = self.detector.detect_image(image)
        self.assertEqual(len(detections), 1)
        self.assertLess(math.hypot(detections[0].x_px - 320, detections[0].y_px - 240), 0.2)

    def test_gaussian_background_retains_correct_top_candidate(self) -> None:
        rng = np.random.default_rng(26169)
        noisy = np.clip(rng.normal(18, 20, (480, 640)), 0, 255).astype(np.uint8)
        image = square_beacon((320, 240), 7, noisy)
        detections = self.detector.detect_image(image)
        self.assertGreaterEqual(len(detections), 1)
        self.assertLess(math.hypot(detections[0].x_px - 320, detections[0].y_px - 240), 0.5)

    def test_multiscale_response_is_generated(self) -> None:
        debug = self.detector.detect_debug(
            type("Packet", (), {"image": square_beacon(), "index": 0})()
        )
        self.assertGreater(np.count_nonzero(debug.multiscale_mask), 0)
        self.assertGreater(np.count_nonzero(debug.candidate_mask), 0)


class AcquisitionGateTests(unittest.TestCase):
    @staticmethod
    def detection(x: float, y: float, confidence: float = 0.9) -> Detection:
        return Detection(x, y, confidence, 8, 8)

    def test_three_consistent_frames_confirm_acquisition(self) -> None:
        gate = AcquisitionGate(confirmation_frames=3)
        self.assertEqual(
            gate.update([self.detection(100, 100)]).state,
            AcquisitionState.VERIFYING,
        )
        gate.update([self.detection(104, 102)])
        result = gate.update([self.detection(108, 104)])
        self.assertEqual(result.state, AcquisitionState.ACQUIRED)
        self.assertEqual(result.confirmation_count, 3)

    def test_inconsistent_candidate_does_not_confirm(self) -> None:
        gate = AcquisitionGate(confirmation_frames=3, missed_frame_tolerance=0)
        gate.update([self.detection(100, 100)])
        result = gate.update([self.detection(300, 300)])
        self.assertEqual(result.state, AcquisitionState.SEARCHING)
        self.assertEqual(result.confirmation_count, 0)

    def test_low_confidence_candidate_is_ignored(self) -> None:
        gate = AcquisitionGate(minimum_confidence=0.5, missed_frame_tolerance=0)
        result = gate.update([self.detection(100, 100, confidence=0.2)])
        self.assertEqual(result.state, AcquisitionState.SEARCHING)


class SimulatorIntegrationTests(unittest.TestCase):
    def test_clear_scenario_acquires_with_accurate_centroids(self) -> None:
        engine = SimulationEngine.from_scenario(
            load_scenario("configs/scenarios/clear_straight.json")
        )
        detector = BeaconDetector()
        gate = AcquisitionGate()
        errors = []
        acquired_at = None
        for _ in range(60):
            snapshot = engine.step()
            detections = detector.detect(snapshot.frame)
            result = gate.update(detections)
            truth = snapshot.target_viewport_positions[0]
            self.assertIsNotNone(truth)
            self.assertGreaterEqual(len(detections), 1)
            errors.append(math.dist((detections[0].x_px, detections[0].y_px), truth))
            if result.state == AcquisitionState.ACQUIRED and acquired_at is None:
                acquired_at = snapshot.frame.timestamp_s
        self.assertIsNotNone(acquired_at)
        self.assertLessEqual(acquired_at, 2.0)
        self.assertLess(float(np.mean(errors)), 0.75)


if __name__ == "__main__":
    unittest.main()
