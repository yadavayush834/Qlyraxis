import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from qlyraxis.ai import (
    OnnxCandidateVerifier,
    PatchDataset,
    TinyConvWeights,
    export_onnx,
    extract_candidate_patch,
    generate_synthetic_dataset,
    train_tiny_conv,
)
from qlyraxis.contracts import Detection


class SyntheticDatasetTests(unittest.TestCase):
    def test_dataset_is_balanced_normalized_and_reproducible(self) -> None:
        first = generate_synthetic_dataset(12, seed=42)
        second = generate_synthetic_dataset(12, seed=42)
        self.assertEqual(first.images.shape, (24, 1, 32, 32))
        self.assertEqual(float(first.labels.sum()), 12.0)
        self.assertGreaterEqual(float(first.images.min()), 0.0)
        self.assertLessEqual(float(first.images.max()), 1.0)
        np.testing.assert_array_equal(first.images, second.images)
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_dataset_round_trip(self) -> None:
        dataset = generate_synthetic_dataset(5, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patches.npz"
            dataset.save(path)
            restored = PatchDataset.load(path)
        np.testing.assert_array_equal(dataset.images, restored.images)
        np.testing.assert_array_equal(dataset.labels, restored.labels)


class TinyConvModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training = generate_synthetic_dataset(120, seed=26169)
        cls.model, cls.history = train_tiny_conv(cls.training, epochs=180)

    def test_training_converges_and_generalizes(self) -> None:
        validation = generate_synthetic_dataset(100, seed=36176)
        probabilities = self.model.predict_proba(validation.images)
        accuracy = np.mean((probabilities >= 0.5) == validation.labels)
        self.assertLess(self.history[-1], self.history[0])
        self.assertGreaterEqual(float(accuracy), 0.88)

    def test_numpy_weights_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            self.model.weights.save(path)
            restored = TinyConvWeights.load(path)
        self.assertEqual(restored.patch_size, 32)
        np.testing.assert_array_equal(
            self.model.weights.linear_weight, restored.linear_weight
        )

    def test_candidate_patch_is_centered_and_normalized(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        image[29:36, 29:36] = 255
        detection = Detection(32, 32, 0.9, 7, 7)
        patch = extract_candidate_patch(image, detection)
        self.assertEqual(patch.shape, (1, 32, 32))
        self.assertEqual(float(patch.max()), 1.0)
        self.assertGreater(float(patch[0, 16, 16]), 0.9)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
    def test_onnx_cpu_inference_matches_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(self.model, Path(directory) / "model.onnx")
            verifier = OnnxCandidateVerifier(path, backend="opencv")
            actual = verifier.score(self.training.images[:12])
        expected = self.model.predict_proba(self.training.images[:12])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_committed_model_runs_with_default_opencv_backend(self) -> None:
        verifier = OnnxCandidateVerifier("models/beacon_verifier.onnx", backend="opencv")
        validation = generate_synthetic_dataset(40, seed=991)
        probabilities = verifier.score(validation.images)
        accuracy = np.mean((probabilities >= 0.5) == validation.labels)
        self.assertGreaterEqual(float(accuracy), 0.85)


if __name__ == "__main__":
    unittest.main()
