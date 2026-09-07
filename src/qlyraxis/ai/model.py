"""A compact NumPy-trained convolutional classifier with ONNX export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from qlyraxis.ai.dataset import PatchDataset


def _fixed_kernels() -> NDArray[np.float32]:
    kernels = np.array(
        [
            [[0, 1, 0], [1, 4, 1], [0, 1, 0]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]],
        ],
        dtype=np.float32,
    )
    scales = np.maximum(np.abs(kernels).sum(axis=(1, 2), keepdims=True), 1.0)
    return (kernels / scales)[:, None]


def _convolution_features(
    images: NDArray[np.float32],
    weights: NDArray[np.float32],
    bias: NDArray[np.float32],
) -> NDArray[np.float32]:
    padded = np.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = sliding_window_view(padded, (3, 3), axis=(2, 3))
    convolved = np.einsum("nchwkl,ockl->nohw", windows, weights, optimize=True)
    activated = np.maximum(convolved + bias[None, :, None, None], 0.0)
    height = activated.shape[2] // 2
    width = activated.shape[3] // 2
    pooled = activated[:, :, : height * 2, : width * 2].reshape(
        len(images), weights.shape[0], height, 2, width, 2
    )
    return pooled.max(axis=(3, 5)).reshape(len(images), -1).astype(np.float32)


@dataclass(frozen=True, slots=True)
class TinyConvWeights:
    conv_weight: NDArray[np.float32]
    conv_bias: NDArray[np.float32]
    linear_weight: NDArray[np.float32]
    linear_bias: NDArray[np.float32]
    patch_size: int

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            conv_weight=self.conv_weight,
            conv_bias=self.conv_bias,
            linear_weight=self.linear_weight,
            linear_bias=self.linear_bias,
            patch_size=np.array(self.patch_size, dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyConvWeights":
        with np.load(path) as archive:
            return cls(
                archive["conv_weight"].astype(np.float32),
                archive["conv_bias"].astype(np.float32),
                archive["linear_weight"].astype(np.float32),
                archive["linear_bias"].astype(np.float32),
                int(archive["patch_size"]),
            )


class TinyConvModel:
    def __init__(self, weights: TinyConvWeights) -> None:
        self.weights = weights

    def predict_proba(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        expected = (1, self.weights.patch_size, self.weights.patch_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"model input must have shape [N, {expected}]")
        features = _convolution_features(
            images.astype(np.float32),
            self.weights.conv_weight,
            self.weights.conv_bias,
        )
        logits = features @ self.weights.linear_weight.T + self.weights.linear_bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return probabilities[:, 0].astype(np.float32)


def train_tiny_conv(
    dataset: PatchDataset,
    epochs: int = 250,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
) -> tuple[TinyConvModel, tuple[float, ...]]:
    """Train the compact classifier head on deterministic convolution features."""

    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid training hyperparameters")
    patch_size = int(dataset.images.shape[-1])
    if dataset.images.shape[2] != patch_size or patch_size % 2:
        raise ValueError("training patches must be square and even-sized")
    conv_weight = _fixed_kernels()
    conv_bias = np.zeros(conv_weight.shape[0], dtype=np.float32)
    raw_features = _convolution_features(dataset.images, conv_weight, conv_bias)
    feature_mean = raw_features.mean(axis=0)
    feature_scale = np.maximum(raw_features.std(axis=0), 1e-3)
    features = (raw_features - feature_mean) / feature_scale
    labels = dataset.labels[:, None]
    weights = np.zeros((features.shape[1], 1), dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    history = []
    for _ in range(epochs):
        logits = features @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        error = probabilities - labels
        weights -= learning_rate * (
            features.T @ error / len(labels) + l2 * weights
        )
        bias -= learning_rate * error.mean(axis=0)
        loss = -np.mean(
            labels * np.log(probabilities + 1e-7)
            + (1.0 - labels) * np.log(1.0 - probabilities + 1e-7)
        )
        history.append(float(loss))
    converted_weight = (weights[:, 0] / feature_scale).astype(np.float32)
    converted_bias = np.array(
        [bias[0] - np.dot(feature_mean, converted_weight)], dtype=np.float32
    )
    trained = TinyConvWeights(
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        linear_weight=converted_weight[None],
        linear_bias=converted_bias,
        patch_size=patch_size,
    )
    return TinyConvModel(trained), tuple(history)


def export_onnx(model: TinyConvModel, path: str | Path) -> Path:
    """Export the NumPy model to a portable ONNX graph."""

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("ONNX export requires: pip install qlyraxis[training]") from exc

    weights = model.weights
    feature_count = int(weights.linear_weight.shape[1])
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["conv"], ["relu"]),
            helper.make_node("MaxPool", ["relu"], ["pool"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "linear_w", "linear_b"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["probability"]),
        ],
        "QlyraxisTinyConv",
        [
            helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [None, 1, weights.patch_size, weights.patch_size],
            )
        ],
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])],
        initializer=[
            numpy_helper.from_array(weights.conv_weight, "conv_w"),
            numpy_helper.from_array(weights.conv_bias, "conv_b"),
            numpy_helper.from_array(weights.linear_weight.T, "linear_w"),
            numpy_helper.from_array(weights.linear_bias, "linear_b"),
        ],
        value_info=[
            helper.make_tensor_value_info("flat", TensorProto.FLOAT, [None, feature_count])
        ],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="qlyraxis",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(onnx_model)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, output)
    return output
