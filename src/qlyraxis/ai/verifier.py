"""ONNX candidate scoring and detector composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from qlyraxis.contracts import Detection, FramePacket
from qlyraxis.vision import BeaconDetector


def extract_candidate_patch(
    image: NDArray,
    detection: Detection,
    patch_size: int = 32,
) -> NDArray[np.float32]:
    if patch_size < 16 or patch_size % 2:
        raise ValueError("patch_size must be an even integer of at least 16")
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        grayscale = image
    else:
        raise ValueError("candidate image must be grayscale or BGR")
    patch = cv2.getRectSubPix(
        grayscale.astype(np.float32),
        (patch_size, patch_size),
        (float(detection.x_px), float(detection.y_px)),
    )
    return (patch[None] / 255.0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    detection: Detection
    ai_score: float


class OnnxCandidateVerifier:
    """Score candidate patches with ONNX Runtime or OpenCV's CPU backend."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        patch_size: int = 32,
        backend: str = "auto",
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if backend not in {"auto", "opencv", "onnxruntime"}:
            raise ValueError("backend must be auto, opencv, or onnxruntime")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.threshold = threshold
        self.patch_size = patch_size
        self.backend = "opencv"
        self._session = None
        self._network = None
        if backend in {"auto", "onnxruntime"}:
            try:
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
                self.backend = "onnxruntime"
            except Exception as exc:
                if backend == "onnxruntime":
                    raise RuntimeError(
                        f"could not initialize onnxruntime backend: {exc}"
                    ) from exc
        if self._session is None:
            self._network = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def score(self, patches: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._session is not None:
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: patches})[0]
        else:
            assert self._network is not None
            self._network.setInput(patches)
            output = self._network.forward()
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def verify(
        self,
        image: NDArray,
        detections: Sequence[Detection],
    ) -> tuple[VerifiedCandidate, ...]:
        if not detections:
            return ()
        patches = np.stack(
            [extract_candidate_patch(image, item, self.patch_size) for item in detections]
        ).astype(np.float32)
        scores = self.score(patches)
        return tuple(
            VerifiedCandidate(detection, float(score))
            for detection, score in zip(detections, scores, strict=True)
            if score >= self.threshold
        )


class VerifiedBeaconDetector:
    """Apply the tiny CNN after the high-recall classical detector."""

    def __init__(
        self,
        verifier: OnnxCandidateVerifier,
        detector: BeaconDetector | None = None,
        ai_weight: float = 0.5,
        verification_interval_frames: int = 1,
    ) -> None:
        if not 0.0 <= ai_weight <= 1.0:
            raise ValueError("AI weight must be in [0, 1]")
        if verification_interval_frames < 1:
            raise ValueError("verification interval must be positive")
        self.verifier = verifier
        self.detector = detector or BeaconDetector()
        self.ai_weight = ai_weight
        self.verification_interval_frames = verification_interval_frames
        self._frame_counter = 0
        self.last_verified: tuple[VerifiedCandidate, ...] = ()

    def detect(self, frame: FramePacket) -> tuple[Detection, ...]:
        candidates = self.detector.detect(frame)
        verify_now = self._frame_counter % self.verification_interval_frames == 0
        self._frame_counter += 1
        if not verify_now:
            self.last_verified = ()
            return candidates
        self.last_verified = self.verifier.verify(frame.image, candidates)
        return tuple(
            Detection(
                item.detection.x_px,
                item.detection.y_px,
                min(
                    1.0,
                    (1.0 - self.ai_weight) * item.detection.confidence
                    + self.ai_weight * item.ai_score,
                ),
                item.detection.width_px,
                item.detection.height_px,
            )
            for item in self.last_verified
        )
