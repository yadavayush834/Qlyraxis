"""Synthetic training data and tiny ONNX beacon verification."""

from qlyraxis.ai.dataset import PatchDataset, generate_synthetic_dataset
from qlyraxis.ai.model import (
    TinyConvModel,
    TinyConvWeights,
    export_onnx,
    train_tiny_conv,
)
from qlyraxis.ai.verifier import (
    OnnxCandidateVerifier,
    VerifiedBeaconDetector,
    VerifiedCandidate,
    extract_candidate_patch,
)

__all__ = [
    "OnnxCandidateVerifier",
    "PatchDataset",
    "TinyConvModel",
    "TinyConvWeights",
    "VerifiedBeaconDetector",
    "VerifiedCandidate",
    "export_onnx",
    "extract_candidate_patch",
    "generate_synthetic_dataset",
    "train_tiny_conv",
]
