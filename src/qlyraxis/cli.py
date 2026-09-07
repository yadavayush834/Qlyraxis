"""Phase 1 command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qlyraxis.config import ConfigError, load_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlyraxis",
        description="Qlyraxis FSOC virtual tracking laboratory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate", "validate a scenario configuration"),
        ("show", "validate and summarize a scenario configuration"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("scenario", help="path to scenario JSON")
    simulate = subparsers.add_parser(
        "simulate", help="run the Phase 2 simulator and save final preview frames"
    )
    simulate.add_argument("scenario", help="path to scenario JSON")
    simulate.add_argument("--frames", type=int, default=90)
    simulate.add_argument("--output-dir", default="work/phase2-preview")
    detect = subparsers.add_parser(
        "detect", help="run Phase 3 beacon detection on simulated camera frames"
    )
    detect.add_argument("scenario", help="path to scenario JSON")
    detect.add_argument("--frames", type=int, default=60)
    detect.add_argument("--output-dir", default="work/phase3-detection")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "show":
        print(scenario.summary())
    elif args.command == "validate":
        print(f"Valid scenario: {scenario.name}")
    elif args.command == "simulate":
        if args.frames <= 0:
            print("--frames must be positive", file=sys.stderr)
            return 2
        import cv2

        from qlyraxis.simulation import SimulationEngine

        engine = SimulationEngine.from_scenario(scenario)
        snapshot = None
        for _ in range(args.frames):
            snapshot = engine.step()
        assert snapshot is not None
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        camera_path = output_dir / f"{scenario.name}_camera.png"
        overview_path = output_dir / f"{scenario.name}_overview.png"
        if not cv2.imwrite(str(camera_path), snapshot.frame.image):
            print(f"could not write {camera_path}", file=sys.stderr)
            return 1
        if not cv2.imwrite(str(overview_path), engine.overview(snapshot)):
            print(f"could not write {overview_path}", file=sys.stderr)
            return 1
        print(
            f"Rendered {args.frames} deterministic frames at "
            f"{scenario.camera['update_hz']} Hz"
        )
        print(f"Camera preview: {camera_path}")
        print(f"World overview: {overview_path}")
    else:
        if args.frames <= 0:
            print("--frames must be positive", file=sys.stderr)
            return 2
        import cv2

        from qlyraxis.simulation import SimulationEngine
        from qlyraxis.vision import AcquisitionGate, BeaconDetector
        from qlyraxis.vision.visualization import annotate_detections

        engine = SimulationEngine.from_scenario(scenario)
        detector = BeaconDetector()
        acquisition_gate = AcquisitionGate()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        detected_frames = 0
        first_acquired_s = None
        debug = None
        acquisition = acquisition_gate.result
        snapshot = None
        for _ in range(args.frames):
            snapshot = engine.step()
            debug = detector.detect_debug(snapshot.frame)
            acquisition = acquisition_gate.update(debug.detections)
            if debug.detections:
                detected_frames += 1
            if acquisition.state == "acquired" and first_acquired_s is None:
                first_acquired_s = snapshot.frame.timestamp_s
        assert snapshot is not None and debug is not None

        annotated = annotate_detections(
            snapshot.frame.image,
            debug.detections,
            acquisition,
        )
        outputs = {
            "annotated": annotated,
            "intensity-mask": debug.intensity_mask,
            "multiscale-mask": debug.multiscale_mask,
            "candidate-mask": debug.candidate_mask,
        }
        for suffix, image in outputs.items():
            path = output_dir / f"{scenario.name}_{suffix}.png"
            if not cv2.imwrite(str(path), image):
                print(f"could not write {path}", file=sys.stderr)
                return 1
        acquisition_text = (
            f"{first_acquired_s:.3f} s" if first_acquired_s is not None else "not acquired"
        )
        print(f"Processed frames: {args.frames}")
        print(f"Frames with a detection: {detected_frames}")
        print(f"First acquisition: {acquisition_text}")
        print(f"Final acquisition state: {acquisition.state}")
        print(f"Diagnostic images: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
