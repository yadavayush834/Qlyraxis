"""Qlyraxis command-line entry point."""

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
        "simulate", help="run the simulator and save clean/disturbed preview frames"
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
    track = subparsers.add_parser(
        "track", help="run disturbed closed-loop tracking and pan-tilt control"
    )
    track.add_argument("scenario", help="path to scenario JSON")
    track.add_argument("--frames", type=int, default=300)
    track.add_argument("--output-dir", default="work/phase4-tracking")
    record = subparsers.add_parser(
        "record", help="render a closed-loop scenario to an MP4 test video"
    )
    record.add_argument("scenario", help="path to scenario JSON")
    record.add_argument("output", help="output MP4 path")
    record.add_argument("--frames", type=int, default=300)
    analyze = subparsers.add_parser(
        "analyze", help="detect and track a beacon in an MP4 or image sequence"
    )
    analyze.add_argument("input", help="video file, image file, or image directory")
    analyze.add_argument("--fps", type=float, default=30.0)
    analyze.add_argument("--max-frames", type=int, default=0)
    analyze.add_argument("--model", default="models/beacon_verifier.onnx")
    analyze.add_argument("--no-ai", action="store_true")
    analyze.add_argument("--output-dir", default="work/phase6-analysis")
    train_ai = subparsers.add_parser(
        "train-ai", help="train and export the tiny beacon verifier"
    )
    train_ai.add_argument("--samples-per-class", type=int, default=400)
    train_ai.add_argument("--epochs", type=int, default=250)
    train_ai.add_argument("--seed", type=int, default=26169)
    train_ai.add_argument("--model", default="models/beacon_verifier.onnx")
    train_ai.add_argument("--weights", default="models/beacon_verifier.npz")
    train_ai.add_argument("--dataset-output")
    return parser


def _train_ai(args: argparse.Namespace) -> int:
    if args.samples_per_class <= 0 or args.epochs <= 0:
        print("sample count and epochs must be positive", file=sys.stderr)
        return 2
    import numpy as np

    from qlyraxis.ai import export_onnx, generate_synthetic_dataset, train_tiny_conv

    training = generate_synthetic_dataset(args.samples_per_class, seed=args.seed)
    validation = generate_synthetic_dataset(
        max(100, args.samples_per_class // 4), seed=args.seed + 10_007
    )
    model, history = train_tiny_conv(training, epochs=args.epochs)
    probabilities = model.predict_proba(validation.images)
    accuracy = float(np.mean((probabilities >= 0.5) == validation.labels))
    if args.dataset_output:
        training.save(args.dataset_output)
    model.weights.save(args.weights)
    try:
        model_path = export_onnx(model, args.model)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Training samples: {len(training.labels)}")
    print(f"Final training loss: {history[-1]:.5f}")
    print(f"Held-out synthetic accuracy: {accuracy * 100:.2f}%")
    print(f"NumPy weights: {args.weights}")
    print(f"ONNX model: {model_path}")
    return 0


def _analyze_recording(args: argparse.Namespace) -> int:
    if args.fps <= 0 or args.max_frames < 0:
        print("--fps must be positive and --max-frames cannot be negative", file=sys.stderr)
        return 2
    from time import perf_counter

    import cv2

    from qlyraxis.ai import OnnxCandidateVerifier, VerifiedBeaconDetector
    from qlyraxis.contracts import TrackingState
    from qlyraxis.recorded import RecordedTrackingSystem
    from qlyraxis.sources import open_frame_source
    from qlyraxis.tracking.visualization import annotate_tracking
    from qlyraxis.vision import BeaconDetector

    source = None
    try:
        source = open_frame_source(args.input, args.fps)
        detector = BeaconDetector()
        backend = "classical"
        if not args.no_ai:
            verifier = OnnxCandidateVerifier(args.model)
            detector = VerifiedBeaconDetector(verifier, detector)
            backend = f"AI/{verifier.backend}"
    except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as exc:
        if source is not None:
            source.close()
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    system = RecordedTrackingSystem(detector)
    processed = 0
    detected_frames = 0
    acquired_at = None
    locked_frames = 0
    post_acquisition_frames = 0
    result = None
    start = perf_counter()
    try:
        while args.max_frames == 0 or processed < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            result = system.step(frame)
            processed += 1
            if result.detections:
                detected_frames += 1
            if result.state == TrackingState.TRACK and acquired_at is None:
                acquired_at = frame.timestamp_s
            if acquired_at is not None:
                post_acquisition_frames += 1
                if result.state in {TrackingState.TRACK, TrackingState.COAST}:
                    locked_frames += 1
    except (ValueError, cv2.error) as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        return 1
    finally:
        source.close()
    elapsed = perf_counter() - start
    if result is None:
        print("Input contains no decodable frames", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "recorded_tracking.png"
    annotated = annotate_tracking(
        result.frame.image,
        result.detections,
        result.estimate,
        result.state,
        result.search_scope,
        None,
    )
    if not cv2.imwrite(str(preview_path), annotated):
        print(f"could not write {preview_path}", file=sys.stderr)
        return 1
    retention = (
        100.0 * locked_frames / post_acquisition_frames
        if post_acquisition_frames
        else 0.0
    )
    acquired_text = (
        f"{acquired_at:.3f} s" if acquired_at is not None else "not acquired"
    )
    print(f"Inference backend: {backend}")
    print(f"Processed frames: {processed}")
    print(f"Frames with verified detections: {detected_frames}")
    print(f"Acquisition time: {acquired_text}")
    print(f"Final tracking state: {result.state}")
    print(f"Lock retention: {retention:.2f}%")
    print(f"Processing throughput: {processed / elapsed:.1f} FPS")
    print(f"Diagnostic image: {preview_path}")
    return 0


def _record_scenario(args: argparse.Namespace, scenario) -> int:
    if args.frames <= 0:
        print("--frames must be positive", file=sys.stderr)
        return 2
    import cv2

    from qlyraxis.tracking import ClosedLoopSystem

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = (int(value) for value in scenario.camera["viewport_px"])
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(scenario.camera["update_hz"]),
        (width, height),
        isColor=True,
    )
    if not writer.isOpened():
        print(f"could not create MP4: {output}", file=sys.stderr)
        return 1
    system = ClosedLoopSystem.from_scenario(scenario)
    try:
        for _ in range(args.frames):
            frame = system.step().simulation.frame.image
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    print(f"Recorded {args.frames} frames to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-ai":
        return _train_ai(args)
    if args.command == "analyze":
        return _analyze_recording(args)
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
        clean_camera_path = output_dir / f"{scenario.name}_clean-camera.png"
        overview_path = output_dir / f"{scenario.name}_overview.png"
        if not cv2.imwrite(str(camera_path), snapshot.frame.image):
            print(f"could not write {camera_path}", file=sys.stderr)
            return 1
        if not cv2.imwrite(str(clean_camera_path), snapshot.clean_frame.image):
            print(f"could not write {clean_camera_path}", file=sys.stderr)
            return 1
        if not cv2.imwrite(str(overview_path), engine.overview(snapshot)):
            print(f"could not write {overview_path}", file=sys.stderr)
            return 1
        print(
            f"Rendered {args.frames} deterministic frames at "
            f"{scenario.camera['update_hz']} Hz"
        )
        print(f"Disturbed camera preview: {camera_path}")
        print(f"Clean camera preview: {clean_camera_path}")
        print(f"World overview: {overview_path}")
    elif args.command == "detect":
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
    elif args.command == "record":
        return _record_scenario(args, scenario)
    else:
        if args.frames <= 0:
            print("--frames must be positive", file=sys.stderr)
            return 2
        import math
        from statistics import mean
        from time import perf_counter

        import cv2

        from qlyraxis.contracts import TrackingState
        from qlyraxis.tracking import ClosedLoopSystem
        from qlyraxis.tracking.visualization import annotate_tracking

        system = ClosedLoopSystem.from_scenario(scenario)
        camera_config = scenario.camera
        centroid_errors: list[float] = []
        pointing_errors: list[float] = []
        acquired_at: float | None = None
        post_acquisition_frames = 0
        locked_frames = 0
        snapshot = None
        detections = ()
        estimate = None
        start = perf_counter()
        for _ in range(args.frames):
            result = system.step()
            snapshot = result.simulation
            detections = result.detections
            estimate = result.estimate
            if result.state == TrackingState.TRACK and acquired_at is None:
                acquired_at = snapshot.frame.timestamp_s
            if acquired_at is not None:
                post_acquisition_frames += 1
                if result.state in {TrackingState.TRACK, TrackingState.COAST}:
                    locked_frames += 1

            truth = snapshot.target_sensor_positions[0]
            if truth is not None:
                pointing_errors.append(
                    math.dist(
                        truth,
                        (
                            float(camera_config["viewport_px"][0]) / 2.0,
                            float(camera_config["viewport_px"][1]) / 2.0,
                        ),
                    )
                )
                if system.tracker.selected_detection is not None:
                    centroid_errors.append(
                        math.dist(
                            (
                                system.tracker.selected_detection.x_px,
                                system.tracker.selected_detection.y_px,
                            ),
                            truth,
                        )
                    )
        elapsed = perf_counter() - start
        assert snapshot is not None

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / f"{scenario.name}_tracking.png"
        overview_path = output_dir / f"{scenario.name}_overview.png"
        annotated = annotate_tracking(
            snapshot.frame.image,
            detections,
            estimate,
            result.state,
            result.search_scope,
            result.next_command,
        )
        if not cv2.imwrite(str(annotated_path), annotated):
            print(f"could not write {annotated_path}", file=sys.stderr)
            return 1
        if not cv2.imwrite(str(overview_path), system.engine.overview(snapshot)):
            print(f"could not write {overview_path}", file=sys.stderr)
            return 1

        acquisition_text = (
            f"{acquired_at:.3f} s" if acquired_at is not None else "not acquired"
        )
        retention = (
            100.0 * locked_frames / post_acquisition_frames
            if post_acquisition_frames
            else 0.0
        )
        print(f"Processed frames: {args.frames}")
        print(f"Acquisition time: {acquisition_text}")
        print(f"Final tracking state: {result.state}")
        print(f"Lock retention: {retention:.2f}%")
        if centroid_errors:
            print(f"Mean centroid error: {mean(centroid_errors):.3f} px")
            print(f"Maximum centroid error: {max(centroid_errors):.3f} px")
        if pointing_errors:
            print(f"Mean camera pointing offset: {mean(pointing_errors):.3f} px")
        print(f"Pipeline throughput: {args.frames / elapsed:.1f} FPS")
        print(f"Diagnostic images: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
