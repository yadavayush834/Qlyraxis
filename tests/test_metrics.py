import json
import tempfile
import unittest
from pathlib import Path

from qlyraxis.contracts import CameraCommand, Detection, TrackEstimate, TrackingState
from qlyraxis.metrics import (
    PerformanceRecorder,
    StressCell,
    export_profile_comparison,
    export_stress_report,
)


def detection(x: float, y: float) -> Detection:
    return Detection(x, y, 0.9, 8, 8)


def estimate(x: float, y: float, state: TrackingState) -> TrackEstimate:
    return TrackEstimate(x, y, 0, 0, 0.9, state)


class PerformanceRecorderTests(unittest.TestCase):
    def test_summary_calculates_required_metrics(self) -> None:
        recorder = PerformanceRecorder("test_run", (640, 480))
        states = (
            TrackingState.ACQUIRE,
            TrackingState.TRACK,
            TrackingState.COAST,
            TrackingState.REACQUIRE,
            TrackingState.TRACK,
        )
        for index, state in enumerate(states):
            selected = detection(321, 240) if state != TrackingState.REACQUIRE else None
            recorder.record(
                frame_index=index,
                timestamp_s=index * 0.1,
                state=state,
                detections=() if selected is None else (selected,),
                selected=selected,
                estimate=None if selected is None else estimate(321, 240, state),
                command=CameraCommand(1, -1),
                truth=(320, 240),
                processing_time_ms=20,
            )
        summary = recorder.summary()
        self.assertEqual(summary.frames, 5)
        self.assertAlmostEqual(summary.acquisition_time_s, 0.1)
        self.assertAlmostEqual(summary.average_tracking_error_px, 0.0)
        self.assertAlmostEqual(summary.p95_tracking_error_px, 0.0)
        self.assertAlmostEqual(summary.maximum_tracking_error_px, 0.0)
        self.assertAlmostEqual(summary.average_centroid_error_px, 1.0)
        self.assertAlmostEqual(summary.maximum_centroid_error_px, 1.0)
        self.assertAlmostEqual(summary.lock_retention_percent, 50.0)
        self.assertAlmostEqual(summary.processing_fps, 50.0)
        self.assertEqual(summary.reacquisition_count, 1)
        self.assertAlmostEqual(summary.maximum_reacquisition_time_s, 0.1)

    def test_strict_lock_requires_a_detection_and_small_camera_offset(self) -> None:
        recorder = PerformanceRecorder("strict_lock", (640, 480))
        samples = (
            (detection(320, 240), (320, 240)),
            (None, (320, 240)),
            (detection(320, 240), (340, 240)),
            (detection(350, 240), (320, 240)),
        )
        for index, (selected, truth) in enumerate(samples):
            recorder.record(
                frame_index=index,
                timestamp_s=index * 0.1,
                state=TrackingState.TRACK,
                detections=() if selected is None else (selected,),
                selected=selected,
                estimate=None
                if selected is None
                else estimate(selected.x_px, selected.y_px, TrackingState.TRACK),
                command=CameraCommand(0, 0),
                truth=truth,
                processing_time_ms=10,
            )
        self.assertAlmostEqual(recorder.summary().lock_retention_percent, 25.0)

    def test_all_report_formats_are_written(self) -> None:
        recorder = PerformanceRecorder(
            "export_test", (640, 480), configuration={"seed": 26169}
        )
        recorder.record(
            frame_index=0,
            timestamp_s=0,
            state=TrackingState.TRACK,
            detections=(detection(320, 240),),
            selected=detection(320, 240),
            estimate=estimate(320, 240, TrackingState.TRACK),
            command=CameraCommand(0, 0),
            truth=(320, 240),
            processing_time_ms=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = recorder.export(directory)
            self.assertEqual(set(paths), {"csv", "json", "html"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(paths["json"].read_text())
            html = paths["html"].read_text()
        self.assertEqual(payload["summary"]["scenario"], "export_test")
        self.assertIn("centroid_error_px", payload["frames"][0])
        self.assertIn("Qlyraxis Performance Report", html)
        self.assertIn("<svg", html)

    def test_profile_comparison_contains_both_runs_and_graph(self) -> None:
        baseline = PerformanceRecorder("comparison", (640, 480))
        improved = PerformanceRecorder("comparison", (640, 480))
        for recorder, truth in (
            (baseline, (340, 240)),
            (improved, (324, 240)),
        ):
            recorder.record(
                frame_index=0,
                timestamp_s=0,
                state=TrackingState.TRACK,
                detections=(detection(*truth),),
                selected=detection(*truth),
                estimate=estimate(*truth, TrackingState.TRACK),
                command=CameraCommand(0, 0),
                truth=truth,
                processing_time_ms=10,
            )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_profile_comparison(
                "comparison", baseline, improved, directory
            )
            payload = json.loads(paths["json"].read_text())
            report = paths["html"].read_text()
        self.assertEqual(set(paths), {"json", "html"})
        self.assertEqual(payload["baseline"]["average_tracking_error_px"], 20.0)
        self.assertEqual(payload["improved"]["average_tracking_error_px"], 4.0)
        self.assertIn("Baseline", report)
        self.assertIn("Improved", report)
        self.assertIn("<svg", report)

    def test_stress_report_exports_safe_envelope_heatmaps(self) -> None:
        cells = (
            StressCell(0, 0, 0.1, 2, 4, 98, 60, True),
            StressCell(0, 10, 0.2, 12, 20, 45, 55, False),
            StressCell(8, 0, 0.3, 5, 9, 85, 40, True),
            StressCell(8, 10, 0.4, 18, 30, 20, 35, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_stress_report("stress", "improved", 90, cells, directory)
            payload = json.loads(paths["json"].read_text())
            report = paths["html"].read_text()
        self.assertEqual(payload["safe_cells"], 2)
        self.assertEqual(payload["robustness_score_percent"], 50.0)
        self.assertIn("Mean camera offset", report)
        self.assertIn("Strict lock retention", report)
        self.assertGreaterEqual(report.count('class="heat"'), 2)

    def test_non_increasing_timestamp_is_rejected(self) -> None:
        recorder = PerformanceRecorder("timestamps", (640, 480))
        kwargs = {
            "state": TrackingState.SEARCH,
            "detections": (),
            "selected": None,
            "estimate": None,
            "command": None,
            "truth": None,
            "processing_time_ms": 1,
        }
        recorder.record(frame_index=0, timestamp_s=0, **kwargs)
        with self.assertRaisesRegex(ValueError, "increase"):
            recorder.record(frame_index=1, timestamp_s=0, **kwargs)


if __name__ == "__main__":
    unittest.main()
