import json
import tempfile
import unittest
from pathlib import Path

from qlyraxis.contracts import CameraCommand, Detection, TrackEstimate, TrackingState
from qlyraxis.metrics import PerformanceRecorder


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
        self.assertAlmostEqual(summary.average_tracking_error_px, 1.0)
        self.assertAlmostEqual(summary.maximum_tracking_error_px, 1.0)
        self.assertAlmostEqual(summary.lock_retention_percent, 75.0)
        self.assertAlmostEqual(summary.processing_fps, 50.0)
        self.assertEqual(summary.reacquisition_count, 1)
        self.assertAlmostEqual(summary.maximum_reacquisition_time_s, 0.1)

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
        self.assertIn("Qlyraxis Performance Report", html)
        self.assertIn("<svg", html)

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
