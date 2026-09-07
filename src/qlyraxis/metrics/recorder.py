"""Per-frame telemetry, summary metrics, and CSV/JSON/HTML reports."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from qlyraxis.contracts import CameraCommand, Detection, TrackEstimate, TrackingState


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    scenario: str
    frames: int
    simulation_duration_s: float
    processing_fps: float
    acquisition_time_s: float | None
    average_tracking_error_px: float | None
    maximum_tracking_error_px: float | None
    average_pointing_offset_px: float | None
    lock_retention_percent: float
    target_loss_percent: float
    average_processing_time_ms: float
    maximum_processing_time_ms: float
    reacquisition_count: int
    average_reacquisition_time_s: float | None
    maximum_reacquisition_time_s: float | None
    final_state: str


class PerformanceRecorder:
    """Collect only observable pipeline data plus optional evaluation truth."""

    FIELDNAMES = (
        "frame_index",
        "timestamp_s",
        "state",
        "detection_count",
        "selected_x_px",
        "selected_y_px",
        "estimate_x_px",
        "estimate_y_px",
        "truth_x_px",
        "truth_y_px",
        "tracking_error_px",
        "pointing_offset_px",
        "pan_rate_deg_s",
        "tilt_rate_deg_s",
        "processing_time_ms",
    )

    def __init__(
        self,
        scenario_name: str,
        viewport_px: tuple[float, float],
        configuration: dict[str, Any] | None = None,
    ) -> None:
        self.scenario_name = scenario_name
        self.viewport_px = viewport_px
        self.configuration = configuration or {}
        self.rows: list[dict[str, Any]] = []
        self._acquired_at: float | None = None
        self._loss_started_at: float | None = None
        self._reacquisition_times: list[float] = []
        self._previous_state: TrackingState | None = None

    def record(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        state: TrackingState,
        detections: Sequence[Detection],
        selected: Detection | None,
        estimate: TrackEstimate | None,
        command: CameraCommand | None,
        truth: tuple[float, float] | None,
        processing_time_ms: float,
    ) -> None:
        if self.rows and timestamp_s <= float(self.rows[-1]["timestamp_s"]):
            raise ValueError("metric timestamps must increase")
        if processing_time_ms < 0:
            raise ValueError("processing_time_ms cannot be negative")
        if state == TrackingState.TRACK and self._acquired_at is None:
            self._acquired_at = timestamp_s
        if (
            self._acquired_at is not None
            and self._previous_state in {TrackingState.TRACK, TrackingState.COAST}
            and state == TrackingState.REACQUIRE
            and self._loss_started_at is None
        ):
            self._loss_started_at = timestamp_s
        if state == TrackingState.TRACK and self._loss_started_at is not None:
            self._reacquisition_times.append(timestamp_s - self._loss_started_at)
            self._loss_started_at = None

        tracking_error = None
        pointing_offset = None
        if truth is not None:
            center = (self.viewport_px[0] / 2.0, self.viewport_px[1] / 2.0)
            pointing_offset = math.dist(truth, center)
            if selected is not None:
                tracking_error = math.dist((selected.x_px, selected.y_px), truth)
        self.rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": timestamp_s,
                "state": state.value,
                "detection_count": len(detections),
                "selected_x_px": None if selected is None else selected.x_px,
                "selected_y_px": None if selected is None else selected.y_px,
                "estimate_x_px": None if estimate is None else estimate.x_px,
                "estimate_y_px": None if estimate is None else estimate.y_px,
                "truth_x_px": None if truth is None else truth[0],
                "truth_y_px": None if truth is None else truth[1],
                "tracking_error_px": tracking_error,
                "pointing_offset_px": pointing_offset,
                "pan_rate_deg_s": None if command is None else command.pan_rate_deg_s,
                "tilt_rate_deg_s": None if command is None else command.tilt_rate_deg_s,
                "processing_time_ms": processing_time_ms,
            }
        )
        self._previous_state = state

    def summary(self) -> PerformanceSummary:
        if not self.rows:
            raise ValueError("cannot summarize an empty performance log")
        processing_times = [float(row["processing_time_ms"]) for row in self.rows]
        tracking_errors = [
            float(row["tracking_error_px"])
            for row in self.rows
            if row["tracking_error_px"] is not None
        ]
        pointing_offsets = [
            float(row["pointing_offset_px"])
            for row in self.rows
            if row["pointing_offset_px"] is not None
        ]
        post_acquisition = [
            row
            for row in self.rows
            if self._acquired_at is not None
            and float(row["timestamp_s"]) >= self._acquired_at
        ]
        locked = sum(
            row["state"] in {TrackingState.TRACK.value, TrackingState.COAST.value}
            for row in post_acquisition
        )
        retention = 100.0 * locked / len(post_acquisition) if post_acquisition else 0.0
        duration = float(self.rows[-1]["timestamp_s"]) - float(
            self.rows[0]["timestamp_s"]
        )
        average_process = mean(processing_times)
        return PerformanceSummary(
            scenario=self.scenario_name,
            frames=len(self.rows),
            simulation_duration_s=duration,
            processing_fps=1000.0 / average_process if average_process > 0 else 0.0,
            acquisition_time_s=self._acquired_at,
            average_tracking_error_px=(
                mean(tracking_errors) if tracking_errors else None
            ),
            maximum_tracking_error_px=(max(tracking_errors) if tracking_errors else None),
            average_pointing_offset_px=(
                mean(pointing_offsets) if pointing_offsets else None
            ),
            lock_retention_percent=retention,
            target_loss_percent=100.0 - retention,
            average_processing_time_ms=average_process,
            maximum_processing_time_ms=max(processing_times),
            reacquisition_count=len(self._reacquisition_times),
            average_reacquisition_time_s=(
                mean(self._reacquisition_times) if self._reacquisition_times else None
            ),
            maximum_reacquisition_time_s=(
                max(self._reacquisition_times) if self._reacquisition_times else None
            ),
            final_state=str(self.rows[-1]["state"]),
        )

    def export(self, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stem = self.scenario_name.replace(" ", "_")
        paths = {
            "csv": output / f"{stem}_performance.csv",
            "json": output / f"{stem}_performance.json",
            "html": output / f"{stem}_report.html",
        }
        with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self.FIELDNAMES, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self.rows)
        payload = {
            "summary": asdict(self.summary()),
            "configuration": self.configuration,
            "state_counts": dict(Counter(row["state"] for row in self.rows)),
            "frames": self.rows,
        }
        paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths["html"].write_text(self._html_report(payload), encoding="utf-8")
        return paths

    def _html_report(self, payload: dict[str, Any]) -> str:
        summary = payload["summary"]
        error_values = [
            float(row["tracking_error_px"])
            for row in self.rows
            if row["tracking_error_px"] is not None
        ]
        chart = _sparkline(error_values, width=760, height=180)
        metric_order = (
            ("Frames", "frames", ""),
            ("Duration", "simulation_duration_s", " s"),
            ("Throughput", "processing_fps", " FPS"),
            ("Acquisition", "acquisition_time_s", " s"),
            ("Mean tracking error", "average_tracking_error_px", " px"),
            ("Maximum tracking error", "maximum_tracking_error_px", " px"),
            ("Lock retention", "lock_retention_percent", "%"),
            ("Maximum processing time", "maximum_processing_time_ms", " ms"),
        )
        rows = []
        for label, key, suffix in metric_order:
            value = summary[key]
            display = "N/A" if value is None else _format_number(value) + suffix
            rows.append(
                f"<tr><th>{html.escape(label)}</th><td>{html.escape(display)}</td></tr>"
            )
        title = html.escape(f"Qlyraxis Performance Report {self.scenario_name}")
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
body{{font:15px system-ui,sans-serif;margin:0;background:#081018;color:#e7eef5}}
main{{max-width:920px;margin:40px auto;padding:0 28px}}
h1{{font-size:30px;margin-bottom:6px}} .sub{{color:#9eb2c5;margin-bottom:28px}}
section{{background:#101c27;border:1px solid #26394a;border-radius:12px;padding:22px;margin:18px 0}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:10px;border-bottom:1px solid #26394a;text-align:left}}
th{{color:#9eb2c5;width:55%;font-weight:500}} td{{font-variant-numeric:tabular-nums}}
svg{{width:100%;height:auto;background:#071018;border-radius:8px}} .line{{fill:none;stroke:#36d6c2;stroke-width:2}}
footer{{color:#7890a4;margin:26px 0;font-size:13px}}</style></head>
<body><main><h1>{title}</h1><p class="sub">Automatically generated deterministic benchmark report</p>
<section><h2>Run summary</h2><table>{''.join(rows)}</table></section>
<section><h2>Tracking error by measured frame</h2>{chart}</section>
<section><h2>State counts</h2><p>{html.escape(json.dumps(payload['state_counts'], sort_keys=True))}</p></section>
<footer>Generated by Qlyraxis PS 26169</footer></main></body></html>"""


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def _sparkline(values: Sequence[float], width: int, height: int) -> str:
    if not values:
        return "<p>No ground-truth error samples were available.</p>"
    maximum = max(max(values), 1.0)
    usable_height = height - 24
    points = []
    for index, value in enumerate(values):
        x_px = index * (width - 20) / max(len(values) - 1, 1) + 10
        y_px = height - 12 - value / maximum * usable_height
        points.append(f"{x_px:.1f},{y_px:.1f}")
    label = html.escape(f"Peak {maximum:.3f} px")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Tracking error">'
        f'<polyline class="line" points="{" ".join(points)}"/>'
        f'<text x="12" y="20" fill="#9eb2c5">{label}</text></svg>'
    )
