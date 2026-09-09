"""Side-by-side baseline and improved closed-loop benchmark reports."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from qlyraxis.metrics.recorder import PerformanceRecorder


def export_profile_comparison(
    scenario_name: str,
    baseline: PerformanceRecorder,
    improved: PerformanceRecorder,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export machine-readable results and a portable comparison dashboard."""

    if not baseline.rows or not improved.rows:
        raise ValueError("both comparison profiles must contain recorded frames")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = scenario_name.replace(" ", "_")
    paths = {
        "json": output / f"{stem}_comparison.json",
        "html": output / f"{stem}_comparison.html",
    }
    baseline_summary = asdict(baseline.summary())
    improved_summary = asdict(improved.summary())
    payload = {
        "scenario": scenario_name,
        "lock_tolerance_px": improved.lock_tolerance_px,
        "baseline": baseline_summary,
        "improved": improved_summary,
        "change": _changes(baseline_summary, improved_summary),
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["html"].write_text(
        _html_report(scenario_name, payload, baseline.rows, improved.rows),
        encoding="utf-8",
    )
    return paths


def _changes(baseline: dict[str, Any], improved: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "average_tracking_error_px",
        "p95_tracking_error_px",
        "maximum_tracking_error_px",
        "lock_retention_percent",
        "acquisition_time_s",
        "processing_fps",
    ):
        before = baseline.get(key)
        after = improved.get(key)
        result[key] = (
            None if before is None or after is None else float(after) - float(before)
        )
    return result


def _html_report(
    scenario_name: str,
    payload: dict[str, Any],
    baseline_rows: Sequence[dict[str, Any]],
    improved_rows: Sequence[dict[str, Any]],
) -> str:
    baseline = payload["baseline"]
    improved = payload["improved"]
    tolerance = float(payload["lock_tolerance_px"])
    metrics = (
        ("Mean camera offset", "average_tracking_error_px", "px", "lower"),
        ("P95 camera offset", "p95_tracking_error_px", "px", "lower"),
        ("Maximum camera offset", "maximum_tracking_error_px", "px", "lower"),
        ("Mean centroid / identity error", "average_centroid_error_px", "px", "lower"),
        (f"Strict lock (≤{tolerance:g} px)", "lock_retention_percent", "%", "higher"),
        ("Acquisition time", "acquisition_time_s", "s", "lower"),
        ("Processing throughput", "processing_fps", "FPS", "higher"),
    )
    table_rows = []
    for label, key, unit, preferred in metrics:
        before = baseline.get(key)
        after = improved.get(key)
        before_text = _display(before, unit)
        after_text = _display(after, unit)
        delta_text, delta_class = _delta(before, after, unit, preferred)
        table_rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{html.escape(before_text)}</td>"
            f"<td>{html.escape(after_text)}</td>"
            f'<td class="{delta_class}">{html.escape(delta_text)}</td>'
            "</tr>"
        )
    baseline_error = _series(baseline_rows, "tracking_error_px")
    improved_error = _series(improved_rows, "tracking_error_px")
    chart = _comparison_chart(baseline_error, improved_error, tolerance)
    title = html.escape(f"Qlyraxis Profile Comparison — {scenario_name}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
:root{{--bg:#071019;--panel:#101d29;--border:#294052;--text:#eef6fc;--muted:#9eb2c5;--cyan:#36d6c2;--amber:#ffb454;--green:#73e3a2;--red:#ff7b86}}
*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
main{{max-width:1040px;margin:38px auto;padding:0 28px}}h1{{font-size:30px;margin:0 0 7px}}.sub{{color:var(--muted);margin:0 0 26px}}
section{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px;margin:18px 0;overflow:auto}}
table{{width:100%;border-collapse:collapse;min-width:650px}}th,td{{padding:11px;border-bottom:1px solid var(--border);text-align:right;font-variant-numeric:tabular-nums}}
th:first-child{{text-align:left;color:var(--muted);font-weight:500}}thead th{{color:var(--text);font-weight:650}}.good{{color:var(--green)}}.bad{{color:var(--red)}}.neutral{{color:var(--muted)}}
.legend{{display:flex;gap:22px;color:var(--muted);font-size:13px;margin:8px 0}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
svg{{width:100%;height:auto;background:#07131d;border-radius:9px}}.base{{fill:none;stroke:var(--amber);stroke-width:2;opacity:.88}}.improved{{fill:none;stroke:var(--cyan);stroke-width:2}}.limit{{stroke:var(--green);stroke-width:1;stroke-dasharray:6 5;opacity:.8}}
footer{{color:#7890a4;margin:25px 0;font-size:13px}}</style></head>
<body><main><h1>{title}</h1><p class="sub">Same deterministic scenario and seed; only the tracking profile changes.</p>
<section><h2>Measured result</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Improved</th><th>Change</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
<section><h2>Camera offset by frame</h2><div class="legend"><span><i class="dot" style="background:#ffb454"></i>Baseline</span><span><i class="dot" style="background:#36d6c2"></i>Improved</span><span><i class="dot" style="background:#73e3a2"></i>Strict-lock threshold</span></div>{chart}</section>
<footer>Generated by Qlyraxis PS 26169 · camera offset is measured from beacon truth to the optical axis</footer></main></body></html>"""


def _display(value: Any, unit: str) -> str:
    return "N/A" if value is None else f"{float(value):.2f} {unit}"


def _delta(before: Any, after: Any, unit: str, preferred: str) -> tuple[str, str]:
    if before is None or after is None:
        return "N/A", "neutral"
    change = float(after) - float(before)
    useful = change > 0 if preferred == "higher" else change < 0
    css = "neutral" if abs(change) < 1e-9 else ("good" if useful else "bad")
    return f"{change:+.2f} {unit}", css


def _series(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _comparison_chart(
    baseline: Sequence[float], improved: Sequence[float], tolerance: float
) -> str:
    width, height = 940, 260
    if not baseline and not improved:
        return "<p>No ground-truth samples were available.</p>"
    combined = sorted([*baseline, *improved])
    percentile_index = round((len(combined) - 1) * 0.98)
    maximum = max(tolerance, 1.0, combined[percentile_index] * 1.15)

    def points(values: Sequence[float]) -> str:
        return " ".join(
            f"{10 + index * (width - 20) / max(len(values) - 1, 1):.1f},"
            f"{height - 14 - min(value, maximum) / maximum * (height - 34):.1f}"
            for index, value in enumerate(values)
        )

    limit_y = height - 14 - tolerance / maximum * (height - 34)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Baseline and improved camera pointing offset">'
        f'<line class="limit" x1="10" x2="{width - 10}" y1="{limit_y:.1f}" y2="{limit_y:.1f}"/>'
        f'<polyline class="base" points="{points(baseline)}"/>'
        f'<polyline class="improved" points="{points(improved)}"/>'
        f'<text x="13" y="20" fill="#9eb2c5">Scale 0–{maximum:.1f} px · peaks clipped</text></svg>'
    )
