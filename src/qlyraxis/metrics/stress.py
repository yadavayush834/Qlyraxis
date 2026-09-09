"""Robustness-envelope data model and portable heatmap reporting."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class StressCell:
    noise_sigma: float
    jitter_px_frame: float
    acquisition_time_s: float | None
    mean_camera_offset_px: float | None
    p95_camera_offset_px: float | None
    strict_lock_percent: float
    processing_fps: float
    passed: bool


def export_stress_report(
    scenario_name: str,
    profile: str,
    frames_per_cell: int,
    cells: Sequence[StressCell],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write JSON evidence and a self-contained two-metric robustness heatmap."""

    if frames_per_cell <= 0:
        raise ValueError("frames per stress cell must be positive")
    if not cells:
        raise ValueError("stress report requires at least one cell")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = scenario_name.replace(" ", "_")
    paths = {
        "json": output / f"{stem}_stress.json",
        "html": output / f"{stem}_stress.html",
    }
    pass_count = sum(cell.passed for cell in cells)
    payload: dict[str, Any] = {
        "scenario": scenario_name,
        "profile": profile,
        "frames_per_cell": frames_per_cell,
        "criteria": {
            "acquisition_time_s_max": 2.0,
            "mean_camera_offset_px_max": 10.0,
            "strict_lock_percent_min": 80.0,
            "processing_fps_min": 20.0,
        },
        "safe_cells": pass_count,
        "total_cells": len(cells),
        "robustness_score_percent": 100.0 * pass_count / len(cells),
        "cells": [asdict(cell) for cell in cells],
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["html"].write_text(_html_report(payload), encoding="utf-8")
    return paths


def _html_report(payload: dict[str, Any]) -> str:
    cells = payload["cells"]
    noises = sorted({float(cell["noise_sigma"]) for cell in cells})
    jitters = sorted({float(cell["jitter_px_frame"]) for cell in cells})
    by_coordinate = {
        (float(cell["noise_sigma"]), float(cell["jitter_px_frame"])): cell
        for cell in cells
    }
    offset_map = _heatmap(
        "Mean camera offset",
        "mean_camera_offset_px",
        "px",
        noises,
        jitters,
        by_coordinate,
        lower_is_better=True,
        target=10.0,
    )
    lock_map = _heatmap(
        "Strict lock retention",
        "strict_lock_percent",
        "%",
        noises,
        jitters,
        by_coordinate,
        lower_is_better=False,
        target=80.0,
    )
    rows = []
    for cell in cells:
        status = "PASS" if cell["passed"] else "LIMIT"
        status_class = "pass" if cell["passed"] else "limit"
        rows.append(
            "<tr>"
            f"<td>{float(cell['noise_sigma']):g}</td>"
            f"<td>{float(cell['jitter_px_frame']):g}</td>"
            f"<td>{_value(cell['acquisition_time_s'], 's')}</td>"
            f"<td>{_value(cell['mean_camera_offset_px'], 'px')}</td>"
            f"<td>{_value(cell['p95_camera_offset_px'], 'px')}</td>"
            f"<td>{float(cell['strict_lock_percent']):.1f}%</td>"
            f"<td>{float(cell['processing_fps']):.1f}</td>"
            f'<td class="{status_class}">{status}</td></tr>'
        )
    title = html.escape(f"Qlyraxis Robustness Lab — {payload['scenario']}")
    score = float(payload["robustness_score_percent"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
:root{{--bg:#071019;--panel:#101d29;--border:#294052;--text:#eef6fc;--muted:#9eb2c5;--cyan:#36d6c2;--green:#73e3a2;--amber:#ffbd66;--red:#ff7b86}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}main{{max-width:1120px;margin:38px auto;padding:0 28px}}
h1{{font-size:30px;margin:0 0 7px}}.sub{{color:var(--muted);margin:0 0 24px}}.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.card,section{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px}}.card span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.card strong{{display:block;font-size:28px;margin-top:8px}}
.maps{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}}section{{margin:18px 0;overflow:auto}}.maps section{{margin:0}}table{{width:100%;border-collapse:collapse;min-width:620px}}th,td{{padding:10px;border-bottom:1px solid var(--border);text-align:right;font-variant-numeric:tabular-nums}}th{{color:var(--muted)}}th:first-child,td:first-child{{text-align:left}}
.heat{{min-width:0}}.heat td{{font-weight:700;color:#071019;border:4px solid var(--panel);border-radius:9px;text-align:center;min-width:64px}}.heat th{{text-align:center;border:0}}.pass{{color:var(--green);font-weight:800}}.limit{{color:var(--red);font-weight:800}}.legend{{color:var(--muted);font-size:12px}}footer{{color:#7890a4;margin:25px 0;font-size:13px}}
@media(max-width:760px){{.summary,.maps{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>{title}</h1><p class="sub">Automated deterministic disturbance sweep · {html.escape(payload['profile'])} tracking profile</p>
<div class="summary"><div class="card"><span>Robustness score</span><strong>{score:.1f}%</strong></div><div class="card"><span>Safe operating cells</span><strong>{payload['safe_cells']} / {payload['total_cells']}</strong></div><div class="card"><span>Evidence frames</span><strong>{payload['frames_per_cell'] * payload['total_cells']}</strong></div></div>
<div class="maps">{offset_map}{lock_map}</div>
<section><h2>Complete evidence matrix</h2><table><thead><tr><th>Noise σ</th><th>Jitter px/frame</th><th>Acquire</th><th>Mean offset</th><th>P95 offset</th><th>Strict lock</th><th>FPS</th><th>Envelope</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<footer>PASS requires acquisition ≤2 s, mean camera offset ≤10 px, strict lock ≥80%, and processing ≥20 FPS. Generated by Qlyraxis PS 26169.</footer></main></body></html>"""


def _heatmap(
    title: str,
    key: str,
    unit: str,
    noises: Sequence[float],
    jitters: Sequence[float],
    cells: dict[tuple[float, float], dict[str, Any]],
    *,
    lower_is_better: bool,
    target: float,
) -> str:
    header = "".join(f"<th>{value:g}</th>" for value in jitters)
    rows = []
    for noise in noises:
        entries = []
        for jitter in jitters:
            cell = cells.get((noise, jitter))
            value = None if cell is None else cell.get(key)
            if value is None:
                entries.append('<td style="background:#6e7b86">N/A</td>')
                continue
            numeric = float(value)
            quality = (
                (2.0 * target - numeric) / target
                if lower_is_better
                else numeric / target
            )
            quality = max(0.0, min(quality, 1.0))
            hue = 8.0 + 142.0 * quality
            entries.append(
                f'<td style="background:hsl({hue:.0f} 70% 62%)">{numeric:.1f}</td>'
            )
        rows.append(f"<tr><th>{noise:g}</th>{''.join(entries)}</tr>")
    return (
        f'<section><h2>{html.escape(title)}</h2><p class="legend">Rows: noise σ · columns: jitter px/frame · target {target:g} {unit}</p>'
        f'<table class="heat"><thead><tr><th>σ ↓ / jitter →</th>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
    )


def _value(value: Any, unit: str) -> str:
    return "N/A" if value is None else f"{float(value):.2f} {unit}"
