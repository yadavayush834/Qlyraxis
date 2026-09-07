#!/usr/bin/env python3
"""Build a compact cross-scenario summary from generated JSON logs."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path("deliverables/performance")
    summaries = []
    for path in sorted(root.glob("*/*_performance.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(payload["summary"])
    output = root / "benchmark_summary.json"
    output.write_text(
        json.dumps(
            {
                "problem_statement": 26169,
                "evaluation_duration_s": 60,
                "scenario_count": len(summaries),
                "thresholds": {
                    "acquisition_time_s_max": 2,
                    "reacquisition_time_s_max": 1,
                    "tracking_error_px_max": 10,
                    "target_loss_percent_max": 5,
                    "processing_fps_min": 20,
                },
                "runs": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
