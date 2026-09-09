# Phase 7 delivery and local testing

Phase 7 completes the Qlyraxis desktop application, performance reporting,
packaging, and SIH documentation. Run all commands from the repository root.

## Launch from source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
qlyraxis gui
```

Select a scenario, optionally change the disturbance overrides, and press
**Start**. The video panel overlays detections, the filtered estimate, tracker
state, and camera command. **Export performance report** writes matching CSV,
JSON, and HTML files for the current run.

## Generate a repeatable report

```bash
qlyraxis benchmark configs/scenarios/clear_straight.json \
  --output-dir reports/clear_straight
```

Omitting `--frames` runs the scenario's configured 60-second duration. The JSON
contains the scenario configuration, summary, state counts, and all frame
records. The CSV is convenient for spreadsheet analysis, while the HTML is a
self-contained judge-facing report with a camera-offset chart. Tracking error is
the beacon's offset from the optical axis; centroid accuracy is recorded
separately. Strict lock counts only genuine detections within 10 pixels.

Generate deterministic baseline-versus-improved evidence with:

```bash
qlyraxis compare configs/scenarios/noisy_circle.json --frames 600 \
  --output-dir reports/noisy_circle_comparison
```

The comparison uses the same scenario and seed for both profiles, then overlays
their camera-offset curves in one HTML report.

## Run the automated suite

```bash
python -m unittest discover -s tests -v
```

The suite covers configuration, deterministic simulation, disturbance replay,
vision, AI inference, tracking, control, recorded sources, reporting, and GUI
chart calculations.

## Build and test the Linux package

```bash
python -m pip install -e '.[packaging]'
scripts/build_executable.sh
dist/Qlyraxis/Qlyraxis --help
dist/Qlyraxis/Qlyraxis validate configs/scenarios/clear_straight.json
dist/Qlyraxis/Qlyraxis benchmark configs/scenarios/clear_straight.json \
  --frames 30 --output-dir work/package-smoke
dist/Qlyraxis/Qlyraxis gui
```

Copy the complete `dist/Qlyraxis` directory, not only the launcher. PyInstaller
bundles Python, Tcl/Tk, NumPy, OpenCV, scenarios, documentation, and the ONNX
model. Build separately on Windows when a Windows executable is required;
PyInstaller does not cross-compile.

## Included evidence

- `deliverables/Qlyraxis_Technical_Report.docx`
- `deliverables/Qlyraxis_User_Manual.docx`
- `deliverables/performance/benchmark_summary.json`
- CSV, JSON, and HTML logs for all five official scenarios under
  `deliverables/performance/`
