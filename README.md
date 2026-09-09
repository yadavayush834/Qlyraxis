# Qlyraxis

Qlyraxis is an AI-assisted virtual camera tracking laboratory for coarse
alignment of mobile Free Space Optical Communication (FSOC) terminals. It
simulates optical beacons and disturbances, acquires and tracks a selected
beacon, controls a constrained virtual pan-tilt camera, and generates
reproducible performance reports.

## Current status

The original seven delivery phases plus the Robustness Lab and CodeLock identity
phase are complete. The repository contains configuration
validation, a deterministic virtual environment, image-only beacon acquisition,
Kalman motion estimation, a complete tracking state machine, bounded PID pan-tilt
control, local/global re-acquisition, and a deterministic disturbance pipeline.
The pipeline supports Gaussian, Poisson, and salt-and-pepper noise; haze, fog,
rain, and low light; defocus and motion blur; camera jitter; platform motion;
atmospheric turbulence; and timed beacon dropout.

Recorded MP4 files and naturally sorted image sequences now use the same frame
contract as the simulator. A compact candidate verifier is trained on synthetic
patches, exported to ONNX, and executed on the CPU through OpenCV or optional
ONNX Runtime.

The final release adds a Tk desktop dashboard, live telemetry and camera-offset history,
automatic CSV/JSON/HTML performance reports, five 60-second benchmark logs, a
PyInstaller Linux application bundle, a technical report, and a user manual.
The improved profile uses velocity feed-forward, maneuver adaptation, and the
bundled AI candidate verifier. A deterministic comparison command runs that
profile against the classical non-predictive baseline and draws both offset
curves in one portable HTML report. In simulation, neural verification runs at
10 Hz inside the 30 Hz control loop, while classical detection and tracking run
on every frame; this preserves CPU headroom without disabling the AI stage.

## Quick start

Python 3.11 or newer is required.

```bash
python -m pip install -e .
qlyraxis validate configs/scenarios/clear_straight.json
qlyraxis show configs/scenarios/clear_straight.json
qlyraxis simulate configs/scenarios/clear_straight.json --frames 90
qlyraxis detect configs/scenarios/clear_straight.json --frames 60
qlyraxis track configs/scenarios/clear_straight.json --frames 300
qlyraxis track configs/scenarios/fog_figure_eight.json --frames 300
qlyraxis track configs/scenarios/reacquisition_dropout.json --frames 660
qlyraxis record configs/scenarios/fog_figure_eight.json work/fog.mp4 --frames 240
qlyraxis analyze work/fog.mp4 --model models/beacon_verifier.onnx
qlyraxis benchmark configs/scenarios/clear_straight.json --output-dir reports/clear
qlyraxis compare configs/scenarios/jitter_random.json --frames 600 --output-dir reports/jitter-comparison
qlyraxis stress-test configs/scenarios/clear_straight.json --frames 180 --output-dir reports/robustness
qlyraxis gui
python -m unittest discover -s tests -v
```

Without installing the package:

```bash
PYTHONPATH=src python -m qlyraxis validate configs/scenarios/clear_straight.json
```

Build a standalone Linux folder with:

```bash
python -m pip install -e '.[packaging]'
scripts/build_executable.sh
dist/Qlyraxis/Qlyraxis gui
```

## Repository layout

```text
configs/scenarios/       Reproducible SIH benchmark scenarios
deliverables/            Technical report, manual, and benchmark evidence
docs/                    Architecture, project plan, and UI wireframe
models/                  Portable ONNX verifier and reproducible NumPy weights
packaging/               PyInstaller application specification
scripts/                 Release and benchmark helper scripts
src/qlyraxis/            Simulator, sources, AI, vision, tracking, and control
tests/                   Unit and closed-loop integration tests
```

## Non-negotiable design rules

1. The tracker receives rendered frames only, never simulator ground truth.
2. Simulated video, MP4 files, and future camera input use one `FrameSource`
   interface.
3. Every run stores its full configuration and random seed.
4. Control commands respect pan, tilt, acceleration, and update-rate limits.
5. Performance claims must be generated automatically from recorded runs.

In reports and the GUI, **camera offset** is the distance between the beacon and
the optical axis and is the primary tracking metric. **Beacon confidence** is a
detector/verifier score, not a lock percentage. **Strict lock retention** counts
only real detections in `TRACK` whose camera offset is at most 10 pixels; predicted
`COAST` frames do not count as locked.

## Phase 8: Robustness Lab

`stress-test` automatically sweeps configurable sensor-noise and camera-jitter
levels against the same deterministic target. It exports JSON evidence and two
judge-friendly heatmaps for mean camera offset and strict lock retention. A cell
is inside the safe operating envelope only when acquisition is at most 2 s, mean
camera offset is at most 10 px, strict lock is at least 80%, and processing
remains at least 20 FPS. Override the grid with `--noise-levels 0,4,8` and
`--jitter-levels 0,5,10`.

## Phase 9: CodeLock optical identity

CodeLock verifies the designated FSOC terminal from its repeating intensity
signature before releasing a candidate to the motion tracker. The bundled
`codelock_decoy` scenario contains two visually identical moving lights: the
registered `QLX-07` beacon begins dim while a complementary-code decoy begins
bright. CodeLock collects 13 temporal samples, searches every cyclic phase,
and locks only when normalized code correlation reaches 0.70.

```bash
qlyraxis track configs/scenarios/codelock_decoy.json --frames 300
qlyraxis compare configs/scenarios/codelock_decoy.json --frames 600 \
  --output-dir reports/codelock
```

For coded scenarios, `compare` keeps AI, maneuver adaptation, controller gains,
scene, and seed identical on both sides; only CodeLock is disabled in the
baseline. This makes the identity result a controlled A/B experiment.

See `docs/architecture.md`, `docs/project-plan.md`, and
`docs/phase7-delivery.md` for the design, local operation, and release workflow.
