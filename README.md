# Qlyraxis

Qlyraxis is an AI-assisted virtual camera tracking laboratory for coarse
alignment of mobile Free Space Optical Communication (FSOC) terminals. It will
simulate optical beacons and disturbances, acquire and track a selected beacon,
control a constrained virtual pan-tilt camera, and generate reproducible
performance reports.

## Current status

Phases 1 through 6 are complete. The repository contains configuration
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
python -m unittest discover -s tests -v
```

Without installing the package:

```bash
PYTHONPATH=src python -m qlyraxis validate configs/scenarios/clear_straight.json
```

## Repository layout

```text
configs/scenarios/       Reproducible SIH benchmark scenarios
docs/                    Architecture, project plan, and UI wireframe
models/                  Portable ONNX verifier and reproducible NumPy weights
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

See `docs/architecture.md`, `docs/project-plan.md`, and
`docs/phase6-testing.md` for the design and local testing workflow.
