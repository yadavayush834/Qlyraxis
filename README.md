# Qlyraxis

Qlyraxis is an AI-assisted virtual camera tracking laboratory for coarse
alignment of mobile Free Space Optical Communication (FSOC) terminals. It will
simulate optical beacons and disturbances, acquire and track a selected beacon,
control a constrained virtual pan-tilt camera, and generate reproducible
performance reports.

## Current status

Phases 1 and 2 are complete. The repository contains configuration validation,
stable module contracts, architecture and UI documents, five benchmark scenario
definitions, and a deterministic virtual environment with camera dynamics and
all four mandatory target trajectories. Computer vision starts in Phase 3.

## Quick start

Python 3.11 or newer is required.

```bash
python -m pip install -e .
qlyraxis validate configs/scenarios/clear_straight.json
qlyraxis show configs/scenarios/clear_straight.json
qlyraxis simulate configs/scenarios/clear_straight.json --frames 90
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
src/qlyraxis/            Application package and cross-module contracts
tests/                   Configuration contract tests
```

## Non-negotiable design rules

1. The tracker receives rendered frames only, never simulator ground truth.
2. Simulated video, MP4 files, and future camera input use one `FrameSource`
   interface.
3. Every run stores its full configuration and random seed.
4. Control commands respect pan, tilt, acceleration, and update-rate limits.
5. Performance claims must be generated automatically from recorded runs.

See `docs/architecture.md` and `docs/project-plan.md` for the agreed design.
