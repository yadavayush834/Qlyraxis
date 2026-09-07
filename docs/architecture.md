# Qlyraxis architecture

## Objective

Autonomously acquire and continuously track a small optical beacon while
controlling a constrained virtual pan-tilt camera. The same tracking pipeline
must operate on simulated frames and evaluator-provided MP4 files.

## System context

```text
Scenario config
      |
      v
Motion model ---> Scene state ---> Renderer ---> Disturbances ---> FrameSource
                       |                                      |
                       +----> Ground-truth channel            v
                                                        Detector
                                                            |
                                                            v
                                                        Estimator
                                                            |
                                                            v
                                                        Controller
                                                            |
                                                            v
                                                     Virtual camera

Ground truth + detections + state + commands ----------> Metrics/reporting
```

The ground-truth channel is visible only to metrics and visualization. Detector,
estimator, and controller modules must not import simulator state.

## Runtime pipeline

1. A `FrameSource` supplies an image and timestamp.
2. A `Detector` returns zero or more beacon candidates.
3. A `Tracker` filters detections and predicts target motion.
4. A state machine selects SEARCH, ACQUIRE, TRACK, COAST, or REACQUIRE behavior.
5. A `Controller` converts image-plane error into constrained pan/tilt commands.
6. A `MetricsSink` records inputs, estimates, commands, timing, and optional truth.

## Module boundaries

| Package | Responsibility | Phase |
|---|---|---:|
| `config` | Load and validate scenarios | 1 |
| `contracts` | Shared immutable data and protocols | 1 |
| `simulation` | World, target motion, camera dynamics | 2 |
| `vision` | Acquisition, candidate verification, centroiding | 3 |
| `tracking` | Filtering, prediction, state machine | 4 |
| `control` | PID and optional predictive control | 4 |
| `disturbances` | Seeded noise, atmosphere, blur, jitter, motion, and dropout | 5 |
| `sources` | Simulation, MP4, image sequence, camera input | 6 |
| `ai` | Candidate classifier and ONNX inference | 6 |
| `metrics` | Live metrics and performance reports | 7 |
| `ui` | PySide6 desktop application | 7 |

## Configuration contract

Scenario JSON contains five top-level sections:

- `name` and `description`
- `camera`: virtual world, viewport, FOV, rate, and motion limits
- `target`: count, shape, size, starting position, and trajectory
- `disturbances`: noise, atmosphere, jitter, dropout, and platform motion
- `evaluation`: duration, deterministic seed, and logging options

## State machine

```text
SEARCH --candidate--> ACQUIRE --confirmed--> TRACK
  ^                       |                     |
  |                       +--rejected----------+
  |                                             |
  +--timeout-- REACQUIRE <--lost-- COAST <------+
                  |                  |
                  +--candidate-------+
```

The exact confidence thresholds and timeouts will be configuration values once
the detection baseline exists.

## Target performance envelope

| Metric | Official minimum | Engineering target |
|---|---:|---:|
| Acquisition time | <= 2 s | < 1 s |
| Re-acquisition time | <= 1 s | < 0.5 s |
| Tracking error | <= 10 px | mean < 4 px; P95 < 8 px |
| Target loss | < 5% | < 2% |
| Processing speed | >= 20 FPS | >= 40 FPS on reference CPU |
| Camera update rate | >= 30 Hz | 30-60 Hz |
| Control update rate | >= 20 Hz | 30 Hz |

## Planned technology

- Python 3.11, NumPy, SciPy, and OpenCV
- PySide6 and PyQtGraph for the desktop interface
- PyTorch for training; ONNX Runtime for deployment
- Pandas and Matplotlib for reporting
- PyInstaller for the standalone executable
