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
5. A predictive controller combines image-plane error, target-velocity
   feed-forward, and measured camera rate into constrained pan/tilt commands.
6. A `MetricsSink` records inputs, estimates, commands, timing, and optional truth.

## Module boundaries

| Package | Responsibility | Phase |
|---|---|---:|
| `config` | Load and validate scenarios | 1 |
| `contracts` | Shared immutable data and protocols | 1 |
| `simulation` | World, target motion, camera dynamics | 2 |
| `vision` | Acquisition, candidate verification, centroiding, CodeLock identity | 3/9 |
| `tracking` | Filtering, prediction, state machine | 4 |
| `control` | PID and optional predictive control | 4 |
| `disturbances` | Seeded noise, atmosphere, blur, jitter, motion, and dropout | 5 |
| `sources` | Simulation, MP4, and naturally sorted image sequences | 6 |
| `ai` | Synthetic patches, tiny convolutional classifier, ONNX inference | 6 |
| `metrics` | Per-frame metrics and CSV, JSON, and HTML reports | 7 |
| `ui` | Tk desktop application and live charts | 7 |
| `resources` | Source and packaged-resource resolution | 7 |

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

Confidence thresholds and timeouts are configuration values shared by simulated
and recorded inputs. A direction-change detector compares measured and predicted
velocity and adapts the Kalman state immediately after a maneuver.

## Target performance envelope

| Metric | Official minimum | Engineering target |
|---|---:|---:|
| Acquisition time | <= 2 s | < 1 s |
| Re-acquisition time | <= 1 s | < 0.5 s |
| Camera pointing offset | <= 10 px | minimize mean and P95 |
| Target loss | < 5% | < 2% |
| Processing speed | >= 20 FPS | >= 40 FPS on reference CPU |
| Camera update rate | >= 30 Hz | 30-60 Hz |
| Control update rate | >= 20 Hz | 30 Hz |

## Implemented technology

- Python 3.11, NumPy, and OpenCV
- Tk for the desktop interface and canvas-based charts
- A NumPy-trained compact CNN exported to ONNX
- OpenCV DNN by default and optional ONNX Runtime for CPU inference
- Standard-library CSV, JSON, and self-contained HTML reporting
- Deterministic baseline-versus-improved A/B reports with overlaid offset curves
- Phase-synchronized temporal beacon-code correlation with decoy rejection
- PyInstaller for the standalone application bundle
