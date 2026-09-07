# Phase plan

## Phase 1 - Planning and setup

Status: complete

- [x] Define architecture and module boundaries
- [x] Define shared runtime data contracts
- [x] Define and validate scenario configuration
- [x] Create five reproducible baseline scenarios
- [x] Create UI wireframes
- [x] Add a runnable CLI and automated configuration tests

Exit criteria: all included scenario files validate and all Phase 1 tests pass.

## Phase 2 - Virtual environment and camera

Status: complete

- [x] Deterministic 2D scene clock
- [x] 2000 x 2000 world and 640 x 480 viewport
- [x] Beacon rendering
- [x] Straight, circular, figure-eight, and random trajectories
- [x] Constrained pan and tilt camera dynamics
- [x] Unit tests for trajectory and camera limits

Exit criteria: every mandatory trajectory renders repeatably and camera commands
cannot exceed configured physical limits.

## Phase 3 - Detection and acquisition

Status: complete

- [x] Preprocessing and adaptive thresholding
- [x] Multi-scale candidate generation
- [x] Blob filtering and centroiding
- [x] Confidence score and acquisition state

Exit criteria: acquire every clear baseline target in less than two seconds.

## Phase 4 - Tracking and control

Status: complete

- [x] Kalman motion estimation
- [x] SEARCH/ACQUIRE/TRACK/COAST/REACQUIRE state machine
- [x] PID pan/tilt control
- [x] Local and global re-acquisition

Exit criteria: clear scenarios meet official error, loss, and speed thresholds.

## Phase 5 - Disturbances

Status: complete

- [x] Gaussian, Poisson, and salt-and-pepper sensor noise
- [x] Haze, fog, rain, low light, defocus, and motion blur
- [x] Camera jitter, platform motion, atmospheric warp, and timed dropout
- [x] Configurable severity values and deterministic replay by seed/frame
- [x] Clean/disturbed frame comparison and disturbance-aware ground truth

Exit criteria: every disturbance is independently selectable and reproducible.

## Phase 6 - MP4 input and AI verification

Status: complete

- [x] MP4/image-sequence `FrameSource`
- [x] Deterministic synthetic candidate-patch dataset
- [x] Tiny convolutional candidate verifier and training command
- [x] ONNX export and CPU inference through OpenCV/ONNX Runtime
- [x] Ground-truth-free recorded-media analysis command

Exit criteria: process an unseen MP4 without simulator state or ground truth.

## Phase 7 - GUI, metrics, packaging, and delivery

Status: complete

- [x] Tk desktop application
- [x] Live telemetry and error-history chart
- [x] CSV, JSON, and self-contained HTML reports
- [x] Standalone PyInstaller application bundle
- [x] User manual, 13-page technical report, and benchmark evidence

Exit criteria: a clean computer can run the packaged application and reproduce
the submitted benchmark report.
