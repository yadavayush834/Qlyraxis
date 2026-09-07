# Desktop UI wireframe

## Screen 1: Scenario setup

```text
+--------------------------------------------------------------------------+
| Qlyraxis                                                   [Load] [Save] |
+----------------------+---------------------------------------------------+
| Camera               | Scenario preview                                  |
| World size           |                                                   |
| Viewport / FOV       |             world + camera rectangle              |
| Pan/tilt limits      |                                                   |
|                      |                                                   |
| Target               |                                                   |
| Shape / size         |                                                   |
| Trajectory / speed   |                                                   |
|                      |                                                   |
| Disturbances         |                                                   |
| Noise / weather      |                                                   |
| Jitter / dropout     |                                                   |
+----------------------+---------------------------------------------------+
| [Validate] [Start simulation] [Open MP4 benchmark]                       |
+--------------------------------------------------------------------------+
```

## Screen 2: Live tracking

```text
+--------------------------------+-----------------------------------------+
| Complete world                 | Virtual camera feed                     |
| target + camera FOV            | detection + centroid + prediction       |
+--------------------------------+-----------------------------------------+
| LOCK: TRACKING | FPS: -- | error: -- px | pan: -- | tilt: --             |
+--------------------------------------------------------------------------+
| error/time chart                    | confidence/time chart               |
+--------------------------------------------------------------------------+
| [Pause] [Reset] [Inject dropout] [Export run]                            |
+--------------------------------------------------------------------------+
```

## Screen 3: Benchmark results

```text
+--------------------------------------------------------------------------+
| Scenario / video | acquisition | reacquisition | RMSE | loss | FPS | pass |
+--------------------------------------------------------------------------+
| ...                                                                      |
+--------------------------------------------------------------------------+
| [Replay failure] [Export CSV] [Export JSON] [Generate report]            |
+--------------------------------------------------------------------------+
```

The first implementation should prioritize readable telemetry and evaluator
workflow over visual effects.
