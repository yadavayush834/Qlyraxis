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
| TRACK | SEARCH: OFF | CONF: -- | CAMERA OFFSET: -- px | STRICT LOCK: --  |
+--------------------------------------------------------------------------+
| camera-offset/time chart            | pan/tilt command                    |
+--------------------------------------------------------------------------+
| [Pause] [Reset] [Inject dropout] [Export run]                            |
+--------------------------------------------------------------------------+
```

## Screen 3: Benchmark results

```text
+--------------------------------------------------------------------------+
| Scenario | mean offset | P95 offset | strict lock | acquisition | FPS      |
+--------------------------------------------------------------------------+
| ...                                                                      |
+--------------------------------------------------------------------------+
| [Baseline vs improved graph] [Export CSV] [Export JSON] [Generate report]|
+--------------------------------------------------------------------------+
```

The first implementation should prioritize readable telemetry and evaluator
workflow over visual effects.
