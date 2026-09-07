"""Tk desktop dashboard for simulation and recorded-media tracking."""

from __future__ import annotations

import base64
import sys
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from qlyraxis.ai import OnnxCandidateVerifier, VerifiedBeaconDetector
from qlyraxis.config import load_scenario
from qlyraxis.contracts import TrackingState
from qlyraxis.metrics import PerformanceRecorder
from qlyraxis.recorded import RecordedTrackingSystem
from qlyraxis.resources import resource_path
from qlyraxis.sources import open_frame_source
from qlyraxis.tracking import ClosedLoopSystem
from qlyraxis.tracking.visualization import annotate_tracking
from qlyraxis.vision import BeaconDetector


class QlyraxisApp:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.root.title("Qlyraxis FSOC Tracking Laboratory")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#071018")
        self._configure_style()

        self.system: ClosedLoopSystem | RecordedTrackingSystem | None = None
        self.source = None
        self.recorder: PerformanceRecorder | None = None
        self.running = False
        self.mode = "simulation"
        self.photo = None
        self.after_id = None
        self.scenario_paths = self._discover_scenarios()

        self.status_var = tk.StringVar(value="READY")
        self.source_var = tk.StringVar(value="Select a scenario and press Start")
        self.state_var = tk.StringVar(value="SEARCH")
        self.frame_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="0.0")
        self.error_var = tk.StringVar(value="N/A")
        self.retention_var = tk.StringVar(value="0.00%")
        self.command_var = tk.StringVar(value="pan +0.00  tilt +0.00 deg/s")
        self.atmosphere_var = tk.StringVar(value="clear")
        self.noise_var = tk.DoubleVar(value=0)
        self.jitter_var = tk.DoubleVar(value=0)
        self.turbulence_var = tk.DoubleVar(value=0)
        first_scenario = next(iter(self.scenario_paths), "")
        self.scenario_var = tk.StringVar(value=first_scenario)

        self._build_layout()
        self._load_scenario_controls()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_style(self) -> None:
        style = self.ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#0d1822", foreground="#e9f1f7")
        style.configure("TFrame", background="#0d1822")
        style.configure("Panel.TFrame", background="#111f2b")
        style.configure("TLabel", background="#0d1822", foreground="#d8e5ee")
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Muted.TLabel", foreground="#8da5b7")
        style.configure("Metric.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Accent.TButton", background="#16b8a6", foreground="#061311")
        style.map("Accent.TButton", background=[("active", "#29d3bf")])
        style.configure(
            "TCombobox",
            fieldbackground="#172835",
            background="#172835",
            foreground="#e9f1f7",
        )

    def _discover_scenarios(self) -> dict[str, Path]:
        root = resource_path("configs/scenarios")
        return {path.stem: path for path in sorted(root.glob("*.json"))}

    def _build_layout(self) -> None:
        header = self.ttk.Frame(self.root, padding=(22, 16))
        header.pack(fill="x")
        self.ttk.Label(header, text="QLYRAXIS", style="Title.TLabel").pack(side="left")
        self.ttk.Label(
            header,
            text="FSOC coarse alignment laboratory",
            style="Muted.TLabel",
        ).pack(side="left", padx=18)
        self.ttk.Label(header, textvariable=self.status_var).pack(side="right")

        body = self.ttk.Frame(self.root, padding=(20, 0, 20, 18))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        visual = self.ttk.Frame(body, style="Panel.TFrame", padding=12)
        visual.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        visual.rowconfigure(1, weight=1)
        visual.columnconfigure(0, weight=1)
        self.ttk.Label(
            visual, textvariable=self.source_var, style="Muted.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.feed_label = self.tk.Label(
            visual,
            text="Camera feed appears here",
            bg="#03080c",
            fg="#60788a",
            font=("TkDefaultFont", 14),
        )
        self.feed_label.grid(row=1, column=0, sticky="nsew")

        panel = self.ttk.Frame(body, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self._build_controls(panel)
        self._build_metrics(panel)

    def _build_controls(self, panel) -> None:
        self.ttk.Label(panel, text="Scenario").pack(anchor="w")
        selector = self.ttk.Combobox(
            panel,
            textvariable=self.scenario_var,
            values=tuple(self.scenario_paths),
            state="readonly",
        )
        selector.pack(fill="x", pady=(5, 12))
        selector.bind("<<ComboboxSelected>>", self._scenario_selected)

        disturbances = self.ttk.LabelFrame(panel, text="Disturbance overrides", padding=10)
        disturbances.pack(fill="x", pady=(0, 12))
        atmosphere = self.ttk.Combobox(
            disturbances,
            textvariable=self.atmosphere_var,
            values=("clear", "haze", "fog", "rain", "low_light"),
            state="readonly",
            width=12,
        )
        self._parameter_row(disturbances, 0, "Atmosphere", atmosphere)
        self._parameter_row(
            disturbances,
            1,
            "Noise sigma",
            self.ttk.Spinbox(
                disturbances, from_=0, to=20, increment=1, textvariable=self.noise_var
            ),
        )
        self._parameter_row(
            disturbances,
            2,
            "Jitter px",
            self.ttk.Spinbox(
                disturbances, from_=0, to=20, increment=1, textvariable=self.jitter_var
            ),
        )
        self._parameter_row(
            disturbances,
            3,
            "Turbulence px",
            self.ttk.Spinbox(
                disturbances,
                from_=0,
                to=20,
                increment=0.5,
                textvariable=self.turbulence_var,
            ),
        )

        row = self.ttk.Frame(panel, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 12))
        self.start_button = self.ttk.Button(
            row, text="Start", style="Accent.TButton", command=self.toggle
        )
        self.start_button.pack(side="left", expand=True, fill="x")
        self.ttk.Button(row, text="Reset", command=self.reset).pack(
            side="left", expand=True, fill="x", padx=8
        )
        self.ttk.Button(row, text="Open recording", command=self.open_recording).pack(
            side="left", expand=True, fill="x"
        )
        self.ttk.Button(panel, text="Export performance report", command=self.export).pack(
            fill="x", pady=(0, 18)
        )

    def _build_metrics(self, panel) -> None:
        self.ttk.Label(panel, text="Live telemetry", style="Metric.TLabel").pack(
            anchor="w", pady=(0, 9)
        )
        grid = self.ttk.Frame(panel, style="Panel.TFrame")
        grid.pack(fill="x")
        for row, (label, variable) in enumerate(
            (
                ("Tracker state", self.state_var),
                ("Frame", self.frame_var),
                ("Pipeline FPS", self.fps_var),
                ("Centroid error", self.error_var),
                ("Lock retention", self.retention_var),
            )
        ):
            self.ttk.Label(grid, text=label, style="Muted.TLabel").grid(
                row=row, column=0, sticky="w", pady=4
            )
            self.ttk.Label(grid, textvariable=variable).grid(
                row=row, column=1, sticky="e", pady=4
            )
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        self.ttk.Label(
            panel, textvariable=self.command_var, style="Muted.TLabel"
        ).pack(anchor="w", pady=(14, 8))
        self.chart = self.tk.Canvas(
            panel,
            height=190,
            background="#081018",
            highlightthickness=1,
            highlightbackground="#263948",
        )
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

    def _parameter_row(self, parent, row: int, label: str, widget) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        widget.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        parent.columnconfigure(1, weight=1)

    def _scenario_selected(self, _event=None) -> None:
        self.reset()
        self._load_scenario_controls()

    def _load_scenario_controls(self) -> None:
        path = self.scenario_paths.get(self.scenario_var.get())
        if path is None:
            return
        scenario = load_scenario(path)
        self.atmosphere_var.set(str(scenario.disturbances["atmosphere"]))
        self.noise_var.set(float(scenario.disturbances["noise_std_px"]))
        self.jitter_var.set(float(scenario.disturbances["camera_jitter_max_px_frame"]))
        self.turbulence_var.set(float(scenario.disturbances["turbulence_strength_px"]))

    def _new_simulation(self) -> None:
        name = self.scenario_var.get()
        if name not in self.scenario_paths:
            raise ValueError("no scenario selected")
        scenario = load_scenario(self.scenario_paths[name])
        disturbance_config = deepcopy(scenario.disturbances)
        disturbance_config["atmosphere"] = self.atmosphere_var.get()
        disturbance_config["noise_std_px"] = float(self.noise_var.get())
        disturbance_config["camera_jitter_max_px_frame"] = float(self.jitter_var.get())
        disturbance_config["turbulence_strength_px"] = float(self.turbulence_var.get())
        if disturbance_config["noise_std_px"] > 0 and not disturbance_config["noise"]:
            disturbance_config["noise"] = ["gaussian"]
        scenario = replace(scenario, disturbances=disturbance_config)
        self.system = ClosedLoopSystem.from_scenario(scenario)
        self.source = None
        self.mode = "simulation"
        self.recorder = PerformanceRecorder(
            scenario.name,
            tuple(float(value) for value in scenario.camera["viewport_px"]),
            configuration=asdict(scenario),
        )
        self.source_var.set(f"SIMULATION  {scenario.name}")

    def open_recording(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.askopenfilename(
            title="Open a recorded camera feed",
            filetypes=[("Video and images", "*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg")],
        )
        if not path:
            return
        self.reset(clear_display=False)
        try:
            self.source = open_frame_source(path)
            detector: Any = BeaconDetector()
            model = resource_path("models/beacon_verifier.onnx")
            if model.exists():
                detector = VerifiedBeaconDetector(OnnxCandidateVerifier(model), detector)
            self.system = RecordedTrackingSystem(detector)
            self.mode = "recorded"
            self.recorder = PerformanceRecorder(
                Path(path).stem,
                (float(self.source.width), float(self.source.height)),
                configuration={"input": str(path), "ground_truth": False},
            )
            self.source_var.set(f"RECORDED  {Path(path).name}")
            self.status_var.set("LOADED")
        except Exception as exc:
            messagebox.showerror("Could not open recording", str(exc))
            self.source = None
            self.system = None

    def toggle(self) -> None:
        from tkinter import messagebox

        if self.running:
            self.running = False
            self.start_button.configure(text="Resume")
            self.status_var.set("PAUSED")
            return
        try:
            if self.system is None:
                self._new_simulation()
        except Exception as exc:
            messagebox.showerror("Could not start", str(exc))
            return
        self.running = True
        self.start_button.configure(text="Pause")
        self.status_var.set("RUNNING")
        self._tick()

    def _tick(self) -> None:
        if not self.running or self.system is None or self.recorder is None:
            return
        started = perf_counter()
        if self.mode == "simulation":
            result = self.system.step()
            frame = result.simulation.frame
            selected = self.system.tracker.selected_detection
            truth = result.simulation.target_sensor_positions[0]
            command = result.next_command
        else:
            frame = self.source.read()
            if frame is None:
                self.running = False
                self.start_button.configure(text="Start")
                self.status_var.set("COMPLETE")
                return
            result = self.system.step(frame)
            selected = self.system.tracker.selected_detection
            truth = None
            command = None
        process_ms = (perf_counter() - started) * 1000.0
        self.recorder.record(
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            state=result.state,
            detections=result.detections,
            selected=selected,
            estimate=result.estimate,
            command=command,
            truth=truth,
            processing_time_ms=process_ms,
        )
        annotated = annotate_tracking(
            frame.image,
            result.detections,
            result.estimate,
            result.state,
            result.search_scope,
            command,
        )
        self._show_frame(annotated)
        self._update_telemetry(result, command)
        delay_ms = max(1, round(1000 / 30 - process_ms))
        self.after_id = self.root.after(delay_ms, self._tick)

    def _show_frame(self, image) -> None:
        success, encoded = cv2.imencode(".png", image)
        if not success:
            return
        data = base64.b64encode(encoded.tobytes())
        self.photo = self.tk.PhotoImage(data=data)
        self.feed_label.configure(image=self.photo, text="")

    def _update_telemetry(self, result, command) -> None:
        assert self.recorder is not None
        summary = self.recorder.summary()
        latest = self.recorder.rows[-1]
        self.state_var.set(result.state.value.upper())
        self.frame_var.set(str(latest["frame_index"]))
        self.fps_var.set(f"{summary.processing_fps:.1f}")
        error = latest["tracking_error_px"]
        self.error_var.set("N/A" if error is None else f"{float(error):.3f} px")
        self.retention_var.set(f"{summary.lock_retention_percent:.2f}%")
        self.command_var.set(
            "recorded input"
            if command is None
            else (
                f"pan {command.pan_rate_deg_s:+.2f}  "
                f"tilt {command.tilt_rate_deg_s:+.2f} deg/s"
            )
        )
        self._draw_chart()

    def _draw_chart(self) -> None:
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 20)
        height = max(self.chart.winfo_height(), 20)
        self.chart.create_text(
            12, 12, text="TRACKING ERROR HISTORY", anchor="nw", fill="#8199ab"
        )
        if self.recorder is None:
            return
        values = [
            float(row["tracking_error_px"])
            for row in self.recorder.rows[-180:]
            if row["tracking_error_px"] is not None
        ]
        points = scale_series(values, width, height, padding=18)
        if len(points) >= 2:
            flat = [coordinate for point in points for coordinate in point]
            self.chart.create_line(*flat, fill="#36d6c2", width=2, smooth=True)

    def export(self) -> None:
        from tkinter import filedialog, messagebox

        if self.recorder is None or not self.recorder.rows:
            messagebox.showinfo("No data", "Run a scenario or recording before exporting.")
            return
        directory = filedialog.askdirectory(title="Choose report directory")
        if not directory:
            return
        paths = self.recorder.export(directory)
        messagebox.showinfo(
            "Report exported",
            "\n".join(f"{kind.upper()}: {path}" for kind, path in paths.items()),
        )

    def reset(self, clear_display: bool = True) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.source is not None:
            self.source.close()
        self.source = None
        self.system = None
        self.recorder = None
        self.start_button.configure(text="Start")
        self.status_var.set("READY")
        self.state_var.set("SEARCH")
        self.frame_var.set("0")
        self.fps_var.set("0.0")
        self.error_var.set("N/A")
        self.retention_var.set("0.00%")
        self.command_var.set("pan +0.00  tilt +0.00 deg/s")
        self.chart.delete("all")
        if clear_display:
            self.photo = None
            self.feed_label.configure(image="", text="Camera feed appears here")
            self.source_var.set("Select a scenario and press Start")

    def close(self) -> None:
        self.reset(clear_display=False)
        self.root.destroy()


def scale_series(
    values: list[float], width: int, height: int, padding: int = 10
) -> list[tuple[float, float]]:
    if not values or width <= 2 * padding or height <= 2 * padding:
        return []
    maximum = max(max(values), 1.0)
    return [
        (
            padding + index * (width - 2 * padding) / max(len(values) - 1, 1),
            height - padding - value / maximum * (height - 2 * padding),
        )
        for index, value in enumerate(values)
    ]


def launch() -> int:
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Could not open the desktop display: {exc}", file=sys.stderr)
        return 1
    QlyraxisApp(root)
    root.mainloop()
    return 0
