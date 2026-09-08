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
import numpy as np

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
        self.root.geometry("1360x840")
        self.root.minsize(1120, 720)
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
        self.control_rate_var = tk.StringVar(value="30 Hz")
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
        font_family = "DejaVu Sans"
        input_background = "#172835"
        input_disabled = "#101c26"
        input_foreground = "#f4f8fb"
        style.configure(
            ".",
            background="#0d1822",
            foreground="#e9f1f7",
            font=(font_family, 11),
        )
        style.configure("TFrame", background="#0d1822")
        style.configure("Panel.TFrame", background="#111f2b")
        style.configure("TLabel", background="#0d1822", foreground="#d8e5ee")
        style.configure(
            "Title.TLabel",
            font=(font_family, 30, "bold"),
            foreground="#f5fbff",
        )
        style.configure(
            "Eyebrow.TLabel",
            font=(font_family, 10, "bold"),
            foreground="#39d8c4",
        )
        style.configure("Muted.TLabel", foreground="#9db2c1")
        style.configure(
            "Section.TLabel",
            font=(font_family, 14, "bold"),
            foreground="#f5fbff",
        )
        style.configure(
            "Source.TLabel",
            font=(font_family, 11, "bold"),
            foreground="#b8cad6",
        )
        style.configure(
            "TButton",
            font=(font_family, 11, "bold"),
            padding=(12, 9),
            background="#213544",
            foreground="#edf7fc",
            bordercolor="#486274",
            lightcolor="#486274",
            darkcolor="#486274",
        )
        style.map(
            "TButton",
            background=[("active", "#2b4658"), ("pressed", "#172835")],
            foreground=[("disabled", "#718696"), ("!disabled", "#edf7fc")],
        )
        style.configure(
            "Accent.TButton",
            font=(font_family, 12, "bold"),
            background="#20c7b3",
            foreground="#041411",
            bordercolor="#58ead8",
            lightcolor="#58ead8",
            darkcolor="#0b8f80",
            padding=(14, 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#48ddcb"), ("pressed", "#10a996")],
        )
        style.configure(
            "Export.TButton",
            background="#243a4a",
            foreground="#d8e8f2",
            padding=(12, 9),
        )
        style.configure(
            "TLabelframe",
            background="#111f2b",
            bordercolor="#385367",
            lightcolor="#385367",
            darkcolor="#385367",
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background="#111f2b",
            foreground="#b9ccd8",
            font=(font_family, 10, "bold"),
        )
        for widget_style in ("Dark.TCombobox", "Dark.TSpinbox"):
            style.configure(
                widget_style,
                font=(font_family, 11),
                fieldbackground=input_background,
                background="#243746",
                foreground=input_foreground,
                arrowcolor=input_foreground,
                bordercolor="#4d687a",
                lightcolor="#4d687a",
                darkcolor="#4d687a",
                insertcolor=input_foreground,
                padding=(8, 6),
            )
            style.map(
                widget_style,
                fieldbackground=[
                    ("disabled", input_disabled),
                    ("readonly", input_background),
                    ("!disabled", input_background),
                ],
                foreground=[
                    ("disabled", "#8da5b7"),
                    ("readonly", input_foreground),
                    ("!disabled", input_foreground),
                ],
                selectbackground=[
                    ("readonly", input_background),
                    ("!disabled", "#245669"),
                ],
                selectforeground=[
                    ("readonly", input_foreground),
                    ("!disabled", input_foreground),
                ],
                arrowcolor=[
                    ("disabled", "#60788a"),
                    ("!disabled", input_foreground),
                ],
            )
        self.root.option_add("*TCombobox*Listbox.background", input_background)
        self.root.option_add("*TCombobox*Listbox.foreground", input_foreground)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#245669")
        self.root.option_add("*TCombobox*Listbox.selectForeground", input_foreground)

    def _discover_scenarios(self) -> dict[str, Path]:
        root = resource_path("configs/scenarios")
        return {path.stem: path for path in sorted(root.glob("*.json"))}

    def _create_brand_image(self):
        """Render a crisp title even on systems exposing only Tk's tiny bitmap font."""
        image = np.full((48, 270, 3), (34, 24, 13), dtype=np.uint8)
        cv2.putText(
            image,
            "QLYRAXIS",
            (0, 34),
            cv2.FONT_HERSHEY_DUPLEX,
            1.05,
            (255, 251, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.line(image, (1, 44), (205, 44), (180, 199, 32), 2, cv2.LINE_AA)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            return None
        return self.tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))

    def _build_layout(self) -> None:
        header = self.ttk.Frame(self.root, padding=(24, 16, 24, 14))
        header.pack(fill="x")
        status_stack = self.tk.Frame(header, bg="#0d1822")
        status_stack.pack(side="right")
        self.tk.Label(
            status_stack,
            text="SYSTEM STATUS",
            bg="#0d1822",
            fg="#9db2c1",
            font=("DejaVu Sans", 9, "bold"),
        ).pack(anchor="e", pady=(0, 4))
        title_stack = self.tk.Frame(header, bg="#0d1822")
        title_stack.pack(side="left", fill="x", expand=True)
        self.brand_photo = self._create_brand_image()
        self.tk.Label(
            title_stack,
            text="QLYRAXIS" if self.brand_photo is None else "",
            image=self.brand_photo,
            bg="#0d1822",
            fg="#f5fbff",
            font=("DejaVu Sans", 30, "bold"),
        ).pack(anchor="w")
        self.tk.Label(
            title_stack,
            text="AI-ASSISTED OPTICAL POINTING  •  ACQUISITION  •  TRACKING",
            bg="#0d1822",
            fg="#39d8c4",
            font=("DejaVu Sans", 10, "bold"),
        ).pack(anchor="w", pady=(1, 0))
        self.status_badge = self.tk.Label(
            status_stack,
            textvariable=self.status_var,
            bg="#17362f",
            fg="#6ff1cf",
            font=("DejaVu Sans", 11, "bold"),
            padx=16,
            pady=5,
        )
        self.status_badge.pack(anchor="e")

        body = self.ttk.Frame(self.root, padding=(20, 0, 20, 20))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=7)
        body.columnconfigure(1, weight=4, minsize=410)
        body.rowconfigure(0, weight=1)

        visual = self.ttk.Frame(body, style="Panel.TFrame", padding=14)
        visual.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        visual.rowconfigure(1, weight=1)
        visual.columnconfigure(0, weight=1)
        feed_header = self.ttk.Frame(visual, style="Panel.TFrame")
        feed_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.ttk.Label(
            feed_header, text="LIVE OPTICAL FEED", style="Section.TLabel"
        ).pack(side="left")
        self.ttk.Label(
            feed_header, textvariable=self.source_var, style="Source.TLabel"
        ).pack(side="right")
        self.feed_label = self.tk.Label(
            visual,
            text="OPTICAL FEED STANDBY\n\nSelect a scenario and press Start",
            bg="#03080c",
            fg="#7890a0",
            font=("DejaVu Sans", 15, "bold"),
            justify="center",
            highlightthickness=1,
            highlightbackground="#294153",
        )
        self.feed_label.grid(row=1, column=0, sticky="nsew")
        legend = self.ttk.Frame(visual, style="Panel.TFrame")
        legend.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.tk.Label(
            legend,
            text="●  DETECTION",
            bg="#111f2b",
            fg="#45e0ca",
            font=("DejaVu Sans", 10, "bold"),
        ).pack(side="left")
        self.tk.Label(
            legend,
            text="＋  FILTERED ESTIMATE",
            bg="#111f2b",
            fg="#ffc857",
            font=("DejaVu Sans", 10, "bold"),
        ).pack(side="left", padx=20)
        self.ttk.Label(
            legend, text="30 Hz CLOSED LOOP", style="Muted.TLabel"
        ).pack(side="right")

        panel = self.ttk.Frame(body, style="Panel.TFrame", padding=20)
        panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._build_controls(panel)
        self._build_metrics(panel)

    def _build_controls(self, panel) -> None:
        self.ttk.Label(panel, text="MISSION CONTROL", style="Section.TLabel").pack(
            anchor="w", pady=(0, 14)
        )
        self.ttk.Label(panel, text="SCENARIO", style="Muted.TLabel").pack(anchor="w")
        selector = self.ttk.Combobox(
            panel,
            textvariable=self.scenario_var,
            values=tuple(self.scenario_paths),
            state="readonly",
            style="Dark.TCombobox",
        )
        selector.pack(fill="x", pady=(6, 14))
        selector.bind("<<ComboboxSelected>>", self._scenario_selected)

        disturbances = self.ttk.LabelFrame(
            panel, text="  ENVIRONMENT & DISTURBANCES  ", padding=(12, 10)
        )
        disturbances.pack(fill="x", pady=(0, 14))
        atmosphere = self.ttk.Combobox(
            disturbances,
            textvariable=self.atmosphere_var,
            values=("clear", "haze", "fog", "rain", "low_light"),
            state="readonly",
            width=12,
            style="Dark.TCombobox",
        )
        self._parameter_row(disturbances, 0, "Atmosphere", atmosphere)
        self._parameter_row(
            disturbances,
            1,
            "Noise sigma",
            self.ttk.Spinbox(
                disturbances,
                from_=0,
                to=20,
                increment=1,
                textvariable=self.noise_var,
                style="Dark.TSpinbox",
            ),
        )
        self._parameter_row(
            disturbances,
            2,
            "Jitter px",
            self.ttk.Spinbox(
                disturbances,
                from_=0,
                to=20,
                increment=1,
                textvariable=self.jitter_var,
                style="Dark.TSpinbox",
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
                style="Dark.TSpinbox",
            ),
        )

        row = self.ttk.Frame(panel, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 9))
        self.start_button = self.ttk.Button(
            row, text="Start", style="Accent.TButton", command=self.toggle
        )
        self.start_button.pack(side="left", expand=True, fill="x")
        self.ttk.Button(row, text="Reset", command=self.reset).pack(
            side="left", expand=True, fill="x", padx=(9, 0)
        )
        secondary_row = self.ttk.Frame(panel, style="Panel.TFrame")
        secondary_row.pack(fill="x", pady=(0, 20))
        self.ttk.Button(
            secondary_row, text="Open recording", command=self.open_recording
        ).pack(side="left", expand=True, fill="x")
        self.ttk.Button(
            secondary_row,
            text="Export report",
            command=self.export,
            style="Export.TButton",
        ).pack(side="left", expand=True, fill="x", padx=(9, 0))

    def _metric_card(self, parent, row: int, column: int, label: str, variable):
        card = self.tk.Frame(
            parent,
            bg="#172835",
            highlightthickness=1,
            highlightbackground="#29485a",
            padx=12,
            pady=8,
        )
        card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        self.tk.Label(
            card,
            text=label.upper(),
            bg="#172835",
            fg="#8fa8b8",
            font=("DejaVu Sans", 9, "bold"),
        ).pack(anchor="w")
        value_label = self.tk.Label(
            card,
            textvariable=variable,
            bg="#172835",
            fg="#f4f9fc",
            font=("DejaVu Sans", 18, "bold"),
            pady=2,
        )
        value_label.pack(anchor="w")
        return value_label

    def _build_metrics(self, panel) -> None:
        heading = self.ttk.Frame(panel, style="Panel.TFrame")
        heading.pack(fill="x", pady=(0, 5))
        self.ttk.Label(heading, text="LIVE TELEMETRY", style="Section.TLabel").pack(
            side="left"
        )
        self.ttk.Label(heading, text="REAL TIME", style="Eyebrow.TLabel").pack(
            side="right"
        )
        grid = self.ttk.Frame(panel, style="Panel.TFrame")
        grid.pack(fill="x")
        self.state_value_label = self._metric_card(
            grid, 0, 0, "Tracker state", self.state_var
        )
        self._metric_card(grid, 0, 1, "Frame", self.frame_var)
        self._metric_card(grid, 1, 0, "Pipeline FPS", self.fps_var)
        self._metric_card(grid, 1, 1, "Centroid error", self.error_var)
        self._metric_card(grid, 2, 0, "Lock retention", self.retention_var)
        self._metric_card(grid, 2, 1, "Control loop", self.control_rate_var)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        command_panel = self.tk.Frame(
            panel,
            bg="#0b151e",
            highlightthickness=1,
            highlightbackground="#263f50",
            padx=10,
            pady=7,
        )
        command_panel.pack(fill="x", pady=(10, 9))
        self.tk.Label(
            command_panel,
            text="CAMERA COMMAND",
            bg="#0b151e",
            fg="#829aaa",
            font=("DejaVu Sans", 9, "bold"),
        ).pack(side="left")
        self.tk.Label(
            command_panel,
            textvariable=self.command_var,
            bg="#0b151e",
            fg="#d8e7ef",
            font=("DejaVu Sans Mono", 10, "bold"),
        ).pack(side="right")
        self.chart = self.tk.Canvas(
            panel,
            height=150,
            background="#081018",
            highlightthickness=1,
            highlightbackground="#263948",
        )
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

    def _parameter_row(self, parent, row: int, label: str, widget) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        widget.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
        parent.columnconfigure(1, weight=1)

    def _scenario_selected(self, _event=None) -> None:
        self.reset()
        self._load_scenario_controls()

    def _set_status(self, status: str) -> None:
        palette = {
            "READY": ("#17362f", "#6ff1cf"),
            "LOADED": ("#173047", "#7cc8ff"),
            "RUNNING": ("#123e38", "#75f3d4"),
            "PAUSED": ("#463717", "#ffd36a"),
            "COMPLETE": ("#2c2851", "#c9b8ff"),
        }
        background, foreground = palette.get(status, ("#293642", "#dbe8ef"))
        self.status_var.set(status)
        self.status_badge.configure(bg=background, fg=foreground)

    def _set_tracker_state(self, state: str) -> None:
        colors = {
            "SEARCH": "#7cc8ff",
            "ACQUIRE": "#ffd36a",
            "TRACK": "#69edcb",
            "COAST": "#ffb45f",
            "REACQUIRE": "#ff8290",
        }
        self.state_var.set(state)
        self.state_value_label.configure(fg=colors.get(state, "#f4f9fc"))

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
            self._set_status("LOADED")
        except Exception as exc:
            messagebox.showerror("Could not open recording", str(exc))
            self.source = None
            self.system = None

    def toggle(self) -> None:
        from tkinter import messagebox

        if self.running:
            self.running = False
            self.start_button.configure(text="Resume")
            self._set_status("PAUSED")
            return
        try:
            if self.system is None:
                self._new_simulation()
        except Exception as exc:
            messagebox.showerror("Could not start", str(exc))
            return
        self.running = True
        self.start_button.configure(text="Pause")
        self._set_status("RUNNING")
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
                self._set_status("COMPLETE")
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
        available_width = max(self.feed_label.winfo_width() - 2, 1)
        available_height = max(self.feed_label.winfo_height() - 2, 1)
        image_height, image_width = image.shape[:2]
        if available_width > 100 and available_height > 100:
            scale = min(
                available_width / image_width,
                available_height / image_height,
            )
            display_width = max(1, round(image_width * scale))
            display_height = max(1, round(image_height * scale))
            if (display_width, display_height) != (image_width, image_height):
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                image = cv2.resize(
                    image,
                    (display_width, display_height),
                    interpolation=interpolation,
                )
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
        self._set_tracker_state(result.state.value.upper())
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
            14,
            12,
            text="TRACKING ERROR • LAST 180 FRAMES",
            anchor="nw",
            fill="#9bb0be",
            font=("DejaVu Sans", 8, "bold"),
        )
        self.chart.create_text(
            width - 14,
            12,
            text="TARGET ≤ 10 PX",
            anchor="ne",
            fill="#39dac6",
            font=("DejaVu Sans", 8, "bold"),
        )
        if self.recorder is None:
            self.chart.create_text(
                width / 2,
                height / 2,
                text="Awaiting telemetry",
                fill="#506879",
                font=("DejaVu Sans", 10),
            )
            return
        values = [
            float(row["tracking_error_px"])
            for row in self.recorder.rows[-180:]
            if row["tracking_error_px"] is not None
        ]
        chart_maximum = max(max(values, default=0.0), 10.0)
        plot_top = 34
        plot_bottom = max(plot_top + 10, height - 20)
        for fraction in (0.0, 0.5, 1.0):
            y = plot_top + fraction * (plot_bottom - plot_top)
            self.chart.create_line(
                28, y, width - 12, y, fill="#172733", dash=(2, 4)
            )
        self.chart.create_text(
            8,
            plot_top,
            text=f"{chart_maximum:.0f}",
            anchor="w",
            fill="#60798a",
            font=("DejaVu Sans", 7),
        )
        self.chart.create_text(
            8,
            plot_bottom,
            text="0",
            anchor="w",
            fill="#60798a",
            font=("DejaVu Sans", 7),
        )
        points = scale_series(
            values,
            width,
            height,
            padding=28,
            maximum_value=chart_maximum,
            top_padding=plot_top,
            bottom_padding=height - plot_bottom,
        )
        if len(points) >= 2:
            flat = [coordinate for point in points for coordinate in point]
            self.chart.create_line(*flat, fill="#39dac6", width=2, smooth=True)
            last_x, last_y = points[-1]
            self.chart.create_oval(
                last_x - 3,
                last_y - 3,
                last_x + 3,
                last_y + 3,
                fill="#84ffeb",
                outline="",
            )

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
        self._set_status("READY")
        self._set_tracker_state("SEARCH")
        self.frame_var.set("0")
        self.fps_var.set("0.0")
        self.error_var.set("N/A")
        self.retention_var.set("0.00%")
        self.command_var.set("pan +0.00  tilt +0.00 deg/s")
        self.chart.delete("all")
        if clear_display:
            self.photo = None
            self.feed_label.configure(
                image="",
                text="OPTICAL FEED STANDBY\n\nSelect a scenario and press Start",
            )
            self.source_var.set("Select a scenario and press Start")

    def close(self) -> None:
        self.reset(clear_display=False)
        self.root.destroy()


def scale_series(
    values: list[float],
    width: int,
    height: int,
    padding: int = 10,
    maximum_value: float | None = None,
    top_padding: int | None = None,
    bottom_padding: int | None = None,
) -> list[tuple[float, float]]:
    top = padding if top_padding is None else top_padding
    bottom = padding if bottom_padding is None else bottom_padding
    if not values or width <= 2 * padding or height <= top + bottom:
        return []
    maximum = max(maximum_value or 0.0, max(values), 1.0)
    return [
        (
            padding + index * (width - 2 * padding) / max(len(values) - 1, 1),
            height - bottom - value / maximum * (height - top - bottom),
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
