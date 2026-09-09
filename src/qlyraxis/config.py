"""Scenario loading and validation using only the Python standard library."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a scenario violates the Qlyraxis configuration contract."""


ALLOWED_MOTIONS = {"straight_line", "circular", "figure_eight", "random"}
ALLOWED_ATMOSPHERES = {"clear", "haze", "fog", "rain", "low_light"}
ALLOWED_NOISE = {"gaussian", "poisson", "salt_pepper"}
ALLOWED_SHAPES = {"square", "circle"}
ALLOWED_PLATFORM_MOTION = {
    "none",
    "linear",
    "circular",
    "figure_eight",
    "spiral",
    "random",
}


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    camera: dict[str, Any]
    target: dict[str, Any]
    disturbances: dict[str, Any]
    evaluation: dict[str, Any]

    def summary(self) -> str:
        viewport = self.camera["viewport_px"]
        motion = self.target["motion"]["type"]
        atmosphere = self.disturbances["atmosphere"]
        return (
            f"{self.name}: {motion}, {viewport[0]}x{viewport[1]} viewport, "
            f"{atmosphere} atmosphere, seed={self.evaluation['random_seed']}"
        )


def _require_keys(mapping: dict[str, Any], keys: set[str], location: str) -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        raise ConfigError(f"{location} is missing: {', '.join(missing)}")


def _positive_pair(value: Any, location: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{location} must contain exactly two numbers")
    if any(not isinstance(item, (int, float)) or item <= 0 for item in value):
        raise ConfigError(f"{location} values must be positive numbers")
    return float(value[0]), float(value[1])


def _positive_number(value: Any, location: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{location} must be a positive number")
    return float(value)


def validate_scenario(data: dict[str, Any]) -> Scenario:
    _require_keys(
        data,
        {"name", "description", "camera", "target", "disturbances", "evaluation"},
        "scenario",
    )
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ConfigError("scenario.name must be a non-empty string")

    camera = data["camera"]
    _require_keys(
        camera,
        {
            "world_size_px",
            "viewport_px",
            "fov_deg",
            "update_hz",
            "control_update_hz",
            "max_pan_speed_deg_s",
            "max_tilt_speed_deg_s",
            "initial_position_px",
        },
        "camera",
    )
    world_w, world_h = _positive_pair(camera["world_size_px"], "camera.world_size_px")
    view_w, view_h = _positive_pair(camera["viewport_px"], "camera.viewport_px")
    _positive_pair(camera["fov_deg"], "camera.fov_deg")
    initial_x, initial_y = _positive_pair(
        camera["initial_position_px"], "camera.initial_position_px"
    )
    if view_w > world_w or view_h > world_h:
        raise ConfigError("camera viewport cannot be larger than the world")
    if not view_w / 2 <= initial_x <= world_w - view_w / 2:
        raise ConfigError("camera initial x-position must keep the viewport in the world")
    if not view_h / 2 <= initial_y <= world_h - view_h / 2:
        raise ConfigError("camera initial y-position must keep the viewport in the world")
    if camera["update_hz"] < 30:
        raise ConfigError("camera.update_hz must be at least 30")
    if camera["control_update_hz"] < 20:
        raise ConfigError("camera.control_update_hz must be at least 20")
    for key in ("max_pan_speed_deg_s", "max_tilt_speed_deg_s"):
        if not 5 <= camera[key] <= 10:
            raise ConfigError(f"camera.{key} must be between 5 and 10 deg/s")

    target = data["target"]
    _require_keys(
        target,
        {"count", "shape", "size_px", "initial_location", "motion"},
        "target",
    )
    if not isinstance(target["count"], int) or target["count"] < 1:
        raise ConfigError("target.count must be an integer of at least 1")
    if target["shape"] not in ALLOWED_SHAPES:
        raise ConfigError(f"target.shape must be one of {sorted(ALLOWED_SHAPES)}")
    target_w, target_h = _positive_pair(target["size_px"], "target.size_px")
    if not 5 <= target_w <= 20 or not 5 <= target_h <= 20:
        raise ConfigError("target.size_px dimensions must be between 5 and 20 pixels")
    initial_location = target["initial_location"]
    if initial_location != "random":
        initial_target_x, initial_target_y = _positive_pair(
            initial_location, "target.initial_location"
        )
        if initial_target_x >= world_w or initial_target_y >= world_h:
            raise ConfigError("target.initial_location must be inside the world")
    motion = target["motion"]
    if not isinstance(motion, dict):
        raise ConfigError("target.motion must be an object")
    motion_type = motion.get("type")
    if motion_type not in ALLOWED_MOTIONS:
        raise ConfigError(f"target.motion.type must be one of {sorted(ALLOWED_MOTIONS)}")
    motion_fields = {
        "straight_line": ("speed_px_s",),
        "circular": ("radius_px", "period_s"),
        "figure_eight": ("width_px", "height_px", "period_s"),
        "random": ("max_speed_px_s", "turn_interval_s"),
    }
    for field_name in motion_fields[motion_type]:
        if field_name not in motion:
            raise ConfigError(f"target.motion is missing: {field_name}")
        _positive_number(motion[field_name], f"target.motion.{field_name}")
    if motion_type == "straight_line" and not isinstance(
        motion.get("heading_deg"), (int, float)
    ):
        raise ConfigError("target.motion.heading_deg must be a number")
    beacon_code = target.get("beacon_code")
    if beacon_code is not None:
        if not isinstance(beacon_code, dict):
            raise ConfigError("target.beacon_code must be an object")
        _require_keys(
            beacon_code,
            {"identity", "pattern", "symbol_frames", "low_intensity"},
            "target.beacon_code",
        )
        identity = beacon_code["identity"]
        pattern = beacon_code["pattern"]
        if not isinstance(identity, str) or not identity.strip():
            raise ConfigError("target.beacon_code.identity must be non-empty")
        if (
            not isinstance(pattern, str)
            or len(pattern) < 7
            or set(pattern) != {"0", "1"}
        ):
            raise ConfigError(
                "target.beacon_code.pattern must contain both 0 and 1 and be at least 7 bits"
            )
        if not isinstance(beacon_code["symbol_frames"], int) or not 1 <= beacon_code[
            "symbol_frames"
        ] <= 5:
            raise ConfigError("target.beacon_code.symbol_frames must be between 1 and 5")
        low_intensity = beacon_code["low_intensity"]
        if not isinstance(low_intensity, (int, float)) or not 32 <= low_intensity <= 180:
            raise ConfigError("target.beacon_code.low_intensity must be between 32 and 180")
        decoy_patterns = beacon_code.get("decoy_patterns", [])
        if not isinstance(decoy_patterns, list) or any(
            not isinstance(item, str)
            or len(item) != len(pattern)
            or set(item) != {"0", "1"}
            for item in decoy_patterns
        ):
            raise ConfigError(
                "target.beacon_code.decoy_patterns must contain binary patterns matching the primary length"
            )

    disturbances = data["disturbances"]
    _require_keys(
        disturbances,
        {
            "noise",
            "noise_std_px",
            "salt_pepper_fraction",
            "camera_jitter_max_px_frame",
            "atmosphere",
            "platform_motion",
            "platform_motion_max_px_frame",
            "dropout",
        },
        "disturbances",
    )
    if not isinstance(disturbances["noise"], list):
        raise ConfigError("disturbances.noise must be a list")
    invalid_noise = set(disturbances["noise"]) - ALLOWED_NOISE
    if invalid_noise:
        raise ConfigError(f"unsupported noise types: {sorted(invalid_noise)}")
    if not 0 <= disturbances["noise_std_px"] <= 20:
        raise ConfigError("disturbances.noise_std_px must be between 0 and 20")
    if not 0 <= disturbances["salt_pepper_fraction"] <= 0.1:
        raise ConfigError("disturbances.salt_pepper_fraction must be between 0 and 0.1")
    if not 0 <= disturbances["camera_jitter_max_px_frame"] <= 20:
        raise ConfigError("camera jitter must be between 0 and 20 px/frame")
    if disturbances["atmosphere"] not in ALLOWED_ATMOSPHERES:
        raise ConfigError(
            f"disturbances.atmosphere must be one of {sorted(ALLOWED_ATMOSPHERES)}"
        )
    if disturbances["platform_motion"] not in ALLOWED_PLATFORM_MOTION:
        raise ConfigError(
            "disturbances.platform_motion must be one of "
            f"{sorted(ALLOWED_PLATFORM_MOTION)}"
        )
    if not 0 <= disturbances["platform_motion_max_px_frame"] <= 20:
        raise ConfigError("platform motion must be between 0 and 20 px/frame")
    for field_name, maximum in (
        ("atmosphere_strength", 1),
        ("turbulence_strength_px", 20),
        ("defocus_blur_px", 15),
        ("motion_blur_px", 31),
    ):
        value = disturbances.get(field_name, 0)
        if not isinstance(value, (int, float)) or not 0 <= value <= maximum:
            raise ConfigError(
                f"disturbances.{field_name} must be between 0 and {maximum}"
            )
    dropout = disturbances["dropout"]
    if not isinstance(dropout, dict):
        raise ConfigError("disturbances.dropout must be an object")
    _require_keys(dropout, {"enabled", "start_s", "duration_s"}, "disturbances.dropout")
    if not isinstance(dropout["enabled"], bool):
        raise ConfigError("disturbances.dropout.enabled must be boolean")
    if dropout["start_s"] < 0 or dropout["duration_s"] < 0:
        raise ConfigError("dropout start and duration cannot be negative")

    evaluation = data["evaluation"]
    _require_keys(
        evaluation,
        {"duration_s", "random_seed", "ground_truth_enabled", "log_format"},
        "evaluation",
    )
    if evaluation["duration_s"] <= 0:
        raise ConfigError("evaluation.duration_s must be positive")
    if not isinstance(evaluation["random_seed"], int):
        raise ConfigError("evaluation.random_seed must be an integer")
    if not set(evaluation["log_format"]).issubset({"csv", "json"}):
        raise ConfigError("evaluation.log_format supports csv and json")

    return Scenario(
        name=data["name"],
        description=data["description"],
        camera=camera,
        target=target,
        disturbances=disturbances,
        evaluation=evaluation,
    )


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path)
    try:
        data = json.loads(scenario_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"scenario file not found: {scenario_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {scenario_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("scenario root must be a JSON object")
    return validate_scenario(data)
