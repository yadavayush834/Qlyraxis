"""Resolve project data both from source checkouts and PyInstaller bundles."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_path(relative: str | Path) -> Path:
    requested = Path(relative)
    if requested.is_absolute() or requested.exists():
        return requested
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        bundled = Path(bundle_root) / requested
        if bundled.exists():
            return bundled
    checkout = Path(__file__).resolve().parents[2] / requested
    return checkout if checkout.exists() else requested
