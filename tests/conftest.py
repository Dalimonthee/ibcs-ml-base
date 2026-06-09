"""Pytest configuration: make scaling-detection importable as a flat module dir."""

from __future__ import annotations

import sys
from pathlib import Path

SCALING_DETECTION_DIR = Path(__file__).resolve().parents[1] / "scaling-detection"
if str(SCALING_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SCALING_DETECTION_DIR))
