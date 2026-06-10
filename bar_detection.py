"""OpenCV bar detection on chart crops via scaling-detection/pipeline.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

_SCALING_DETECTION_DIR = Path(__file__).resolve().parent / "scaling-detection"
if str(_SCALING_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALING_DETECTION_DIR))

from pipeline import Params, app_default_params, run_pipeline  # noqa: E402


def _serialize_pipeline_out(out: dict) -> Dict[str, Any]:
    winner = out["winner"]
    winner_data = out["per_mask"][winner]
    corner_bboxes = out["corner_bboxes"]

    return {
        "status": "ok" if corner_bboxes else "no_bars",
        "winner_mask": winner,
        "winner_score": float(winner_data["score"]),
        "bar_count": len(corner_bboxes),
        "bars": [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "close_source": str(src)}
            for x, y, w, h, src in corner_bboxes
        ],
        "projection_bars": [
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "fill_ratio": round(float(fill_ratio), 3),
            }
            for x, y, w, h, fill_ratio in out["winner_boxes"]
        ],
        "axes": [
            {"y": int(y), "weight": round(float(weight), 3), "source": str(src)}
            for y, weight, src in out["axes_peaks"]
        ],
    }


def detect_bars(
    crop_bgr: np.ndarray,
    orientation: str,
    params: Optional[Params] = None,
) -> Dict[str, Any]:
    """Run the OpenCV bar pipeline on a chart crop.

    Returns a JSON-serializable dict. When detection runs, includes
    ``_pipeline_out`` with the raw pipeline result for overlay saving.
    """
    if orientation != "vertical":
        return {
            "status": "skipped",
            "reason": "OpenCV bar pipeline supports vertical bar charts only",
            "orientation": orientation,
            "winner_mask": None,
            "winner_score": None,
            "bar_count": 0,
            "bars": [],
            "projection_bars": [],
            "axes": [],
        }

    pipeline_out = run_pipeline(crop_bgr, params or app_default_params())
    result = _serialize_pipeline_out(pipeline_out)
    result["orientation"] = orientation
    result["_pipeline_out"] = pipeline_out
    return result


def draw_bar_overlay(
    pipeline_out: dict,
    output_path: Path | str,
) -> str:
    """Save the RGB overlay produced by run_pipeline."""
    overlay_rgb = pipeline_out["overlay"]
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay_bgr)
    return str(output_path)
