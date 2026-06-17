"""Detect numeric data labels on bar chart crops using EasyOCR and axis geometry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Hough parameters ported from OCRlabel.ipynb
_HOUGH_THRESHOLD = 100
_HOUGH_MIN_LINE_LENGTH = 200
_HOUGH_MAX_LINE_GAP = 20
_HOUGH_ANGLE_TOL = 10


def parse_data_label_value(text: Any) -> Optional[float]:
    """Parse a data-label string, including k/M unit suffixes."""
    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    cleaned = raw.replace(",", "").replace("O", "0").replace("o", "0")
    cleaned_lower = cleaned.lower()

    multiplier = 1.0
    if cleaned_lower.endswith("k"):
        multiplier = 1_000.0
        cleaned = cleaned[:-1]
    elif cleaned_lower.endswith("m") and not cleaned_lower.endswith("mm"):
        # Treat trailing m as millions when preceded by digits (e.g. 2.4m).
        tail = cleaned_lower.rstrip("m")
        if tail and tail[-1].isdigit():
            multiplier = 1_000_000.0
            cleaned = cleaned[:-1]

    cleaned = cleaned.replace("%", "").strip()
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return None

    try:
        return float(matches[0]) * multiplier
    except ValueError:
        return None


def detect_x_axis(img: np.ndarray) -> Optional[int]:
    """Return Y coordinate of the longest horizontal line in the bottom half."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )

    x_axis_y: Optional[int] = None
    longest_line = 0
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy < _HOUGH_ANGLE_TOL:
            length = dx
            if length > longest_line and y1 > img.shape[0] * 0.5:
                longest_line = length
                x_axis_y = int((y1 + y2) / 2)

    return x_axis_y


def detect_y_axis(img: np.ndarray) -> Optional[int]:
    """Return X coordinate of the longest vertical line in the left half."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )

    y_axis_x: Optional[int] = None
    longest_line = 0
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < _HOUGH_ANGLE_TOL:
            length = dy
            mid_x = int((x1 + x2) / 2)
            if length > longest_line and mid_x < img.shape[1] * 0.5:
                longest_line = length
                y_axis_x = mid_x

    return y_axis_x


def _bbox_from_easyocr(box: Sequence[Sequence[float]]) -> Tuple[int, int, int, int, float, float]:
    x_coords = [float(p[0]) for p in box]
    y_coords = [float(p[1]) for p in box]
    x1 = int(min(x_coords))
    y1 = int(min(y_coords))
    x2 = int(max(x_coords))
    y2 = int(max(y_coords))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return x1, y1, x2, y2, cx, cy


def ocr_all_boxes(image: np.ndarray, reader: Any) -> List[Dict[str, Any]]:
    """Run EasyOCR on a BGR crop and return all detections with geometry."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)

    boxes: List[Dict[str, Any]] = []
    for box, text, confidence in results:
        x1, y1, x2, y2, cx, cy = _bbox_from_easyocr(box)
        boxes.append(
            {
                "text": str(text),
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
            }
        )
    return boxes


def filter_labels_above_x_axis(
    ocr_boxes: Sequence[Dict[str, Any]],
    x_axis_y: int,
) -> List[Dict[str, Any]]:
    """Keep OCR boxes whose center is above the detected x-axis."""
    filtered: List[Dict[str, Any]] = []
    for item in ocr_boxes:
        if float(item["cy"]) < x_axis_y:
            filtered.append(dict(item))
    return filtered


def filter_labels_right_of_y_axis(
    ocr_boxes: Sequence[Dict[str, Any]],
    y_axis_x: int,
) -> List[Dict[str, Any]]:
    """Keep OCR boxes whose center is to the right of the detected y-axis."""
    filtered: List[Dict[str, Any]] = []
    for item in ocr_boxes:
        if float(item["cx"]) > y_axis_x:
            filtered.append(dict(item))
    return filtered


def _relative_position_in_crop(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
) -> Dict[str, float]:
    """Normalize label geometry to 0-1 coordinates within the crop."""
    w = max(img_w, 1)
    h = max(img_h, 1)
    return {
        "x1": round(x1 / w, 6),
        "y1": round(y1 / h, 6),
        "x2": round(x2 / w, 6),
        "y2": round(y2 / h, 6),
        "cx": round(cx / w, 6),
        "cy": round(cy / h, 6),
    }


def _relative_position_to_axis(
    cx: float,
    cy: float,
    axis_position: int,
    orientation: str,
    img_w: int,
    img_h: int,
) -> Dict[str, float]:
    """Position relative to the detected category baseline (x- or y-axis)."""
    if orientation == "vertical":
        plot_h = max(axis_position, 1)
        plot_w = max(img_w, 1)
        return {
            "along_axis": round(cx / plot_w, 6),
            "from_axis": round((axis_position - cy) / plot_h, 6),
        }

    plot_w = max(img_w - axis_position, 1)
    plot_h = max(img_h, 1)
    return {
        "along_axis": round(cy / plot_h, 6),
        "from_axis": round((cx - axis_position) / plot_w, 6),
    }


def _to_label_records(
    candidates: Sequence[Dict[str, Any]],
    img_w: int,
    img_h: int,
    axis_position: int,
    orientation: str,
) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    for item in candidates:
        value = parse_data_label_value(item["text"])
        if value is None:
            continue

        x1 = int(item["x1"])
        y1 = int(item["y1"])
        x2 = int(item["x2"])
        y2 = int(item["y2"])
        cx = float(item["cx"])
        cy = float(item["cy"])

        labels.append(
            {
                "text": item["text"],
                "value": value,
                "confidence": float(item["confidence"]),
                "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "cx": cx,
                "cy": cy,
                "relative_position": _relative_position_in_crop(x1, y1, x2, y2, cx, cy, img_w, img_h),
                "relative_to_axis": _relative_position_to_axis(
                    cx, cy, axis_position, orientation, img_w, img_h
                ),
            }
        )
    return labels


def draw_label_overlay(
    image: np.ndarray,
    labels: Sequence[Dict[str, Any]],
    axis_position: Optional[int],
    orientation: str,
    output_path: Path | str,
) -> None:
    """Save a debug image with detected labels and axis line."""
    output = image.copy()
    for label in labels:
        bbox = label["bbox_xyxy"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            str(label["text"]),
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if axis_position is not None:
        if orientation == "vertical":
            cv2.line(output, (0, axis_position), (output.shape[1], axis_position), (255, 0, 0), 2)
        else:
            cv2.line(output, (axis_position, 0), (axis_position, output.shape[0]), (255, 0, 0), 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)


def detect_data_labels(
    crop_path: Path | str,
    orientation: str,
    reader: Any,
) -> Dict[str, Any]:
    """Detect numeric bar data labels on a chart crop."""
    image = cv2.imread(str(crop_path))
    if image is None:
        raise ValueError(f"Image not found: {crop_path}")

    if orientation not in {"vertical", "horizontal"}:
        return {
            "status": "no_axis",
            "orientation": orientation,
            "axis_position": None,
            "labels": [],
            "label_count": 0,
            "reason": f"Unsupported orientation: {orientation!r}",
        }

    img_h, img_w = image.shape[:2]
    ocr_boxes = ocr_all_boxes(image, reader)

    if orientation == "vertical":
        axis_position = detect_x_axis(image)
        if axis_position is None:
            return {
                "status": "no_axis",
                "orientation": orientation,
                "axis_position": None,
                "labels": [],
                "label_count": 0,
                "reason": "Could not detect x-axis baseline",
            }
        candidates = filter_labels_above_x_axis(ocr_boxes, axis_position)
    else:
        axis_position = detect_y_axis(image)
        if axis_position is None:
            return {
                "status": "no_axis",
                "orientation": orientation,
                "axis_position": None,
                "labels": [],
                "label_count": 0,
                "reason": "Could not detect y-axis baseline",
            }
        candidates = filter_labels_right_of_y_axis(ocr_boxes, axis_position)

    labels = _to_label_records(candidates, img_w, img_h, axis_position, orientation)
    if not labels:
        return {
            "status": "no_labels",
            "orientation": orientation,
            "axis_position": axis_position,
            "labels": [],
            "label_count": 0,
            "reason": "Axis detected but no numeric data labels found above/right of axis",
        }

    return {
        "status": "ok",
        "orientation": orientation,
        "axis_position": axis_position,
        "labels": labels,
        "label_count": len(labels),
        "reason": f"Found {len(labels)} numeric data label(s)",
    }
