#!/usr/bin/env python3
"""
Fuse Roboflow bar-chart detection with start-at-zero checking and label detection.

This script is intentionally compatible with visualize_results_notebook.ipynb.
It writes results.json as a LIST of chart result objects with these fields:

- chart_id
- detector_label
- detector_confidence
- orientation
- bbox_xyxy: {x1, y1, x2, y2}
- crop_path
- start_at_zero_result
- label_detection_result
- bar_detection_result

Example:
    export ROBOFLOW_API_KEY="your_api_key"

    python main.py \
      --image Dataset/Compliant/17.png \
      --workspace-name khas-workspace-3cwa2 \
      --workflow-id bar-chart-detection-and-crop-1779799226718 \
      --output-json results.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from shutil import which
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

from bar_detection import detect_bars, draw_bar_overlay
from label_detection import detect_data_labels, draw_label_overlay

try:
    import easyocr  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency message
    easyocr = None
    EASYOCR_IMPORT_ERROR = exc
else:
    EASYOCR_IMPORT_ERROR = None

try:
    import pytesseract  # type: ignore
    from PIL import Image as PILImage  # type: ignore
except Exception:
    pytesseract = None
    PILImage = None


_EASYOCR_READER = None


def get_easyocr_reader():
    """Lazy-load EasyOCR so Roboflow-only failures are easier to debug."""
    global _EASYOCR_READER
    if easyocr is None:
        raise RuntimeError(
            "easyocr is not installed or failed to import. Install dependencies with:\n"
            "    pip install inference-sdk opencv-python easyocr numpy pillow pytesseract\n"
            f"Original error: {EASYOCR_IMPORT_ERROR}"
        )
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def configure_tesseract() -> bool:
    """Return True if pytesseract can be used. Tesseract is optional."""
    if pytesseract is None or PILImage is None:
        return False

    cmd = os.getenv("TESSERACT_CMD") or which("tesseract") or "/opt/homebrew/bin/tesseract"
    if cmd and Path(cmd).exists():
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd
            return True
        except Exception:
            return False
    return False


TESSERACT_AVAILABLE = configure_tesseract()


# -----------------------------------------------------------------------------
# Start-at-zero OCR logic
# -----------------------------------------------------------------------------


def extract_number(text: Any) -> Optional[float]:
    """Extract the first number from OCR text."""
    if text is None:
        return None

    cleaned = str(text)
    cleaned = cleaned.replace(",", "").replace("%", "").replace("m", "").strip()
    cleaned = cleaned.replace("O", "0").replace("o", "0")

    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return None

    try:
        return float(matches[0])
    except ValueError:
        return None


def box_geometry(box: Sequence[Sequence[float]], scale: float = 1.0) -> Dict[str, float]:
    """Return OCR box geometry in original crop coordinates."""
    xs = [float(p[0]) / scale for p in box]
    ys = [float(p[1]) / scale for p in box]
    return {
        "left": min(xs),
        "right": max(xs),
        "top": min(ys),
        "bottom": max(ys),
        "cx": sum(xs) / 4.0,
        "cy": sum(ys) / 4.0,
    }


def preprocess_for_ocr(image: np.ndarray, scale: float = 2.0) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray, scale


def ocr_left_strip_for_percent_axis(image: np.ndarray, strip_ratio: float = 0.10) -> List[Dict[str, Any]]:
    """
    Optional Tesseract pass for faint percentage labels near the left axis.

    Coordinates are converted back to original crop coordinates before returning.
    If Tesseract is not installed, this silently returns an empty list.
    """
    if not TESSERACT_AVAILABLE or pytesseract is None or PILImage is None:
        return []

    h, w = image.shape[:2]
    strip = image[:, : max(1, int(w * strip_ratio))]
    if strip.size == 0:
        return []

    scale = 2.0
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    pil_img = PILImage.fromarray(gray)
    config = "--psm 4 -c tessedit_char_whitelist=0123456789.%"

    try:
        data = pytesseract.image_to_data(
            pil_img,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        print(f"Warning: Tesseract strip pass skipped: {exc}", file=sys.stderr)
        return []

    detected: List[Dict[str, Any]] = []
    seen_cy: set[int] = set()

    for i, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            continue
        if conf < 10:
            continue

        number = extract_number(text)
        if number is None:
            continue

        x = float(data["left"][i]) / scale
        y = float(data["top"][i]) / scale
        bw = float(data["width"][i]) / scale
        bh = float(data["height"][i]) / scale

        cy_key = round(y + bh / 2.0)
        if cy_key in seen_cy:
            continue
        seen_cy.add(cy_key)

        detected.append(
            {
                "value": number,
                "text": text,
                "confidence": conf / 100.0,
                "left": x,
                "right": x + bw,
                "top": y,
                "bottom": y + bh,
                "cx": x + bw / 2.0,
                "cy": y + bh / 2.0,
                "source": "tesseract_left_strip",
            }
        )

    return detected


def ocr_numeric_boxes(image: np.ndarray) -> Tuple[List[Dict[str, Any]], int, int]:
    """OCR all numeric labels. Returned coordinates are original crop coordinates."""
    reader = get_easyocr_reader()
    processed, scale = preprocess_for_ocr(image, scale=2.0)
    results = reader.readtext(processed)

    numeric_boxes: List[Dict[str, Any]] = []
    for box, text, confidence in results:
        if confidence < 0.20:
            continue
        number = extract_number(text)
        if number is None:
            continue

        # Critical: EasyOCR ran on the 2x preprocessed image, so divide by scale.
        geo = box_geometry(box, scale=scale)
        numeric_boxes.append(
            {
                "value": number,
                "text": str(text),
                "confidence": float(confidence),
                "left": geo["left"],
                "right": geo["right"],
                "top": geo["top"],
                "bottom": geo["bottom"],
                "cx": geo["cx"],
                "cy": geo["cy"],
                "source": "easyocr",
            }
        )

    for item in ocr_left_strip_for_percent_axis(image):
        duplicate = any(
            abs(float(item["cy"]) - float(n["cy"])) < 10
            and abs(float(item["value"]) - float(n["value"])) < 1e-6
            for n in numeric_boxes
        )
        if not duplicate:
            numeric_boxes.append(item)

    h, w = image.shape[:2]
    return numeric_boxes, w, h


def is_monotonic(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return False
    increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return increasing or decreasing


def is_evenly_spaced(values: Sequence[float], tolerance_ratio: float = 0.22) -> bool:
    if len(values) < 3:
        return False
    vals = sorted(float(v) for v in values)
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if any(abs(d) < 1e-9 for d in diffs):
        return False
    avg_diff = sum(diffs) / len(diffs)
    if abs(avg_diff) < 1e-9:
        return False
    return all(abs(d - avg_diff) <= abs(avg_diff) * tolerance_ratio for d in diffs)


def find_vertical_value_axis(numbers: Sequence[Dict[str, Any]], image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """For vertical bar charts, the value axis is normally the y-axis on the left."""
    if len(numbers) < 3:
        return []

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    bin_size = max(1.0, image_width * 0.04)
    for item in numbers:
        x_bin = int(float(item["right"]) // bin_size)
        groups[x_bin].append(item)

    best_group: List[Dict[str, Any]] = []
    best_score = 0

    for group in groups.values():
        if len(group) < 3:
            continue

        group = sorted(group, key=lambda x: float(x["cy"]))
        values = [float(g["value"]) for g in group]
        if not is_evenly_spaced(values):
            continue

        x_positions = [float(g["right"]) for g in group]
        y_positions = [float(g["cy"]) for g in group]
        mean_x = sum(x_positions) / len(x_positions)
        x_spread = max(x_positions) - min(x_positions)
        y_spread = max(y_positions) - min(y_positions)

        score = 0
        if mean_x < image_width * 0.30:
            score += 3
        if x_spread < image_width * 0.12:
            score += 3
        if y_spread > image_height * 0.25:
            score += 3
        if len(group) >= 3:
            score += 2
        if is_monotonic(values):
            score += 4
        if is_evenly_spaced(values):
            score += 4

        if score > best_score and score >= 12:
            best_score = score
            best_group = group

    return best_group


def find_horizontal_value_axis(numbers: Sequence[Dict[str, Any]], image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """For horizontal bar charts, the value axis is normally the x-axis at the bottom."""
    if len(numbers) < 3:
        return []

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    bin_size = max(1.0, image_height * 0.08)
    for item in numbers:
        y_bin = int(float(item["cy"]) // bin_size)
        groups[y_bin].append(item)

    best_group: List[Dict[str, Any]] = []
    best_score = 0

    for group in groups.values():
        if len(group) < 3:
            continue

        group = sorted(group, key=lambda x: float(x["cx"]))
        values = [float(g["value"]) for g in group]
        if not is_evenly_spaced(values):
            continue

        x_positions = [float(g["cx"]) for g in group]
        y_positions = [float(g["cy"]) for g in group]
        mean_y = sum(y_positions) / len(y_positions)
        x_spread = max(x_positions) - min(x_positions)
        y_spread = max(y_positions) - min(y_positions)

        score = 0
        if mean_y > image_height * 0.65:
            score += 3
        if x_spread > image_width * 0.25:
            score += 3
        if y_spread < image_height * 0.10:
            score += 3
        if len(group) >= 3:
            score += 2
        if is_monotonic(values):
            score += 4
        if is_evenly_spaced(values):
            score += 4

        if score > best_score and score >= 12:
            best_score = score
            best_group = group

    return best_group


def ocr_bottom_left_for_zero(image: np.ndarray) -> bool:
    """Rescue pass for a small zero near the chart origin."""
    h, w = image.shape[:2]
    crop = image[int(h * 0.65) :, : int(w * 0.32)]
    if crop.size == 0:
        return False

    reader = get_easyocr_reader()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    strategies = [resized]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    strategies.append(clahe.apply(resized.copy()))

    is_dark = np.mean(resized) < 128
    threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if is_dark else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, thresholded = cv2.threshold(resized.copy(), 0, 255, threshold_type)
    strategies.append(thresholded)

    for img in strategies:
        results = reader.readtext(img, allowlist="0123456789.-")
        for _, text, conf in results:
            number = extract_number(text)
            if conf > 0.05 and number is not None and abs(number) < 1.0:
                return True
    return False


def hunt_zero_on_value_axis(image: np.ndarray, orientation: str) -> bool:
    """Last-resort scan in the likely zero area for the selected orientation."""
    h, w = image.shape[:2]
    if orientation == "vertical":
        search = image[:, : int(w * 0.28)]
    elif orientation == "horizontal":
        search = image[int(h * 0.60) :, : int(w * 0.45)]
    else:
        return False

    if search.size == 0:
        return False

    reader = get_easyocr_reader()
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    results = reader.readtext(gray, allowlist="0123456789.m%")
    for _, text, conf in results:
        number = extract_number(text)
        if conf > 0.05 and number is not None and abs(number) < 1.0:
            return True
    return False


def check_starts_at_zero(
    image_path: Path | str,
    orientation: str,
    assume_compliant_if_axis_missing: bool = True,
) -> Dict[str, Any]:
    """Check whether a cropped chart's value axis starts at zero."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    numbers, w, h = ocr_numeric_boxes(image)

    if orientation == "vertical":
        axis_group = find_vertical_value_axis(numbers, w, h)
        axis_type = "y-axis"
    elif orientation == "horizontal":
        axis_group = find_horizontal_value_axis(numbers, w, h)
        axis_type = "x-axis"
    else:
        return {
            "status": "unknown",
            "starts_at_zero": None,
            "orientation": orientation,
            "axis_type": None,
            "detected_axis_values": [],
            "ocr_numbers_count": len(numbers),
            "zero_source": None,
            "reason": "Invalid or unknown chart orientation",
        }

    if not axis_group:
        return {
            "status": "compliant" if assume_compliant_if_axis_missing else "unknown",
            "starts_at_zero": True if assume_compliant_if_axis_missing else None,
            "orientation": orientation,
            "axis_type": axis_type,
            "detected_axis_values": [],
            "ocr_numbers_count": len(numbers),
            "zero_source": None,
            "reason": "No reliable value-axis label group detected",
        }

    axis_values = [float(item["value"]) for item in axis_group]
    contains_zero = any(abs(v) < 1.0 for v in axis_values)
    zero_source = "axis_group" if contains_zero else None

    if not contains_zero:
        contains_zero = ocr_bottom_left_for_zero(image)
        if contains_zero:
            zero_source = "bottom_left_rescue"
            axis_values.append(0.0)

    if not contains_zero:
        contains_zero = hunt_zero_on_value_axis(image, orientation)
        if contains_zero:
            zero_source = "axis_area_rescue"
            axis_values.append(0.0)

    reason_map = {
        "axis_group": "Zero found directly in the selected value-axis label group",
        "bottom_left_rescue": "Zero found by the bottom-left origin rescue pass",
        "axis_area_rescue": "Zero found by the orientation-specific axis-area rescue pass",
        None: "Value-axis labels detected, but zero was not found",
    }

    return {
        "status": "compliant" if contains_zero else "non_compliant",
        "starts_at_zero": bool(contains_zero),
        "orientation": orientation,
        "axis_type": axis_type,
        "detected_axis_values": sorted(set(axis_values)),
        "ocr_numbers_count": len(numbers),
        "zero_source": zero_source,
        "reason": reason_map[zero_source],
    }


# -----------------------------------------------------------------------------
# Roboflow Workflow logic
# -----------------------------------------------------------------------------


def run_roboflow_workflow(
    image_path: Path | str,
    workspace_name: str,
    workflow_id: str,
    api_key: str,
    api_url: str,
    image_input_name: str,
) -> Any:
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return client.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={image_input_name: str(image_path)},
    )


def unwrap_single_result(result: Any) -> Any:
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


def is_prediction_dict(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    keys = set(item.keys())
    has_center_box = {"x", "y", "width", "height"}.issubset(keys)
    has_xyxy_box = {"x_min", "y_min", "x_max", "y_max"}.issubset(keys) or {"xmin", "ymin", "xmax", "ymax"}.issubset(keys)
    has_ltrb_box = {"left", "top", "right", "bottom"}.issubset(keys)
    return has_center_box or has_xyxy_box or has_ltrb_box


def collect_prediction_lists(obj: Any, path: str = "root") -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Recursively find every list that looks like object-detection predictions."""
    found: List[Tuple[str, List[Dict[str, Any]]]] = []

    if isinstance(obj, list):
        pred_items = [x for x in obj if is_prediction_dict(x)]
        if pred_items:
            found.append((path, pred_items))
        for i, value in enumerate(obj):
            found.extend(collect_prediction_lists(value, f"{path}[{i}]"))

    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(collect_prediction_lists(value, f"{path}.{key}"))

    return found


def prediction_class(pred: Dict[str, Any]) -> str:
    for key in ["class", "class_name", "label", "name"]:
        if key in pred and pred[key] is not None:
            return str(pred[key])
    return ""


def prediction_class_lower(pred: Dict[str, Any]) -> str:
    return prediction_class(pred).lower().strip()


def orientation_from_prediction(pred: Dict[str, Any]) -> Optional[str]:
    label = prediction_class_lower(pred)
    if "horizontal" in label:
        return "horizontal"
    if "vertical" in label:
        return "vertical"
    return None


def prediction_confidence(pred: Dict[str, Any]) -> Optional[float]:
    for key in ["confidence", "class_confidence", "score"]:
        if key in pred and pred[key] is not None:
            try:
                return float(pred[key])
            except Exception:
                return None
    return None


def extract_chart_predictions(workflow_result: Any) -> List[Dict[str, Any]]:
    """
    Extract bar-chart predictions from flexible Roboflow Workflow output.

    Your screenshot shows `detect.predictions`; this function also handles nested
    outputs by selecting the prediction list with the most horizontal/vertical classes.
    """
    result = unwrap_single_result(workflow_result)
    candidates = collect_prediction_lists(result)
    if not candidates:
        return []

    def score_candidate(pair: Tuple[str, List[Dict[str, Any]]]) -> Tuple[int, int]:
        _, preds = pair
        orient_count = sum(1 for p in preds if orientation_from_prediction(p) in {"horizontal", "vertical"})
        return orient_count, len(preds)

    best_path, best_preds = max(candidates, key=score_candidate)
    orient_count = sum(1 for p in best_preds if orientation_from_prediction(p) in {"horizontal", "vertical"})
    print(f"Using predictions from {best_path}: {len(best_preds)} boxes, {orient_count} chart boxes")
    return best_preds


def bbox_from_prediction(
    pred: Dict[str, Any],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.08,
) -> Tuple[int, int, int, int]:
    """Convert a Roboflow prediction to clipped x1,y1,x2,y2 image coordinates."""
    keys = set(pred.keys())

    if {"x", "y", "width", "height"}.issubset(keys):
        x = float(pred["x"])
        y = float(pred["y"])
        bw = float(pred["width"])
        bh = float(pred["height"])
        x1 = x - bw / 2.0
        y1 = y - bh / 2.0
        x2 = x + bw / 2.0
        y2 = y + bh / 2.0
    elif {"x_min", "y_min", "x_max", "y_max"}.issubset(keys):
        x1 = float(pred["x_min"])
        y1 = float(pred["y_min"])
        x2 = float(pred["x_max"])
        y2 = float(pred["y_max"])
    elif {"xmin", "ymin", "xmax", "ymax"}.issubset(keys):
        x1 = float(pred["xmin"])
        y1 = float(pred["ymin"])
        x2 = float(pred["xmax"])
        y2 = float(pred["ymax"])
    elif {"left", "top", "right", "bottom"}.issubset(keys):
        x1 = float(pred["left"])
        y1 = float(pred["top"])
        x2 = float(pred["right"])
        y2 = float(pred["bottom"])
    else:
        raise ValueError(f"Unsupported prediction box format: {pred}")

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    x1_i = int(max(0, round(x1 - pad_x)))
    y1_i = int(max(0, round(y1 - pad_y)))
    x2_i = int(min(image_width, round(x2 + pad_x)))
    y2_i = int(min(image_height, round(y2 + pad_y)))

    if x2_i <= x1_i or y2_i <= y1_i:
        raise ValueError(f"Invalid crop box after clipping: {(x1_i, y1_i, x2_i, y2_i)}")

    return x1_i, y1_i, x2_i, y2_i


# -----------------------------------------------------------------------------
# Output helpers compatible with visualize_results_notebook.ipynb
# -----------------------------------------------------------------------------


def save_json(obj: Any, output_path: Path | str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_possible_base64_image(value: Any, output_path: Path | str) -> Optional[str]:
    """Save Roboflow visualization outputs if they are returned as base64 strings."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value") or value.get("base64") or value.get("image")
    if not isinstance(value, str):
        return None

    try:
        if value.startswith("data:image"):
            value = value.split(",", 1)[1]
        data = base64.b64decode(value)
        with open(output_path, "wb") as f:
            f.write(data)
        return str(output_path)
    except Exception:
        return None


def draw_labeled_image(image_path: Path | str, results: Sequence[Dict[str, Any]], output_path: Path | str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    box_color = (0, 255, 0)  # Green (BGR) — compliance detection overlay

    for item in results:
        bbox = item["bbox_xyxy"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        label = item.get("detector_label", "unknown")
        chart_id = item.get("chart_id", "?")
        text = f"#{chart_id} {label}"

        color = box_color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)


def build_visualizer_item(
    chart_id: int,
    pred: Dict[str, Any],
    orientation: str,
    bbox: Tuple[int, int, int, int],
    crop_path: Path,
    zero_result: Dict[str, Any],
    label_result: Dict[str, Any],
    bar_result: Dict[str, Any],
) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "chart_id": chart_id,
        "detector_label": prediction_class(pred),
        "detector_confidence": prediction_confidence(pred),
        "orientation": orientation,
        "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "crop_path": str(crop_path),
        "start_at_zero_result": zero_result,
        "label_detection_result": label_result,
        "bar_detection_result": bar_result,
    }


def analyze_image(
    image_path: Path,
    workspace_name: str,
    workflow_id: str,
    api_key: str,
    api_url: str,
    image_input_name: str,
    output_dir: Path,
    output_json: Path,
    crop_padding_ratio: float,
    assume_compliant_if_axis_missing: bool,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    label_overlays_dir = output_dir / "label_overlays"
    label_overlays_dir.mkdir(parents=True, exist_ok=True)
    bar_overlays_dir = output_dir / "bar_overlays"
    bar_overlays_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_h, img_w = image.shape[:2]

    print("Running Roboflow Workflow...")
    raw_result = run_roboflow_workflow(
        image_path=image_path,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        api_key=api_key,
        api_url=api_url,
        image_input_name=image_input_name,
    )
    save_json(raw_result, output_dir / "roboflow_raw_result.json")

    unwrapped = unwrap_single_result(raw_result)
    if isinstance(unwrapped, dict):
        save_possible_base64_image(unwrapped.get("annotated_image"), output_dir / "roboflow_annotated_image.png")

    predictions = extract_chart_predictions(raw_result)
    results: List[Dict[str, Any]] = []

    for raw_idx, pred in enumerate(predictions):
        orientation = orientation_from_prediction(pred)
        if orientation not in {"vertical", "horizontal"}:
            print(f"Skipping box {raw_idx}: label is not horizontal/vertical -> {prediction_class(pred)!r}")
            continue

        bbox = bbox_from_prediction(pred, img_w, img_h, padding_ratio=crop_padding_ratio)
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]

        chart_id = len(results)
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", prediction_class_lower(pred) or orientation)
        crop_path = crops_dir / f"chart_{chart_id}_{safe_label}.png"
        cv2.imwrite(str(crop_path), crop)

        print(f"Checking chart {chart_id}: {prediction_class(pred)!r}, orientation={orientation}, crop={crop_path}")
        zero_result = check_starts_at_zero(
            crop_path,
            orientation=orientation,
            assume_compliant_if_axis_missing=assume_compliant_if_axis_missing,
        )

        print(f"Detecting data labels for chart {chart_id}...")
        label_result = detect_data_labels(
            crop_path,
            orientation=orientation,
            reader=get_easyocr_reader(),
        )
        print(
            f"  label status={label_result.get('status')}, "
            f"count={label_result.get('label_count', 0)}"
        )

        if label_result.get("labels"):
            draw_label_overlay(
                crop,
                label_result["labels"],
                label_result.get("axis_position"),
                orientation,
                label_overlays_dir / f"chart_{chart_id}.png",
            )

        print(f"Detecting bars for chart {chart_id}...")
        bar_result = detect_bars(crop, orientation=orientation)
        if bar_result.get("status") != "skipped":
            overlay_path = draw_bar_overlay(
                bar_result["_pipeline_out"],
                bar_overlays_dir / f"chart_{chart_id}.png",
            )
            bar_result["overlay_path"] = overlay_path
            bar_result.pop("_pipeline_out", None)
            print(
                f"  bar status={bar_result.get('status')}, "
                f"count={bar_result.get('bar_count', 0)}"
            )
        else:
            print(f"  bar detection skipped: {bar_result.get('reason')}")

        results.append(
            build_visualizer_item(
                chart_id=chart_id,
                pred=pred,
                orientation=orientation,
                bbox=bbox,
                crop_path=crop_path,
                zero_result=zero_result,
                label_result=label_result,
                bar_result=bar_result,
            )
        )

    save_json(results, output_json)
    draw_labeled_image(image_path, results, output_dir / "labeled_output.png")

    summary = {
        "image_path": str(image_path),
        "workspace_name": workspace_name,
        "workflow_id": workflow_id,
        "api_url": api_url,
        "image_input_name": image_input_name,
        "image_size": {"width": img_w, "height": img_h},
        "num_raw_predictions": len(predictions),
        "num_analyzed_charts": len(results),
        "results_json": str(output_json),
        "labeled_output": str(output_dir / "labeled_output.png"),
        "label_overlays_dir": str(label_overlays_dir),
        "bar_overlays_dir": str(bar_overlays_dir),
        "bar_detection_counts": [
            {
                "chart_id": item["chart_id"],
                "status": item.get("bar_detection_result", {}).get("status"),
                "bar_count": item.get("bar_detection_result", {}).get("bar_count", 0),
            }
            for item in results
        ],
        "raw_result_json": str(output_dir / "roboflow_raw_result.json"),
        "tesseract_available": TESSERACT_AVAILABLE,
    }
    save_json(summary, output_dir / "summary.json")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Roboflow bar-chart detection and start-at-zero checking, outputting results.json for the visualization notebook."
    )
    parser.add_argument("--image", required=True, help="Path to the original dashboard image.")
    parser.add_argument("--workspace-name", required=True, help="Roboflow workspace name.")
    parser.add_argument("--workflow-id", required=True, help="Roboflow Workflow ID.")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key. Defaults to ROBOFLOW_API_KEY env var.")
    parser.add_argument(
        "--api-url",
        default=os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com"),
        help="Roboflow API URL. Use the exact URL shown in your Workflow Deploy snippet if different.",
    )
    parser.add_argument("--image-input-name", default="image", help="Workflow image input name. Your screenshot shows this is 'image'.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for crops, raw Roboflow output, and labeled image.")
    parser.add_argument("--output-json", default="results.json", help="JSON path consumed by visualize_results_notebook.ipynb.")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.08, help="Extra padding around detected chart boxes before OCR.")
    parser.add_argument(
        "--no-assume-compliant-if-axis-missing",
        action="store_true",
        help="Return unknown instead of compliant when no reliable value-axis labels are detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set ROBOFLOW_API_KEY.", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json)

    try:
        results = analyze_image(
            image_path=image_path,
            workspace_name=args.workspace_name,
            workflow_id=args.workflow_id,
            api_key=args.api_key,
            api_url=args.api_url,
            image_input_name=args.image_input_name,
            output_dir=output_dir,
            output_json=output_json,
            crop_padding_ratio=args.crop_padding_ratio,
            assume_compliant_if_axis_missing=not args.no_assume_compliant_if_axis_missing,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Done. Wrote {len(results)} chart result(s) to: {output_json}")
    print(f"Open visualize_results_notebook.ipynb and set RESULTS_PATH = {str(output_json)!r}")
    print(f"Set IMAGE_PATH = {str(image_path)!r}")
    print(f"Labeled image saved to: {output_dir / 'labeled_output.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
