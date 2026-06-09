"""
main.py  —  corrected to match Startingatzero.ipynb logic exactly
Pipeline:
1. Use Roboflow Workflow to detect vertical/horizontal bar charts.
2. Crop each detected chart from the original dashboard image.
3. Run OCR + axis logic (from notebook) to decide whether the value axis includes 0.

Install:
    pip install inference-sdk opencv-python easyocr numpy

Run:
    export ROBOFLOW_API_KEY="your_key_here"
    python main.py --image dashboard.png --workspace-name your-ws --workflow-id your-wf-id

Changes from previous main.py (all match notebook cells exactly):
    FIX 1 — ocr_bottom_left_for_zero: crop width changed from 0.35 → 0.25,
             crop height start changed from 0.65 → 0.70 (notebook Cell 12).
             This was the root cause of reading bar data labels in the chart middle.
    FIX 2 — ocr_bottom_left_for_zero: restored missing Otsu threshold strategy
             (Strategy 3 from notebook Cell 12).
    FIX 3 — ocr_bottom_left_for_zero: stricter allowlist '0123456789.-'
             and confidence threshold > 0.05 (matching notebook Cell 12).
    FIX 4 — hunt_zero_in_full_image: entire function was missing, now added
             (notebook Cell 14).
    FIX 5 — check_starts_at_zero_image: added third rescue pass calling
             hunt_zero_in_full_image with zero_source='rescue_hunt'
             (notebook Cell 15 logic).
    FIX 6 — find_vertical_value_axis: mean_x threshold restored to 0.25
             (not 0.30) and y_spread threshold restored to 0.25 (not 0.20)
             (notebook Cell 10).
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import easyocr
import numpy as np

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

# ── EasyOCR reader (shared across all calls) ─────────────────────────────────
reader = easyocr.Reader(["en"], gpu=False)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def extract_number(text):
    text = text.replace(",", "").replace("%", "").replace("m", "").strip()

    matches = re.findall(r"-?\d+\.?\d*", text)

    if not matches:
        return None

    return float(matches[0])


def box_geometry(box):
    """Return left/right/top/bottom/cx/cy from a 4-point box.
    Matches notebook Cell 4 exactly.
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return {
        "left":   min(xs),
        "right":  max(xs),
        "top":    min(ys),
        "bottom": max(ys),
        "cx":     sum(xs) / 4,
        "cy":     sum(ys) / 4,
    }


def preprocess_for_ocr(image):
    """Upscale + CLAHE to make faint labels readable.
    Matches notebook Cell 5 exactly.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def ocr_numeric_boxes(image):
    """OCR all numeric text and return values with bounding-box geometry.
    Matches notebook Cell 6 exactly.
    """
    processed = preprocess_for_ocr(image)
    results = reader.readtext(processed)

    numeric_boxes = []
    for box, text, confidence in results:
        if confidence < 0.2:
            continue
        number = extract_number(text)
        if number is None:
            continue
        geo = box_geometry(box)
        numeric_boxes.append({
            "value":      number,
            "text":       text,
            "confidence": float(confidence),
            "left":       geo["left"],
            "right":      geo["right"],
            "top":        geo["top"],
            "bottom":     geo["bottom"],
            "cx":         geo["cx"],
            "cy":         geo["cy"],
        })

    processed_h, processed_w = processed.shape[:2]
    return numeric_boxes, processed_w, processed_h


# ═════════════════════════════════════════════════════════════════════════════
# AXIS SCORING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def is_monotonic(values):
    """Matches notebook Cell 8 exactly."""
    if len(values) < 2:
        return False
    increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return increasing or decreasing


def is_evenly_spaced(values, tolerance_ratio=0.20):
    if len(values) < 3:
        return False

    values = sorted(values)
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

    if any(d == 0 for d in diffs):
        return False

    avg_diff = sum(diffs) / len(diffs)

    return all(abs(d - avg_diff) <= avg_diff * tolerance_ratio for d in diffs)


def infer_zero_on_axis(axis_values, tolerance=0.01):
    """Check if zero would be the next step below the detected axis range.
    Matches notebook Cell 13 exactly.
    """
    if len(axis_values) < 2:
        return False
    vals = sorted(axis_values)
    step = (vals[-1] - vals[0]) / (len(vals) - 1)
    if step <= 0:
        return False
    extrapolated = vals[0] - step
    return abs(extrapolated) <= abs(step) * tolerance


# ═════════════════════════════════════════════════════════════════════════════
# AXIS DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def find_vertical_value_axis(numbers, image_width, image_height):
    if len(numbers) < 3:
        return []

    groups = defaultdict(list)
    bin_size = image_width * 0.04

    for item in numbers:
        x_bin = int(item["right"] // bin_size)
        groups[x_bin].append(item)

    best_group = []
    best_score = 0

    for group in groups.values():
        if len(group) < 3:
            continue

        group = sorted(group, key=lambda x: x["cy"])
        values = [g["value"] for g in group]

        # Hard gate: must be evenly spaced to even be considered
        # This immediately rejects data labels and random number clusters
        if not is_evenly_spaced(values):
            continue

        x_positions = [g["right"] for g in group]
        y_positions = [g["cy"] for g in group]

        mean_x = sum(x_positions) / len(x_positions)
        x_spread = max(x_positions) - min(x_positions)
        y_spread = max(y_positions) - min(y_positions)

        score = 0

        if mean_x < image_width * 0.25:
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


def find_horizontal_value_axis(numbers, image_width, image_height):
    if len(numbers) < 3:
        return []

    groups = defaultdict(list)
    bin_size = image_height * 0.08

    for item in numbers:
        y_bin = int(item["cy"] // bin_size)
        groups[y_bin].append(item)

    best_group = []
    best_score = 0

    for group in groups.values():
        if len(group) < 3:
            continue

        group = sorted(group, key=lambda x: x["cx"])
        values = [g["value"] for g in group]

        # Hard gate: must be evenly spaced
        if not is_evenly_spaced(values):
            continue

        x_positions = [g["cx"] for g in group]
        y_positions = [g["cy"] for g in group]

        mean_y = sum(y_positions) / len(y_positions)
        x_spread = max(x_positions) - min(x_positions)
        y_spread = max(y_positions) - min(y_positions)

        score = 0

        # 1. x-axis labels sit near the bottom of the chart
        if mean_y > image_height * 0.75:
            score += 3

        # 2. x-axis labels should be spread horizontally
        if x_spread > image_width * 0.25:
            score += 3

        # 3. x-axis labels should be vertically tight
        if y_spread < image_height * 0.08:
            score += 3

        # 4. At least 3 tick values
        if len(group) >= 3:
            score += 2

        # 5. Values increase left to right
        if is_monotonic(values):
            score += 4

        # 6. Even spacing (already passed hard gate, so this always adds 4)
        if is_evenly_spaced(values):
            score += 4

        if score > best_score and score >= 12:
            best_score = score
            best_group = group

    return best_group


# ═════════════════════════════════════════════════════════════════════════════
# ZERO RESCUE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def ocr_bottom_left_for_zero(image, image_w, image_h):
    """Rescue pass: look for a standalone '0' in the bottom-left crop.
    Matches notebook Cell 12 exactly.

    FIX 1: crop is image[h*0.70:, :w*0.25]  — NOT 0.65/0.35 as in old main.py.
    FIX 2: Otsu threshold strategy restored (was missing from old main.py).
    FIX 3: allowlist='0123456789.-' and confidence > 0.05 (stricter than old main.py).
    """
    h, w = image.shape[:2]

    # FIX 1 — correct crop region matching notebook exactly
    crop = image[int(h * 0.70):, :int(w * 0.25)]

    if crop.size == 0:
        return False

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    strategies = []

    # Strategy 1: raw grayscale
    strategies.append(resized)

    # Strategy 2: CLAHE only
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    strategies.append(clahe.apply(resized.copy()))

    # Strategy 3: Otsu threshold  ← FIX 2: this was completely missing from old main.py
    is_dark = np.mean(resized) < 128
    tt = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if is_dark else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, s3 = cv2.threshold(resized.copy(), 0, 255, tt)
    strategies.append(s3)

    for img in strategies:
        # FIX 3 — strict allowlist and confidence matching notebook
        results = reader.readtext(img, allowlist="0123456789.-")
        for _, text, conf in results:
            num = extract_number(text)
            if conf > 0.05 and num is not None and abs(num) < 1.0:
                return True

    return False


def hunt_zero_in_full_image(image, image_width, image_height):
    """Dedicated pass: scan the left 25% of the full image for a standalone '0'.
    Matches notebook Cell 14 exactly.

    FIX 4: this entire function was missing from old main.py.
    """
    h, w = image.shape[:2]
    left_strip = image[:, :int(w * 0.25)]   # only left 25%

    gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    results = reader.readtext(gray, allowlist="0123456789.m%")
    for _, text, conf in results:
        num = extract_number(text)
        if conf > 0.05 and num is not None and abs(num) < 1.0:
            return True

    return False


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHECK FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def check_starts_at_zero_image(image, orientation, tolerance=0.01):
    """Return compliance result for one cropped bar-chart image.
    Matches notebook Cell 15 exactly — three rescue passes, correct zero_source labels.

    FIX 5: added third rescue pass (hunt_zero_in_full_image) which was
           missing from old main.py. zero_source labels also match notebook.
    """
    numbers, w, h = ocr_numeric_boxes(image)

    if orientation == "vertical":
        axis_group = find_vertical_value_axis(numbers, w, h)
        axis_type  = "y-axis"
    elif orientation == "horizontal":
        axis_group = find_horizontal_value_axis(numbers, w, h)
        axis_type  = "x-axis"
    else:
        return {
            "status":        "unknown",
            "starts_at_zero": None,
            "orientation":   orientation,
            "reason":        "Invalid or unknown orientation",
        }

    # Client rule: no axis labels detected → compliant
    if not axis_group:
        return {
            "status":               "compliant",
            "starts_at_zero":       True,
            "orientation":          orientation,
            "axis_type":            axis_type,
            "detected_axis_values": [],
            "reason":               "No value-axis labels detected; treated as compliant by client rule",
        }

    axis_values    = [item["value"] for item in axis_group]
    contains_zero  = any(abs(v) < 1.0 for v in axis_values)
    zero_source    = "axis_group" if contains_zero else None

    # Rescue pass 1 — bottom-left crop (FIX 1/2/3 applied inside function)
    if not contains_zero and orientation == "vertical":
        contains_zero = ocr_bottom_left_for_zero(image, w, h)
        if contains_zero:
            zero_source = "rescue_crop"
            axis_values = axis_values + [0.0]

    # Rescue pass 2 — full left-strip hunt  ← FIX 5: was completely missing
    if not contains_zero and orientation == "vertical":
        contains_zero = hunt_zero_in_full_image(image, w, h)
        if contains_zero:
            zero_source = "rescue_hunt"
            axis_values = axis_values + [0.0]

    # zero_source → human-readable reason (matches notebook Cell 15)
    reason_map = {
        "axis_group":   "Zero found on value axis",
        "rescue_crop":  "Zero found via bottom-left crop rescue",
        "rescue_hunt":  "Zero found via left-strip digit scan rescue",
        None:           "Value-axis labels detected but zero was not found",
    }

    return {
        "status":               "compliant" if contains_zero else "non_compliant",
        "starts_at_zero":       contains_zero,
        "orientation":          orientation,
        "axis_type":            axis_type,
        "detected_axis_values": sorted(axis_values),
        "zero_source":          zero_source,
        "reason":               reason_map[zero_source],
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROBOFLOW WORKFLOW PIPELINE  (unchanged — wraps the notebook logic above)
# ═════════════════════════════════════════════════════════════════════════════

def value_from_obj(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def find_prediction_lists(obj):
    found = []
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            if any(any(k in x for k in ("class", "class_name", "label", "x", "y", "width", "height")) for x in obj):
                found.append(obj)
        for item in obj:
            found.extend(find_prediction_lists(item))
    elif isinstance(obj, dict):
        if "predictions" in obj and isinstance(obj["predictions"], list):
            found.append(obj["predictions"])
        for value in obj.values():
            found.extend(find_prediction_lists(value))
    return found


def normalize_predictions_from_workflow(workflow_result):
    prediction_lists = find_prediction_lists(workflow_result)
    normalized = []
    for predictions in prediction_lists:
        for p in predictions:
            label      = value_from_obj(p, "class_name", "class", "label", default="unknown")
            confidence = value_from_obj(p, "confidence", "score",  default=None)
            x          = value_from_obj(p, "x", "center_x")
            y          = value_from_obj(p, "y", "center_y")
            width      = value_from_obj(p, "width",  "w")
            height     = value_from_obj(p, "height", "h")
            x1         = value_from_obj(p, "x1", "left")
            y1         = value_from_obj(p, "y1", "top")
            x2         = value_from_obj(p, "x2", "right")
            y2         = value_from_obj(p, "y2", "bottom")

            if None not in (x, y, width, height):
                normalized.append({
                    "label":      str(label),
                    "confidence": float(confidence) if confidence is not None else None,
                    "x": float(x), "y": float(y),
                    "width": float(width), "height": float(height),
                    "format": "cxcywh", "raw": p,
                })
            elif None not in (x1, y1, x2, y2):
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                normalized.append({
                    "label":      str(label),
                    "confidence": float(confidence) if confidence is not None else None,
                    "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                    "width": x2 - x1,   "height": y2 - y1,
                    "format": "xyxy", "raw": p,
                })

    unique, seen = [], set()
    for p in normalized:
        key = (p["label"], round(p["x"], 2), round(p["y"], 2),
               round(p["width"], 2), round(p["height"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def label_to_orientation(label: str) -> str:
    label_lower = label.lower()
    if "vertical"   in label_lower: return "vertical"
    if "horizontal" in label_lower: return "horizontal"
    return "unknown"


def crop_prediction(image, pred, padding=5):
    img_h, img_w = image.shape[:2]
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    x1 = int(max(0,     x - w / 2 - padding))
    y1 = int(max(0,     y - h / 2 - padding))
    x2 = int(min(img_w, x + w / 2 + padding))
    y2 = int(min(img_h, y + h / 2 + padding))
    return image[y1:y2, x1:x2], {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def run_workflow_detection(image_path, api_url, api_key, workspace_name, workflow_id, confidence_threshold=0.5):
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return client.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={"image": image_path},
        parameters={"confidence": confidence_threshold}  # passed to workflow node
    )


def run_pipeline(
    image_path,
    api_url,
    api_key,
    workspace_name,
    workflow_id,
    confidence_threshold=0.5,
    save_crops=False,
    raw_workflow_output_path=None,
):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    workflow_result = run_workflow_detection(
        image_path=image_path,
        api_url=api_url,
        api_key=api_key,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        confidence_threshold=confidence_threshold,
    )

    if raw_workflow_output_path:
        with open(raw_workflow_output_path, "w", encoding="utf-8") as f:
            json.dump(workflow_result, f, indent=2)

    predictions = normalize_predictions_from_workflow(workflow_result)
    outputs     = []
    crops_dir   = Path("crops")
    if save_crops:
        crops_dir.mkdir(exist_ok=True)

    for i, pred in enumerate(predictions):
        if pred["confidence"] is not None and pred["confidence"] < confidence_threshold:
            continue
        orientation = label_to_orientation(pred["label"])
        if orientation == "unknown":
            continue
        crop, bbox_xyxy = crop_prediction(image, pred)
        if crop.size == 0:
            continue

        zero_result = check_starts_at_zero_image(crop, orientation)
        crop_path   = None
        if save_crops:
            crop_path = str(crops_dir / f"chart_{i}_{orientation}.png")
            cv2.imwrite(crop_path, crop)

        outputs.append({
            "chart_id":            i,
            "detector_label":      pred["label"],
            "detector_confidence": pred["confidence"],
            "orientation":         orientation,
            "bbox_center_format":  {
                "x": pred["x"], "y": pred["y"],
                "width": pred["width"], "height": pred["height"],
            },
            "bbox_xyxy":           bbox_xyxy,
            "crop_path":           crop_path,
            "start_at_zero_result": zero_result,
        })

    return outputs


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow Workflow + start-at-zero checker")
    parser.add_argument("--image",           required=True,  help="Path to dashboard/chart image")
    parser.add_argument("--api-url",         default="https://detect.roboflow.com")
    parser.add_argument("--api-key",         default=os.getenv("ROBOFLOW_API_KEY"))
    parser.add_argument("--workspace-name",  required=True)
    parser.add_argument("--workflow-id",     required=True)
    parser.add_argument("--conf",            type=float, default=0.25)
    parser.add_argument("--save-crops",      action="store_true")
    parser.add_argument("--output",          default="results.json")
    parser.add_argument("--raw-workflow-output", default="workflow_raw.json")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Use --api-key or set ROBOFLOW_API_KEY.")

    final_results = run_pipeline(
        image_path=args.image,
        api_url=args.api_url,
        api_key=args.api_key,
        workspace_name=args.workspace_name,
        workflow_id=args.workflow_id,
        confidence_threshold=args.conf,
        save_crops=args.save_crops,
        raw_workflow_output_path=args.raw_workflow_output,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(final_results), f, indent=2)

    print(json.dumps(make_json_safe(final_results), indent=2))
    print(f"\nSaved results to: {args.output}")
    print(f"Saved raw workflow response to: {args.raw_workflow_output}")
