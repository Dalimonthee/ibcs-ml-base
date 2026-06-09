"""Unit tests for label_detection geometry and parsing (no EasyOCR)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from label_detection import (
    detect_x_axis,
    detect_y_axis,
    detect_data_labels,
    filter_labels_above_x_axis,
    filter_labels_right_of_y_axis,
    parse_data_label_value,
    _relative_position_in_crop,
    _relative_position_to_axis,
)


def _blank_image(h: int = 600, w: int = 800) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def test_parse_data_label_value_plain_and_suffixes():
    assert parse_data_label_value("240") == 240.0
    assert parse_data_label_value("240k") == 240_000.0
    assert parse_data_label_value("2.4m") == 2_400_000.0
    assert parse_data_label_value("12%") == 12.0
    assert parse_data_label_value("N/A") is None


def test_detect_x_axis_finds_horizontal_line_in_bottom_half():
    img = _blank_image()
    y_line = 500
    cv2.line(img, (50, y_line), (750, y_line), (0, 0, 0), 3)
    detected = detect_x_axis(img)
    assert detected is not None
    assert abs(detected - y_line) <= 5


def test_detect_y_axis_finds_vertical_line_in_left_half():
    img = _blank_image()
    x_line = 120
    cv2.line(img, (x_line, 50), (x_line, 550), (0, 0, 0), 3)
    detected = detect_y_axis(img)
    assert detected is not None
    assert abs(detected - x_line) <= 5


def test_filter_labels_above_x_axis():
    boxes = [
        {"text": "100", "confidence": 0.9, "x1": 10, "y1": 100, "x2": 50, "y2": 130, "cx": 30, "cy": 115},
        {"text": "Q1", "confidence": 0.8, "x1": 10, "y1": 520, "x2": 50, "y2": 550, "cx": 30, "cy": 535},
    ]
    filtered = filter_labels_above_x_axis(boxes, x_axis_y=500)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "100"


def test_filter_labels_right_of_y_axis():
    boxes = [
        {"text": "200", "confidence": 0.9, "x1": 200, "y1": 100, "x2": 250, "y2": 130, "cx": 225, "cy": 115},
        {"text": "0", "confidence": 0.8, "x1": 30, "y1": 100, "x2": 60, "y2": 130, "cx": 45, "cy": 115},
    ]
    filtered = filter_labels_right_of_y_axis(boxes, y_axis_x=100)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "200"


def test_detect_data_labels_no_axis(tmp_path: Path):
    crop_path = tmp_path / "crop.png"
    cv2.imwrite(str(crop_path), _blank_image())

    reader = MagicMock()
    reader.readtext.return_value = []

    result = detect_data_labels(crop_path, orientation="vertical", reader=reader)
    assert result["status"] == "no_axis"
    assert result["labels"] == []
    assert result["label_count"] == 0


def test_relative_position_in_crop():
    rel = _relative_position_in_crop(100, 200, 200, 240, 150.0, 220.0, 800, 600)
    assert rel["cx"] == pytest.approx(0.1875)
    assert rel["cy"] == pytest.approx(220 / 600)
    assert rel["x1"] == pytest.approx(0.125)
    assert rel["y2"] == pytest.approx(0.4)


def test_relative_position_to_axis_vertical():
    rel = _relative_position_to_axis(400.0, 200.0, axis_position=500, orientation="vertical", img_w=800, img_h=600)
    assert rel["along_axis"] == pytest.approx(0.5)
    assert rel["from_axis"] == pytest.approx(0.6)


def test_relative_position_to_axis_horizontal():
    rel = _relative_position_to_axis(300.0, 150.0, axis_position=100, orientation="horizontal", img_w=800, img_h=600)
    assert rel["along_axis"] == pytest.approx(0.25)
    assert rel["from_axis"] == pytest.approx(200 / 700)


def test_detect_data_labels_ok_with_mock_reader(tmp_path: Path):
    img = _blank_image()
    y_line = 500
    cv2.line(img, (50, y_line), (750, y_line), (0, 0, 0), 3)
    crop_path = tmp_path / "crop.png"
    cv2.imwrite(str(crop_path), img)

    # EasyOCR-style box: four corners around "150" above the axis
    box = [[200.0, 200.0], [260.0, 200.0], [260.0, 230.0], [200.0, 230.0]]
    reader = MagicMock()
    reader.readtext.return_value = [(box, "150", 0.95)]

    result = detect_data_labels(crop_path, orientation="vertical", reader=reader)
    assert result["status"] == "ok"
    assert result["label_count"] == 1
    assert result["labels"][0]["value"] == 150.0
    rel = result["labels"][0]["relative_position"]
    assert 0.0 <= rel["cx"] <= 1.0
    assert 0.0 <= rel["cy"] <= 1.0
    axis_rel = result["labels"][0]["relative_to_axis"]
    assert "along_axis" in axis_rel
    assert "from_axis" in axis_rel
