"""Dashboard layout templates and slot utilities for multi-chart composition."""

from __future__ import annotations

import random

# Each slot is (x, y, w, h) in normalized canvas coordinates (top-left origin).
LAYOUT_SLOTS: dict[str, list[tuple[float, float, float, float]]] = {
    "one_over_two": [
        (0.04, 0.04, 0.92, 0.40),
        (0.04, 0.50, 0.44, 0.46),
        (0.52, 0.50, 0.44, 0.46),
    ],
    "one_over_three": [
        (0.04, 0.04, 0.92, 0.38),
        (0.04, 0.48, 0.28, 0.48),
        (0.36, 0.48, 0.28, 0.48),
        (0.68, 0.48, 0.28, 0.48),
    ],
    "grid_2x2": [
        (0.04, 0.04, 0.44, 0.44),
        (0.52, 0.04, 0.44, 0.44),
        (0.04, 0.52, 0.44, 0.44),
        (0.52, 0.52, 0.44, 0.44),
    ],
    "row_2": [
        (0.04, 0.12, 0.44, 0.76),
        (0.52, 0.12, 0.44, 0.76),
    ],
    "row_3": [
        (0.03, 0.12, 0.29, 0.76),
        (0.355, 0.12, 0.29, 0.76),
        (0.68, 0.12, 0.29, 0.76),
    ],
    "column_2": [
        (0.12, 0.04, 0.76, 0.44),
        (0.12, 0.52, 0.76, 0.44),
    ],
    "column_3": [
        (0.12, 0.03, 0.76, 0.29),
        (0.12, 0.355, 0.76, 0.29),
        (0.12, 0.68, 0.76, 0.29),
    ],
    "one_large_one_small": [
        (0.04, 0.08, 0.58, 0.84),
        (0.66, 0.08, 0.30, 0.84),
    ],
}

LAYOUTS_BY_COUNT: dict[int, list[str]] = {}
for name, slots in LAYOUT_SLOTS.items():
    LAYOUTS_BY_COUNT.setdefault(len(slots), []).append(name)

DEFAULT_CHARTS_PER_IMAGE: dict[int, float] = {
    1: 0.40,
    2: 0.25,
    3: 0.25,
    4: 0.10,
}


def pick_chart_count(rng: random.Random,
                     weights: dict[int, float] | None = None) -> int:
    """Sample how many charts to place on one canvas."""
    w = weights or DEFAULT_CHARTS_PER_IMAGE
    counts = sorted(w.keys())
    probs = [w[c] for c in counts]
    return rng.choices(counts, weights=probs, k=1)[0]


def pick_layout(rng: random.Random, n_charts: int) -> str:
    """Pick a layout template with exactly n_charts slots."""
    options = LAYOUTS_BY_COUNT.get(n_charts)
    if not options:
        raise ValueError(f"No layout defined for {n_charts} charts")
    return rng.choice(options)


def _shrink_slot(slot: tuple[float, float, float, float],
                 gap_px: int,
                 canvas_w: int,
                 canvas_h: int) -> tuple[float, float, float, float]:
    """Shrink a normalized slot inward to enforce whitespace between charts."""
    x, y, w, h = slot
    gx = gap_px / canvas_w
    gy = gap_px / canvas_h
    x = min(max(x + gx / 2, 0.0), 1.0)
    y = min(max(y + gy / 2, 0.0), 1.0)
    w = max(w - gx, 0.01)
    h = max(h - gy, 0.01)
    if x + w > 1.0:
        w = 1.0 - x
    if y + h > 1.0:
        h = 1.0 - y
    return (x, y, w, h)


def get_slots(layout: str,
              canvas_size: tuple[int, int] = (640, 640),
              gap_px: int = 24) -> list[tuple[float, float, float, float]]:
    """Return gap-shrunk normalized slot rectangles for a layout name."""
    if layout not in LAYOUT_SLOTS:
        raise KeyError(f"Unknown layout: {layout}")
    canvas_w, canvas_h = canvas_size
    return [
        _shrink_slot(slot, gap_px, canvas_w, canvas_h)
        for slot in LAYOUT_SLOTS[layout]
    ]
