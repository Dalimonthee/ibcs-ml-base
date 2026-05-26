"""Compose chart images onto a canvas and compute YOLO bounding boxes."""

from __future__ import annotations

import io
import random

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .layouts import get_slots


def fig_to_image(fig: plt.Figure, dpi: int = 100) -> Image.Image:
    """Render a matplotlib figure to a PIL Image (RGBA)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.1, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img


def _random_bg_color(rng: random.Random) -> tuple[int, int, int]:
    """Generate a random light background color for the canvas."""
    presets = [
        (255, 255, 255), (245, 245, 245), (240, 240, 240),
        (250, 250, 250), (248, 248, 255), (255, 250, 240),
        (240, 248, 255), (245, 245, 220), (255, 245, 238),
        (240, 255, 240), (230, 230, 250), (255, 240, 245),
    ]
    return rng.choice(presets)


def _normalize_bbox(paste_x: int, paste_y: int, cw: int, ch: int,
                  canvas_w: int, canvas_h: int) -> tuple[float, float, float, float]:
    """Convert pixel paste coordinates to YOLO normalized bbox."""
    x_center = (paste_x + cw / 2) / canvas_w
    y_center = (paste_y + ch / 2) / canvas_h
    bbox_w = cw / canvas_w
    bbox_h = ch / canvas_h
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, bbox_w)),
        max(0.0, min(1.0, bbox_h)),
    )


def _paste_and_bbox(canvas: Image.Image,
                    chart_img: Image.Image,
                    paste_x: int,
                    paste_y: int) -> tuple[tuple[int, int, int, int],
                                           tuple[float, float, float, float]]:
    """Paste chart onto canvas and return pixel box + normalized YOLO bbox."""
    canvas_w, canvas_h = canvas.size
    chart_rgb = chart_img.convert("RGB")
    cw, ch = chart_rgb.size
    canvas.paste(chart_rgb, (paste_x, paste_y))
    pixel_box = (paste_x, paste_y, paste_x + cw, paste_y + ch)
    return pixel_box, _normalize_bbox(paste_x, paste_y, cw, ch, canvas_w, canvas_h)


def _add_noise_elements(draw: ImageDraw.Draw, rng: random.Random,
                        canvas_w: int, canvas_h: int,
                        chart_box: tuple[int, int, int, int]) -> None:
    """Add random rectangles and lines around the chart to simulate dashboard UI."""
    cx1, cy1, cx2, cy2 = chart_box
    n_elements = rng.randint(0, 6)

    for _ in range(n_elements):
        elem_type = rng.choice(["rect", "rect", "line"])
        gray = rng.randint(180, 240)
        color = (gray, gray, gray)

        if elem_type == "rect":
            rw = rng.randint(30, 150)
            rh = rng.randint(20, 80)

            side = rng.choice(["top", "bottom", "left", "right"])
            if side == "top" and cy1 > rh + 5:
                rx = rng.randint(0, canvas_w - rw)
                ry = rng.randint(0, cy1 - rh)
            elif side == "bottom" and cy2 < canvas_h - rh - 5:
                rx = rng.randint(0, canvas_w - rw)
                ry = rng.randint(cy2 + 2, canvas_h - rh)
            elif side == "left" and cx1 > rw + 5:
                rx = rng.randint(0, cx1 - rw)
                ry = rng.randint(0, canvas_h - rh)
            elif side == "right" and cx2 < canvas_w - rw - 5:
                rx = rng.randint(cx2 + 2, canvas_w - rw)
                ry = rng.randint(0, canvas_h - rh)
            else:
                continue

            draw.rectangle([rx, ry, rx + rw, ry + rh], fill=color)

        elif elem_type == "line":
            if rng.random() < 0.5:
                y = rng.randint(0, canvas_h)
                draw.line([(0, y), (canvas_w, y)], fill=color, width=1)
            else:
                x = rng.randint(0, canvas_w)
                draw.line([(x, 0), (x, canvas_h)], fill=color, width=1)


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Return the bounding union of pixel boxes."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _fit_chart_in_slot(chart_img: Image.Image,
                       slot_x: int, slot_y: int,
                       slot_w: int, slot_h: int,
                       min_slot_fill: float,
                       rng: random.Random) -> tuple[Image.Image, int, int]:
    """Scale chart to fit within a slot and return (image, paste_x, paste_y)."""
    max_w = max(1, int(slot_w * min_slot_fill))
    max_h = max(1, int(slot_h * min_slot_fill))
    fitted = chart_img.copy()
    fitted.thumbnail((max_w, max_h), Image.LANCZOS)
    cw, ch = fitted.size

    max_offset_x = max(0, slot_w - cw)
    max_offset_y = max(0, slot_h - ch)
    offset_x = rng.randint(0, max_offset_x) if max_offset_x else 0
    offset_y = rng.randint(0, max_offset_y) if max_offset_y else 0

    paste_x = slot_x + offset_x
    paste_y = slot_y + offset_y
    return fitted, paste_x, paste_y


def compose(fig: plt.Figure, rng: random.Random,
            canvas_size: tuple[int, int] = (640, 640),
            dpi: int = 100,
            ) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Render chart onto a canvas and return (image, normalized_bbox).

    The bbox is in YOLO format: (x_center, y_center, width, height),
    all normalized to [0, 1].
    """
    canvas_w, canvas_h = canvas_size

    chart_img = fig_to_image(fig, dpi=dpi)

    scale_factor = rng.uniform(0.4, 0.95)
    max_chart_w = int(canvas_w * scale_factor)
    max_chart_h = int(canvas_h * scale_factor)

    chart_img.thumbnail((max_chart_w, max_chart_h), Image.LANCZOS)
    cw, ch = chart_img.size

    max_x = canvas_w - cw
    max_y = canvas_h - ch
    paste_x = rng.randint(0, max(0, max_x))
    paste_y = rng.randint(0, max(0, max_y))

    bg_color = _random_bg_color(rng)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    draw = ImageDraw.Draw(canvas)
    if rng.random() < 0.5:
        chart_box = (paste_x, paste_y, paste_x + cw, paste_y + ch)
        _add_noise_elements(draw, rng, canvas_w, canvas_h, chart_box)

    _, bbox = _paste_and_bbox(canvas, chart_img, paste_x, paste_y)

    plt.close(fig)

    return canvas, bbox


def compose_multi(figures: list[plt.Figure],
                  rng: random.Random,
                  layout: str,
                  canvas_size: tuple[int, int] = (640, 640),
                  dpi: int = 100,
                  gap_px: int = 24,
                  min_slot_fill: float = 0.85,
                  ) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """Render multiple charts onto a dashboard canvas with one bbox per chart."""
    canvas_w, canvas_h = canvas_size
    slots = get_slots(layout, canvas_size=canvas_size, gap_px=gap_px)

    if len(figures) != len(slots):
        raise ValueError(
            f"Layout {layout!r} expects {len(slots)} charts, got {len(figures)}"
        )

    bg_color = _random_bg_color(rng)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    pixel_boxes: list[tuple[int, int, int, int]] = []
    bboxes: list[tuple[float, float, float, float]] = []

    for fig, slot in zip(figures, slots):
        sx, sy, sw, sh = slot
        slot_x = int(sx * canvas_w)
        slot_y = int(sy * canvas_h)
        slot_w = max(1, int(sw * canvas_w))
        slot_h = max(1, int(sh * canvas_h))

        chart_img = fig_to_image(fig, dpi=dpi)
        fitted, paste_x, paste_y = _fit_chart_in_slot(
            chart_img, slot_x, slot_y, slot_w, slot_h, min_slot_fill, rng,
        )
        pixel_box, bbox = _paste_and_bbox(canvas, fitted, paste_x, paste_y)
        pixel_boxes.append(pixel_box)
        bboxes.append(bbox)
        plt.close(fig)

    if rng.random() < 0.5 and pixel_boxes:
        draw = ImageDraw.Draw(canvas)
        union = _union_box(pixel_boxes)
        _add_noise_elements(draw, rng, canvas_w, canvas_h, union)

    return canvas, bboxes
