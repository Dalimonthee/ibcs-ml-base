"""Compose chart images onto a canvas and compute YOLO bounding boxes."""

from __future__ import annotations

import io
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def _add_noise_elements(draw: ImageDraw.Draw, rng: random.Random,
                        canvas_w: int, canvas_h: int,
                        chart_box: tuple[int, int, int, int]) -> None:
    """Add random rectangles and lines around the chart to simulate dashboard UI."""
    cx1, cy1, cx2, cy2 = chart_box
    n_elements = rng.randint(0, 6)

    for _ in range(n_elements):
        elem_type = rng.choice(["rect", "rect", "line"])
        gray = rng.randint(180, 240)
        color = (gray, gray, gray, 255)

        if elem_type == "rect":
            rw = rng.randint(30, 150)
            rh = rng.randint(20, 80)

            # Place outside the chart bounding box
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

    # Scale chart to fit within 40-95% of canvas dimensions
    scale_factor = rng.uniform(0.4, 0.95)
    max_chart_w = int(canvas_w * scale_factor)
    max_chart_h = int(canvas_h * scale_factor)

    chart_img.thumbnail((max_chart_w, max_chart_h), Image.LANCZOS)
    cw, ch = chart_img.size

    # Random placement within canvas bounds
    max_x = canvas_w - cw
    max_y = canvas_h - ch
    paste_x = rng.randint(0, max(0, max_x))
    paste_y = rng.randint(0, max(0, max_y))

    bg_color = _random_bg_color(rng)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    chart_box = (paste_x, paste_y, paste_x + cw, paste_y + ch)

    draw = ImageDraw.Draw(canvas)
    if rng.random() < 0.5:
        _add_noise_elements(draw, rng, canvas_w, canvas_h, chart_box)

    chart_rgb = chart_img.convert("RGB")
    canvas.paste(chart_rgb, (paste_x, paste_y))

    x_center = (paste_x + cw / 2) / canvas_w
    y_center = (paste_y + ch / 2) / canvas_h
    bbox_w = cw / canvas_w
    bbox_h = ch / canvas_h

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    bbox_w = max(0.0, min(1.0, bbox_w))
    bbox_h = max(0.0, min(1.0, bbox_h))

    plt.close(fig)

    return canvas, (x_center, y_center, bbox_w, bbox_h)
