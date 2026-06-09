"""Synthetic vertical black-and-white bar charts with per-bar ground-truth boxes."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LabeledChart:
    bgr: np.ndarray
    """Ground-truth boxes for detectable (black) bars only — ``(x, y, w, h)``."""
    bars: list[tuple[int, int, int, int]]
    seed: int


def _bar_to_xywh(bar, ax, img_h: int) -> tuple[int, int, int, int]:
    """Convert a matplotlib bar patch to OpenCV-style (x, y, w, h) pixels."""
    x0, y0 = bar.get_xy()
    w_data, h_data = bar.get_width(), bar.get_height()
    corners = np.array(
        [[x0, y0], [x0 + w_data, y0], [x0 + w_data, y0 + h_data], [x0, y0 + h_data]],
        dtype=float,
    )
    pix = ax.transData.transform(corners)
    px = pix[:, 0]
    py = pix[:, 1]
    x = int(np.floor(px.min()))
    x2 = int(np.ceil(px.max()))
    y_top = int(np.floor(img_h - py.max()))
    y_bot = int(np.ceil(img_h - py.min()))
    return max(0, x), max(0, y_top), max(1, x2 - x), max(1, y_bot - y_top)


def generate_bw_vertical_chart(
    rng: random.Random,
    *,
    n_bars: int | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 100,
    show_grid: bool | None = None,
    show_title: bool | None = None,
) -> LabeledChart:
    """Render a vertical B&W bar chart and return BGR image + GT bar boxes."""
    if n_bars is None:
        n_bars = rng.randint(3, 12)
    if figsize is None:
        figsize = (rng.uniform(6.0, 10.0), rng.uniform(4.0, 7.0))
    if show_grid is None:
        show_grid = rng.random() < 0.3
    if show_title is None:
        show_title = rng.random() < 0.4

    categories = [f"C{i}" for i in range(n_bars)]
    values = [abs(rng.gauss(50, 25)) + 5 for _ in range(n_bars)]
    bar_width = rng.uniform(0.4, 0.75)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.set_facecolor("white")

    bar_colors: list[str] = []
    edge_colors: list[str] = []
    for i in range(n_bars):
        if i % 2 == 0:
            bar_colors.append("#000000")
            edge_colors.append("#000000")
        else:
            bar_colors.append("#FFFFFF")
            edge_colors.append("#000000")

    bars = ax.bar(
        categories,
        values,
        width=bar_width,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
    )

    ax.set_xlim(-0.6, n_bars - 0.4)
    ymin = 0.0
    ymax = max(values) * rng.uniform(1.1, 1.35)
    ax.set_ylim(ymin, ymax)

    if show_grid:
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#888888")
    if show_title:
        ax.set_title("Synthetic B&W bars", fontsize=12)

    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")

    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    rgb = rgba[:, :, :3].copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Only black bars (even indices) are detection targets; white bars are decoys.
    gt_bars = [
        _bar_to_xywh(bar, ax, height)
        for i, bar in enumerate(bars)
        if i % 2 == 0
    ]
    plt.close(fig)

    return LabeledChart(bgr=bgr, bars=gt_bars, seed=0)


def generate_batch(
    base_seed: int,
    count: int,
) -> list[LabeledChart]:
    """Generate ``count`` charts with seeds ``base_seed + i``."""
    out: list[LabeledChart] = []
    for i in range(count):
        rng = random.Random(base_seed + i)
        chart = generate_bw_vertical_chart(rng)
        chart.seed = base_seed + i
        out.append(chart)
    return out
