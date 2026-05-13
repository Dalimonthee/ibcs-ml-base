"""Chart factory: weighted random chart type selection and dispatch."""

from __future__ import annotations

import random
from typing import Callable

import matplotlib.axes
import matplotlib.pyplot as plt

from .charts.vertical import render_vertical
from .charts.horizontal import render_horizontal
from .charts.grouped import render_grouped
from .charts.stacked import render_stacked
from .charts.overlapping import render_overlapping
from .styling import StyleParams, random_style


CHART_RENDERERS: dict[str, Callable] = {
    "vertical": lambda ax, rng, style: render_vertical(ax, rng, style),
    "horizontal": lambda ax, rng, style: render_horizontal(ax, rng, style),
    "grouped_vertical": lambda ax, rng, style: render_grouped(ax, rng, style, horizontal=False),
    "grouped_horizontal": lambda ax, rng, style: render_grouped(ax, rng, style, horizontal=True),
    "stacked_vertical": lambda ax, rng, style: render_stacked(ax, rng, style, horizontal=False),
    "stacked_horizontal": lambda ax, rng, style: render_stacked(ax, rng, style, horizontal=True),
    "overlapping_vertical": lambda ax, rng, style: render_overlapping(ax, rng, style, horizontal=False),
    "overlapping_horizontal": lambda ax, rng, style: render_overlapping(ax, rng, style, horizontal=True),
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "vertical": 0.20,
    "horizontal": 0.15,
    "grouped_vertical": 0.15,
    "grouped_horizontal": 0.10,
    "stacked_vertical": 0.15,
    "stacked_horizontal": 0.10,
    "overlapping_vertical": 0.10,
    "overlapping_horizontal": 0.05,
}


def pick_chart_type(rng: random.Random,
                    weights: dict[str, float] | None = None) -> str:
    """Select a chart type based on configured weights."""
    w = weights or DEFAULT_WEIGHTS
    types = list(w.keys())
    probs = [w[t] for t in types]
    return rng.choices(types, weights=probs, k=1)[0]


def create_chart(rng: random.Random,
                 chart_type: str | None = None,
                 weights: dict[str, float] | None = None,
                 ) -> tuple[plt.Figure, matplotlib.axes.Axes, StyleParams, str]:
    """Create a random bar chart figure.

    Returns (fig, ax, style, chart_type).
    """
    if chart_type is None:
        chart_type = pick_chart_type(rng, weights)

    style = random_style(rng)
    fig, ax = plt.subplots(figsize=(style.fig_width, style.fig_height))
    fig.patch.set_facecolor(style.bg_color)

    renderer = CHART_RENDERERS[chart_type]
    renderer(ax, rng, style)

    fig.tight_layout()
    return fig, ax, style, chart_type
