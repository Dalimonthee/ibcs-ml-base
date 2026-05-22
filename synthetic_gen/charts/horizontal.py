"""Single horizontal bar chart renderer."""

from __future__ import annotations

import random

import matplotlib.axes
import numpy as np

from ..styling import (
    StyleParams, apply_style, random_categories, random_data,
)


def render_horizontal(ax: matplotlib.axes.Axes, rng: random.Random,
                      style: StyleParams) -> None:
    """Render a single horizontal bar chart on the given axes."""
    n_bars = rng.randint(2, 15)
    categories = random_categories(rng, n_bars)
    values = random_data(rng, n_bars)

    colors = [style.palette[i % len(style.palette)] for i in range(n_bars)]

    bars = ax.barh(
        categories, values,
        height=style.bar_width,
        color=colors,
        edgecolor=style.edge_color,
        alpha=style.bar_alpha,
    )

    if style.show_values:
        for bar, val in zip(bars, values):
            x_pos = bar.get_width()
            ha = "left" if x_pos >= 0 else "right"
            ax.text(
                x_pos, bar.get_y() + bar.get_height() / 2,
                f" {val:.0f}", ha=ha, va="center",
                fontsize=style.value_fontsize,
            )

    apply_style(ax, style)
