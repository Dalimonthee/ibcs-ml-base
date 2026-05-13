"""Single vertical bar chart renderer."""

from __future__ import annotations

import random

import matplotlib.axes
import numpy as np

from ..styling import (
    StyleParams, apply_style, random_categories, random_data, random_style,
)


def render_vertical(ax: matplotlib.axes.Axes, rng: random.Random,
                    style: StyleParams) -> None:
    """Render a single vertical bar chart on the given axes."""
    n_bars = rng.randint(2, 15)
    categories = random_categories(rng, n_bars)
    values = random_data(rng, n_bars)

    colors = [style.palette[i % len(style.palette)] for i in range(n_bars)]

    bars = ax.bar(
        categories, values,
        width=style.bar_width,
        color=colors,
        edgecolor=style.edge_color,
        alpha=style.bar_alpha,
    )

    if style.tick_rotation:
        ax.tick_params(axis="x", rotation=style.tick_rotation)

    if style.show_values:
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            va = "bottom" if y_pos >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.0f}", ha="center", va=va,
                fontsize=style.value_fontsize,
            )

    apply_style(ax, style)
