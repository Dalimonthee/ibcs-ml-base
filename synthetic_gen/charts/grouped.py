"""Grouped bar chart renderer (vertical and horizontal)."""

from __future__ import annotations

import random

import matplotlib.axes
import numpy as np

from ..styling import (
    StyleParams, apply_style, random_categories, random_multi_series_data,
    random_series_names,
)


def render_grouped(ax: matplotlib.axes.Axes, rng: random.Random,
                   style: StyleParams, horizontal: bool = False) -> None:
    """Render a grouped bar chart on the given axes."""
    n_categories = rng.randint(2, 10)
    n_series = rng.randint(2, 5)

    categories = random_categories(rng, n_categories)
    series_names = random_series_names(rng, n_series)
    data = random_multi_series_data(rng, n_categories, n_series)

    x = np.arange(n_categories)
    total_width = min(style.bar_width, 0.8)
    bar_w = total_width / n_series
    offsets = np.linspace(
        -(total_width - bar_w) / 2, (total_width - bar_w) / 2, n_series
    )

    for i, (offset, series_data) in enumerate(zip(offsets, data)):
        color = style.palette[i % len(style.palette)]
        label = series_names[i]

        if horizontal:
            ax.barh(
                x + offset, series_data, height=bar_w,
                color=color, edgecolor=style.edge_color,
                alpha=style.bar_alpha, label=label,
            )
        else:
            ax.bar(
                x + offset, series_data, width=bar_w,
                color=color, edgecolor=style.edge_color,
                alpha=style.bar_alpha, label=label,
            )

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if style.tick_rotation:
            ax.tick_params(axis="x", rotation=style.tick_rotation)

    if style.show_legend:
        ax.legend(loc=style.legend_loc, fontsize=style.value_fontsize)

    apply_style(ax, style)
