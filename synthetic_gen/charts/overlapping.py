"""Overlapping bar chart renderer (vertical and horizontal)."""

from __future__ import annotations

import random

import matplotlib.axes
import numpy as np

from ..styling import (
    StyleParams, apply_style, random_categories, random_multi_series_data,
    random_series_names,
)


def render_overlapping(ax: matplotlib.axes.Axes, rng: random.Random,
                       style: StyleParams, horizontal: bool = False) -> None:
    """Render an overlapping bar chart on the given axes.

    Bars from different series are drawn at the same position with decreasing
    widths and partial transparency so all series remain visible.
    """
    n_categories = rng.randint(2, 10)
    n_series = rng.randint(2, 4)

    categories = random_categories(rng, n_categories)
    series_names = random_series_names(rng, n_series)
    data = random_multi_series_data(
        rng, n_categories, n_series, allow_negative=False
    )

    x = np.arange(n_categories)

    widths = np.linspace(style.bar_width, style.bar_width * 0.4, n_series)
    alphas = np.linspace(0.5, 0.95, n_series)

    sort_order = np.argsort(-data.mean(axis=1))
    data = data[sort_order]

    for i, (series_data, w, a) in enumerate(zip(data, widths, alphas)):
        color = style.palette[i % len(style.palette)]
        label = series_names[sort_order[i]]

        if horizontal:
            ax.barh(
                x, series_data, height=w,
                color=color, edgecolor=style.edge_color,
                alpha=a, label=label, zorder=i + 1,
            )
        else:
            ax.bar(
                x, series_data, width=w,
                color=color, edgecolor=style.edge_color,
                alpha=a, label=label, zorder=i + 1,
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
