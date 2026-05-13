"""Stacked bar chart renderer (vertical and horizontal)."""

from __future__ import annotations

import random

import matplotlib.axes
import numpy as np

from ..styling import (
    StyleParams, apply_style, random_categories, random_multi_series_data,
    random_series_names,
)


def render_stacked(ax: matplotlib.axes.Axes, rng: random.Random,
                   style: StyleParams, horizontal: bool = False) -> None:
    """Render a stacked bar chart on the given axes."""
    n_categories = rng.randint(2, 10)
    n_series = rng.randint(2, 5)

    categories = random_categories(rng, n_categories)
    series_names = random_series_names(rng, n_series)
    data = random_multi_series_data(
        rng, n_categories, n_series, allow_negative=False
    )

    x = np.arange(n_categories)
    bottoms = np.zeros(n_categories)

    for i, series_data in enumerate(data):
        color = style.palette[i % len(style.palette)]
        label = series_names[i]

        if horizontal:
            ax.barh(
                x, series_data, left=bottoms, height=style.bar_width,
                color=color, edgecolor=style.edge_color,
                alpha=style.bar_alpha, label=label,
            )
        else:
            ax.bar(
                x, series_data, bottom=bottoms, width=style.bar_width,
                color=color, edgecolor=style.edge_color,
                alpha=style.bar_alpha, label=label,
            )
        bottoms += series_data

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if style.tick_rotation:
            ax.tick_params(axis="x", rotation=style.tick_rotation)

    if style.show_values:
        totals = data.sum(axis=0)
        for j, total in enumerate(totals):
            if horizontal:
                ax.text(
                    total, j, f" {total:.0f}",
                    ha="left", va="center",
                    fontsize=style.value_fontsize,
                )
            else:
                ax.text(
                    j, total, f"{total:.0f}",
                    ha="center", va="bottom",
                    fontsize=style.value_fontsize,
                )

    if style.show_legend:
        ax.legend(loc=style.legend_loc, fontsize=style.value_fontsize)

    apply_style(ax, style)
