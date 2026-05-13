"""Random style parameter generation for synthetic bar charts."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.colors as mcolors
import numpy as np


PALETTES = [
    ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
     "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"],
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087",
     "#f95d6a", "#ff7c43", "#ffa600"],
    ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    ["#606060", "#808080", "#a0a0a0", "#c0c0c0", "#d0d0d0"],
    ["#333333", "#555555", "#777777", "#999999", "#bbbbbb"],
    ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd"],
    ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"],
    ["#e63946", "#f1faee", "#a8dadc", "#457b9d", "#1d3557"],
    ["#2b2d42", "#8d99ae", "#edf2f4", "#ef233c", "#d90429"],
]

FONT_FAMILIES = ["sans-serif", "serif", "monospace"]

GRID_STYLES = ["solid", "dashed", "dotted", "dashdot"]

LEGEND_LOCATIONS = [
    "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
]

CATEGORY_WORDS = [
    "Q1", "Q2", "Q3", "Q4", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "2020", "2021", "2022",
    "2023", "2024", "2025", "North", "South", "East", "West", "USA",
    "EU", "APAC", "LATAM", "Product A", "Product B", "Product C",
    "Alpha", "Beta", "Gamma", "Delta", "Region 1", "Region 2",
    "Region 3", "Group A", "Group B", "Group C", "Dept X", "Dept Y",
]

SERIES_NAMES = [
    "Revenue", "Cost", "Profit", "Sales", "Returns", "Budget",
    "Actual", "Target", "Forecast", "Plan", "Series A", "Series B",
    "Series C", "2024", "2025", "Male", "Female", "Online", "Offline",
]

TITLE_TEMPLATES = [
    "Revenue by {}", "Sales per {}", "{} Performance", "Monthly {}",
    "Quarterly {}", "{} Breakdown", "{} Comparison", "{} Overview",
    "Annual {}", "{} Distribution", "{} Analysis", "{} Report",
]

TITLE_SUBJECTS = [
    "Revenue", "Sales", "Profit", "Cost", "Growth", "Market Share",
    "Budget", "Output", "Performance", "Expenses", "Units", "Volume",
]


@dataclass
class StyleParams:
    """Container for all randomized style parameters for a single chart."""
    palette: list[str] = field(default_factory=list)
    bar_width: float = 0.6
    fig_width: float = 8.0
    fig_height: float = 6.0
    dpi: int = 100
    bg_color: str = "white"
    title: Optional[str] = None
    title_fontsize: int = 14
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    axis_label_fontsize: int = 11
    tick_label_fontsize: int = 9
    tick_rotation: int = 0
    font_family: str = "sans-serif"
    show_grid: bool = False
    grid_axis: str = "y"
    grid_style: str = "dashed"
    grid_alpha: float = 0.5
    show_legend: bool = False
    legend_loc: str = "upper right"
    show_values: bool = False
    value_fontsize: int = 8
    spines_visible: dict = field(default_factory=lambda: {
        "top": True, "right": True, "bottom": True, "left": True
    })
    edge_color: Optional[str] = None
    bar_alpha: float = 1.0


def random_style(rng: random.Random) -> StyleParams:
    """Generate a fully randomized StyleParams using the given RNG."""
    palette = rng.choice(PALETTES).copy()
    rng.shuffle(palette)

    aspect_w = rng.uniform(4.0, 12.0)
    aspect_h = rng.uniform(3.0, 8.0)

    bg_choices = [
        "white", "#ffffff", "#f9f9f9", "#f5f5f5", "#f0f0f0",
        "#fafafa", "#fffdf7", "#f7f9fc", "#fdf6f0", "#f0f4f8",
    ]

    spine_preset = rng.choice([
        {"top": True, "right": True, "bottom": True, "left": True},
        {"top": False, "right": False, "bottom": True, "left": True},
        {"top": False, "right": False, "bottom": False, "left": False},
        {"top": False, "right": False, "bottom": True, "left": False},
    ])

    show_title = rng.random() < 0.7
    title = None
    if show_title:
        tmpl = rng.choice(TITLE_TEMPLATES)
        subj = rng.choice(TITLE_SUBJECTS)
        title = tmpl.format(subj)

    show_xlabel = rng.random() < 0.5
    show_ylabel = rng.random() < 0.5

    return StyleParams(
        palette=palette,
        bar_width=rng.uniform(0.3, 0.9),
        fig_width=aspect_w,
        fig_height=aspect_h,
        dpi=rng.choice([72, 96, 100, 120, 150]),
        bg_color=rng.choice(bg_choices),
        title=title,
        title_fontsize=rng.randint(10, 20),
        xlabel=rng.choice(CATEGORY_WORDS) if show_xlabel else None,
        ylabel=rng.choice(TITLE_SUBJECTS) if show_ylabel else None,
        axis_label_fontsize=rng.randint(9, 15),
        tick_label_fontsize=rng.randint(7, 13),
        tick_rotation=rng.choice([0, 0, 0, 15, 30, 45, 60, 90]),
        font_family=rng.choice(FONT_FAMILIES),
        show_grid=rng.random() < 0.5,
        grid_axis=rng.choice(["x", "y", "both"]),
        grid_style=rng.choice(GRID_STYLES),
        grid_alpha=rng.uniform(0.2, 0.8),
        show_legend=rng.random() < 0.5,
        legend_loc=rng.choice(LEGEND_LOCATIONS),
        show_values=rng.random() < 0.3,
        value_fontsize=rng.randint(6, 11),
        spines_visible=spine_preset,
        edge_color=rng.choice([None, None, "black", "#333333", "white"]),
        bar_alpha=rng.choice([1.0, 1.0, 1.0, 0.85, 0.9, 0.95]),
    )


def random_categories(rng: random.Random, n: int) -> list[str]:
    """Pick n random category labels."""
    pool = CATEGORY_WORDS.copy()
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return [pool[i % len(pool)] + f"_{i}" for i in range(n)]


def random_series_names(rng: random.Random, n: int) -> list[str]:
    """Pick n random series names for grouped/stacked charts."""
    pool = SERIES_NAMES.copy()
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return [pool[i % len(pool)] + f"_{i}" for i in range(n)]


def random_data(rng: random.Random, n_bars: int,
                allow_negative: bool = True) -> np.ndarray:
    """Generate random bar values."""
    base = rng.uniform(5, 500)
    spread = rng.uniform(0.2, 1.5)
    values = np.array([rng.gauss(base, base * spread) for _ in range(n_bars)])
    if allow_negative and rng.random() < 0.15:
        neg_count = rng.randint(1, max(1, n_bars // 3))
        indices = rng.sample(range(n_bars), neg_count)
        for i in indices:
            values[i] = -abs(values[i]) * rng.uniform(0.2, 0.8)
    return values


def random_multi_series_data(rng: random.Random, n_categories: int,
                              n_series: int,
                              allow_negative: bool = False) -> np.ndarray:
    """Generate data for grouped/stacked charts: shape (n_series, n_categories)."""
    data = np.zeros((n_series, n_categories))
    for s in range(n_series):
        data[s] = random_data(rng, n_categories, allow_negative=allow_negative)
        data[s] = np.abs(data[s])
    return data


def apply_style(ax, style: StyleParams) -> None:
    """Apply common style parameters to an axes object."""
    if style.title:
        ax.set_title(style.title, fontsize=style.title_fontsize,
                     fontfamily=style.font_family)
    if style.xlabel:
        ax.set_xlabel(style.xlabel, fontsize=style.axis_label_fontsize,
                      fontfamily=style.font_family)
    if style.ylabel:
        ax.set_ylabel(style.ylabel, fontsize=style.axis_label_fontsize,
                      fontfamily=style.font_family)

    ax.tick_params(axis="both", labelsize=style.tick_label_fontsize)

    for spine_name, visible in style.spines_visible.items():
        ax.spines[spine_name].set_visible(visible)

    if style.show_grid:
        ax.grid(True, axis=style.grid_axis, linestyle=style.grid_style,
                alpha=style.grid_alpha)
    else:
        ax.grid(False)

    ax.set_facecolor(style.bg_color)
