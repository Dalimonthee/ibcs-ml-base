#!/usr/bin/env python3
"""Render a flowchart of the OpenCV bar-detection pipeline as PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EVAL_OUTPUTS = Path(__file__).resolve().parent / "eval_outputs"
DEFAULT_OUT = EVAL_OUTPUTS / "pipeline_diagram.png"

# Box style
BOX = dict(boxstyle="round,pad=0.35", linewidth=1.2, edgecolor="#334155")
STAGE = {**BOX, "facecolor": "#e0f2fe", "edgecolor": "#0369a1"}
MASK = {**BOX, "facecolor": "#fef3c7", "edgecolor": "#b45309"}
DETECT = {**BOX, "facecolor": "#dcfce7", "edgecolor": "#15803d"}
OPTIONAL = {**BOX, "facecolor": "#f3e8ff", "edgecolor": "#7c3aed", "linestyle": "--"}
OUTPUT = {**BOX, "facecolor": "#ffe4e6", "edgecolor": "#be123c"}


def _box(ax, x, y, w, h, text, style: dict, fontsize=8):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        transform=ax.transData,
        **style,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, x0, y0, x1, y1, color="#64748b"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def render_pipeline_diagram(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 22)
    ax.axis("off")
    ax.set_title(
        "IBCS Bar Detection — OpenCV Pipeline",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    cx = 7.0
    w_main, h_main = 5.2, 0.85
    w_wide, h_wide = 6.8, 0.75
    w_mask, h_mask = 2.0, 0.65

    y = 21.0

    # --- Input ---
    _box(ax, cx, y, w_main, h_main, "Input: BGR image", STAGE)
    y -= 1.1
    _arrow(ax, cx, y + 0.45, cx, y + 0.95)
    _box(ax, cx, y, w_main, h_main, "BGR → RGB + Grayscale\n(optional invert)", STAGE)
    y -= 1.2

    # --- Mask building ---
    _arrow(ax, cx, y + 0.5, cx, y + 1.0)
    _box(ax, cx, y, w_wide, 1.0, "_build_masks(gray)", STAGE, fontsize=9)
    y -= 1.5

    mask_y = y
    masks = [
        ("A", "THRESH_BINARY_INV\n(black bars)"),
        ("B", "inRange\n(grey bars)"),
        ("C", "Canny → hatch\nmorph close"),
        ("filled", "Canny → contours\n→ filled rects"),
        ("combined", "OR(A,B,C,filled)"),
    ]
    xs = [2.2, 4.4, 6.6, 8.8, 11.0]
    for xi, (name, desc) in zip(xs, masks):
        _box(ax, xi, mask_y, w_mask, h_mask + 0.15, f"Mask {name}\n{desc}", MASK, fontsize=6.5)

    y = mask_y - 1.0
    _box(
        ax,
        cx,
        y,
        w_wide,
        0.7,
        "Each mask: morph close/open + filter_by_largest\n(connected-components noise drop)",
        MASK,
        fontsize=7,
    )
    y -= 1.2

    # --- Projection bars ---
    _arrow(ax, cx, y + 0.55, cx, y + 1.05)
    _box(
        ax,
        cx,
        y,
        w_wide,
        1.1,
        "Per mask: projection_bars\n"
        "X-profile runs → Y-profile runs → size/fill/aspect filters",
        DETECT,
        fontsize=7.5,
    )
    y -= 1.3
    _box(ax, cx, y, w_main, h_main, "score_mask → pick winner\n(A,B,C,filled > combined)", DETECT)
    y -= 1.2

    # --- Corners branch ---
    _arrow(ax, cx, y + 0.5, cx, y + 1.0)
    _box(ax, 3.5, y - 0.1, 4.2, 1.5, "Shi-Tomasi\ngoodFeaturesToTrack\n(optional)", OPTIONAL, fontsize=7.5)
    _box(
        ax,
        10.5,
        y - 0.1,
        5.0,
        1.7,
        "Oriented corners (4 kernels)\n"
        "filter2D → peak NMS\n"
        "TL/TR convex & concave",
        DETECT,
        fontsize=7,
    )
    y -= 2.0

    # --- Axis detection ---
    _box(
        ax,
        cx,
        y,
        w_wide + 1.2,
        1.35,
        "Horizontal axis fusion (if bbox_from_corners)\n"
        "• corners: cluster Y from oriented points\n"
        "• morph: horizontal MORPH_OPEN on Canny edges\n"
        "• hough: HoughLinesP (near-horizontal)\n"
        "→ fuse_axes (weighted merge)",
        DETECT,
        fontsize=7,
    )
    y -= 1.6

    # --- BBox building ---
    _arrow(ax, cx, y + 0.65, cx, y + 1.25)
    _box(
        ax,
        cx,
        y,
        w_wide + 0.5,
        1.4,
        "build_bboxes_from_corners\n"
        "Pair TL_convex + TR_convex → width/height checks\n"
        "Close bottom: infer_bar_floor (column walk) or nearest axis below\n"
        "→ align_corner_bboxes_to_lowest_bottom (shared baseline)",
        DETECT,
        fontsize=7,
    )
    y -= 1.7

    # --- Parallel path note ---
    _box(
        ax,
        2.0,
        y + 0.3,
        3.6,
        0.9,
        "winner_boxes\n(projection on\nwinner mask)",
        OUTPUT,
        fontsize=7,
    )

    # --- Output ---
    _arrow(ax, cx, y + 0.85, cx, y + 1.45)
    _box(
        ax,
        cx,
        y,
        w_wide,
        1.0,
        "Draw overlay on RGB\n"
        "green = projection boxes | orange = corner bboxes\n"
        "colored dots = oriented corners | magenta = axes",
        OUTPUT,
        fontsize=7.5,
    )

    # Legend
    leg_y = 1.2
    for i, (label, style) in enumerate(
        [
            ("Stage", STAGE),
            ("Mask ops", MASK),
            ("Detection", DETECT),
            ("Optional", OPTIONAL),
            ("Output", OUTPUT),
        ]
    ):
        xi = 1.5 + i * 2.6
        patch = FancyBboxPatch(
            (xi - 0.35, leg_y - 0.18),
            0.7,
            0.36,
            transform=ax.transData,
            **style,
        )
        ax.add_patch(patch)
        ax.text(xi + 0.55, leg_y, label, fontsize=7, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    path = render_pipeline_diagram(args.output.resolve())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
