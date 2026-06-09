#!/usr/bin/env python3
"""Run overlap eval and save a PNG summary of all metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline
from metrics import ChartMetrics, corner_bboxes_to_xywh, evaluate_chart, evaluate_dataset
from synthetic_bw import generate_batch

EVAL_OUTPUTS = Path(__file__).resolve().parent / "eval_outputs"
DEFAULT_OUT = EVAL_OUTPUTS / "metrics_report.png"


def run_eval(n: int, seed: int) -> tuple[list[ChartMetrics], object]:
    params = pipeline.eval_params()
    chart_metrics: list[ChartMetrics] = []
    for chart in generate_batch(seed, n):
        out = pipeline.run_pipeline(chart.bgr, params)
        pred = corner_bboxes_to_xywh(out["corner_bboxes"])
        chart_metrics.append(evaluate_chart(chart.bars, pred))
    return chart_metrics, params


def render_metrics_png(
    chart_metrics: list[ChartMetrics],
    dataset,
    params: pipeline.Params,
    *,
    n: int,
    seed: int,
    out_path: Path,
) -> Path:
    """Build and save a multi-panel metrics figure."""
    total_tp = sum(m.tp for m in chart_metrics)
    total_fp = sum(m.fp for m in chart_metrics)
    total_fn = sum(m.fn for m in chart_metrics)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )

    f1s = [m.f1 for m in chart_metrics]
    precisions = [m.precision for m in chart_metrics]
    recalls = [m.recall for m in chart_metrics]
    ious = [m.mean_iou for m in chart_metrics]

    fig = plt.figure(figsize=(14, 9), facecolor="#fafafa")
    fig.suptitle(
        "Corner bbox overlap eval (orange boxes vs black-bar GT)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # --- Summary text panel ---
    ax_text = fig.add_axes([0.05, 0.52, 0.42, 0.40])
    ax_text.axis("off")
    summary = (
        f"Dataset\n"
        f"  Charts evaluated:     {dataset.n_charts}\n"
        f"  Random seed:          {seed}\n"
        f"  IoU match threshold:  0.50\n"
        f"  GT targets:           black bars only (white = decoy)\n\n"
        f"Macro averages (per-chart mean)\n"
        f"  Precision:            {dataset.mean_precision:.3f}\n"
        f"  Recall:               {dataset.mean_recall:.3f}\n"
        f"  F1:                   {dataset.mean_f1:.3f}\n"
        f"  Mean IoU (matched):   {dataset.mean_iou:.3f}\n"
        f"  Charts with F1 ≥ 0.5: {dataset.charts_f1_above_half} / {dataset.n_charts}\n\n"
        f"Micro totals (pooled TP/FP/FN)\n"
        f"  TP / FP / FN:         {total_tp} / {total_fp} / {total_fn}\n"
        f"  Precision:            {micro_p:.3f}\n"
        f"  Recall:               {micro_r:.3f}\n"
        f"  F1:                   {micro_f1:.3f}\n\n"
        f"Per-chart ranges\n"
        f"  F1:       [{min(f1s):.3f}, {max(f1s):.3f}]\n"
        f"  Precision:[{min(precisions):.3f}, {max(precisions):.3f}]\n"
        f"  Recall:   [{min(recalls):.3f}, {max(recalls):.3f}]\n"
        f"  Mean IoU: [{min(ious):.3f}, {max(ious):.3f}]\n\n"
        f"Eval params\n"
        f"  oriented_source:          {params.oriented_source}\n"
        f"  oriented_threshold_frac:  {params.oriented_threshold_frac}\n"
        f"  floor_enabled:            {params.floor_enabled}\n"
        f"  bbox bottom align:        yes (lowest shared edge)"
    )
    ax_text.text(
        0.0, 1.0, summary,
        transform=ax_text.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"),
    )

    # --- Metric bar chart (macro) ---
    ax_bars = fig.add_axes([0.55, 0.58, 0.40, 0.34])
    names = ["Precision", "Recall", "F1", "Mean IoU"]
    values = [
        dataset.mean_precision,
        dataset.mean_recall,
        dataset.mean_f1,
        dataset.mean_iou,
    ]
    colors = ["#2e6da4", "#5cb85c", "#f0ad4e", "#9b59b6"]
    bars = ax_bars.bar(names, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax_bars.set_ylim(0, 1.05)
    ax_bars.set_ylabel("Score")
    ax_bars.set_title("Macro-averaged metrics")
    ax_bars.axhline(0.5, color="#999999", linestyle="--", linewidth=1, label="0.5")
    for bar, val in zip(bars, values):
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # --- F1 histogram ---
    ax_hist = fig.add_axes([0.55, 0.10, 0.40, 0.38])
    ax_hist.hist(f1s, bins=20, range=(0, 1), color="#f0ad4e", edgecolor="#333333", alpha=0.85)
    ax_hist.axvline(dataset.mean_f1, color="#c0392b", linewidth=2, label=f"mean F1={dataset.mean_f1:.3f}")
    ax_hist.axvline(0.5, color="#999999", linestyle="--", linewidth=1, label="F1 = 0.5")
    ax_hist.set_xlabel("Per-chart F1")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("F1 distribution across charts")
    ax_hist.legend(loc="upper left", fontsize=9)

    # --- Per-chart F1 sparkline / sorted ---
    ax_sorted = fig.add_axes([0.05, 0.08, 0.42, 0.36])
    sorted_f1 = np.sort(f1s)
    ax_sorted.plot(range(len(sorted_f1)), sorted_f1, color="#2e6da4", linewidth=2)
    ax_sorted.fill_between(range(len(sorted_f1)), sorted_f1, alpha=0.25, color="#2e6da4")
    ax_sorted.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    ax_sorted.set_xlim(0, len(sorted_f1) - 1)
    ax_sorted.set_ylim(0, 1.05)
    ax_sorted.set_xlabel("Chart index (sorted by F1)")
    ax_sorted.set_ylabel("F1")
    ax_sorted.set_title("Per-chart F1 (sorted)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="Number of charts")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUT),
        help="Output PNG path",
    )
    args = parser.parse_args()

    chart_metrics, params = run_eval(args.n, args.seed)
    dataset = evaluate_dataset(chart_metrics)
    out_path = render_metrics_png(
        chart_metrics, dataset, params,
        n=args.n, seed=args.seed, out_path=Path(args.output),
    )
    print(f"Saved metrics report: {out_path}")
    print(
        f"  P={dataset.mean_precision:.3f}  R={dataset.mean_recall:.3f}  "
        f"F1={dataset.mean_f1:.3f}  mean_IoU={dataset.mean_iou:.3f}"
    )


if __name__ == "__main__":
    main()
