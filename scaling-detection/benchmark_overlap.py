#!/usr/bin/env python3
"""CLI benchmark for corner-bbox overlap on synthetic B&W bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import pipeline
from metrics import corner_bboxes_to_xywh, evaluate_chart, evaluate_dataset
from synthetic_bw import generate_batch

EVAL_OUTPUTS = Path(__file__).resolve().parent / "eval_outputs"


def _draw_overlay(bgr, gt, pred):
    """Green = GT (black bars only). Orange = corner-bbox predictions."""
    out = bgr.copy()
    for x, y, w, h in gt:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
    for x, y, w, h in pred:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Number of charts")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save a few overlay PNGs under eval_outputs/",
    )
    parser.add_argument("--overlay-count", type=int, default=5)
    args = parser.parse_args()

    params = pipeline.eval_params()
    chart_metrics = []
    charts = generate_batch(args.seed, args.n)

    if args.save_overlays:
        EVAL_OUTPUTS.mkdir(parents=True, exist_ok=True)

    for i, chart in enumerate(charts):
        out = pipeline.run_pipeline(chart.bgr, params)
        pred = corner_bboxes_to_xywh(out["corner_bboxes"])
        m = evaluate_chart(chart.bars, pred)
        chart_metrics.append(m)

        if args.save_overlays and i < args.overlay_count:
            overlay = _draw_overlay(chart.bgr, chart.bars, pred)
            path = EVAL_OUTPUTS / f"overlay_{args.seed + i:04d}.png"
            cv2.imwrite(str(path), overlay)

    dataset = evaluate_dataset(chart_metrics)
    print(
        f"Charts: {dataset.n_charts}  |  "
        f"P={dataset.mean_precision:.3f}  R={dataset.mean_recall:.3f}  "
        f"F1={dataset.mean_f1:.3f}  mean_IoU={dataset.mean_iou:.3f}"
    )
    print(
        f"Charts with F1 >= 0.5: {dataset.charts_f1_above_half}/{dataset.n_charts}"
    )
    if args.save_overlays:
        print(f"Overlays saved to {EVAL_OUTPUTS}/")


if __name__ == "__main__":
    main()
