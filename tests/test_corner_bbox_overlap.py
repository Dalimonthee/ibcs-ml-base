"""Overlap evaluation for orange corner-derived bounding boxes on synthetic B&W charts."""

from __future__ import annotations

import pipeline
from metrics import corner_bboxes_to_xywh, evaluate_chart, evaluate_dataset
from synthetic_bw import generate_batch

N_CHARTS = 80
BASE_SEED = 42


def _run_eval():
    params = pipeline.eval_params()
    chart_metrics = []
    for chart in generate_batch(BASE_SEED, N_CHARTS):
        out = pipeline.run_pipeline(chart.bgr, params)
        pred = corner_bboxes_to_xywh(out["corner_bboxes"])
        chart_metrics.append(evaluate_chart(chart.bars, pred))
    return evaluate_dataset(chart_metrics), chart_metrics


def test_corner_bbox_overlap_batch(capsys):
    """Orange corner bboxes should overlap GT boxes for black bars only (white = decoy)."""
    dataset, _per_chart = _run_eval()

    print(
        f"\nOverlap eval ({N_CHARTS} charts, eval_params): "
        f"P={dataset.mean_precision:.3f} R={dataset.mean_recall:.3f} "
        f"F1={dataset.mean_f1:.3f} mean_IoU={dataset.mean_iou:.3f} "
        f"charts_F1>=0.5: {dataset.charts_f1_above_half}/{dataset.n_charts}"
    )

    # Calibrated on 80-chart batch; GT = black bars only, white bars are decoys.
    assert dataset.mean_f1 >= 0.50, (
        f"mean F1 {dataset.mean_f1:.3f} below 0.50 "
        f"(P={dataset.mean_precision:.3f}, R={dataset.mean_recall:.3f})"
    )
    assert dataset.mean_recall >= 0.45, (
        f"mean recall {dataset.mean_recall:.3f} below 0.45"
    )
    assert dataset.charts_f1_above_half >= dataset.n_charts // 2, (
        f"only {dataset.charts_f1_above_half}/{dataset.n_charts} charts have F1 >= 0.5"
    )
