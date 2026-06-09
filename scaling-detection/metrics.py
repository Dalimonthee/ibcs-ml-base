"""IoU overlap metrics for bar bounding-box evaluation."""

from __future__ import annotations

from dataclasses import dataclass


Box = tuple[int, int, int, int]


def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-union for axis-aligned boxes ``(x, y, w, h)``."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_boxes(
    gt: list[Box],
    pred: list[Box],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching by descending IoU.

    Returns ``(matches, unmatched_gt_indices, unmatched_pred_indices)`` where
    each match is ``(gt_index, pred_index, iou)``.
    """
    if not gt or not pred:
        return [], list(range(len(gt))), list(range(len(pred)))

    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = box_iou(g, p)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


@dataclass
class ChartMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    n_gt: int
    n_pred: int


def evaluate_chart(
    gt: list[Box],
    pred: list[Box],
    iou_threshold: float = 0.5,
) -> ChartMetrics:
    """Precision / recall / F1 and mean IoU on matched pairs."""
    matches, unmatched_gt, unmatched_pred = match_boxes(gt, pred, iou_threshold)
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(m[2] for m in matches) / tp if tp > 0 else 0.0
    return ChartMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        n_gt=len(gt),
        n_pred=len(pred),
    )


@dataclass
class DatasetMetrics:
    n_charts: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_iou: float
    charts_f1_above_half: int


def evaluate_dataset(
    chart_results: list[ChartMetrics],
) -> DatasetMetrics:
    """Macro-average per-chart metrics."""
    n = len(chart_results)
    if n == 0:
        return DatasetMetrics(0, 0.0, 0.0, 0.0, 0.0, 0)
    return DatasetMetrics(
        n_charts=n,
        mean_precision=sum(r.precision for r in chart_results) / n,
        mean_recall=sum(r.recall for r in chart_results) / n,
        mean_f1=sum(r.f1 for r in chart_results) / n,
        mean_iou=sum(r.mean_iou for r in chart_results) / n,
        charts_f1_above_half=sum(1 for r in chart_results if r.f1 >= 0.5),
    )


def corner_bboxes_to_xywh(
    corner_bboxes: list[tuple[int, int, int, int, str]],
) -> list[Box]:
    """Strip the diagnostic source field from pipeline corner bboxes."""
    return [(x, y, w, h) for x, y, w, h, _ in corner_bboxes]
