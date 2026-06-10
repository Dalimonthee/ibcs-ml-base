"""IBCS bar-detection pipeline (OpenCV, no UI).

Imported by the Streamlit app and by overlap-evaluation tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "Dataset"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

MASK_NAMES = ["A", "B", "C", "filled", "combined"]
FLOOR_SOURCES = ["winner mask", "Mask A", "Mask B", "Mask C", "Mask filled", "Combined"]
MASK_CAPTIONS = {
    "A": "Mask A - black",
    "B": "Mask B - grey",
    "C": "Mask C - hatched",
    "filled": "Mask filled (outlines)",
    "combined": "Combined (fallback)",
}


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class Params:
    invert_input: bool
    black_max: int
    grey_low: int
    grey_high: int
    canny_low: int
    canny_high: int
    hatch_close_iters: int
    noise_keep_frac: float
    proj_threshold_frac: float
    min_area_floor: int
    min_area_fraction: float
    min_fill_ratio: float
    min_bar_w: int
    min_bar_h: int
    max_aspect: float
    corner_enabled: bool
    corner_source: str
    corner_max: int
    corner_quality: float
    corner_min_dist: int
    oriented_enabled: bool
    oriented_source: str
    oriented_kernel_size: int
    oriented_threshold_frac: float
    oriented_min_dist: int
    bbox_from_corners_enabled: bool
    bbox_axis_types: tuple[str, ...]
    bbox_axis_y_tolerance: int
    bbox_axis_min_count: int
    bbox_axis_max_count: int
    bbox_pair_y_tolerance: int
    bbox_min_width: int
    bbox_max_width_frac: float
    bbox_min_height: int
    bbox_max_height_frac: float
    axis_use_corners: bool
    axis_use_morph: bool
    axis_use_hough: bool
    axis_corner_weight: float
    axis_morph_weight: float
    axis_hough_weight: float
    axis_min_total_support: float
    axis_morph_kernel_w_frac: float
    axis_morph_abs_floor: int
    axis_morph_rel_floor: float
    axis_hough_min_len_frac: float
    axis_hough_threshold: int
    axis_hough_max_gap: int
    axis_hough_angle_tol_deg: float
    floor_enabled: bool
    floor_source: str
    floor_active_frac: float


ORIENTED_SOURCES = [
    "grayscale",
    "Mask A",
    "Mask B",
    "Mask C",
    "Mask filled",
    "Combined",
    "winner mask",
]


ORIENTED_TYPES = ["TL_convex", "TR_convex", "TL_concave", "TR_concave"]
ORIENTED_COLORS_BGR = {
    "TL_convex":  (0, 220, 0),     # green
    "TR_convex":  (220, 200, 0),   # cyan
    "TL_concave": (0, 220, 220),   # yellow
    "TR_concave": (220, 30, 220),  # magenta
}


# ---------------------------------------------------------------------------
# Projection-profile bar finder (per mask)
# ---------------------------------------------------------------------------


def _find_runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs for runs of True in a 1D bool array."""
    if active.size == 0:
        return []
    padded = np.concatenate(([False], active, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def projection_bars(mask: np.ndarray, p: Params) -> list[tuple[int, int, int, int, float]]:
    """Find bars in a single mask via X-then-Y projection profiles.

    Returns a list of ``(x, y, w, h, fill_ratio)`` tuples.
    """
    if mask.size == 0:
        return []

    col_sums = mask.sum(axis=0).astype(np.float64)
    if col_sums.max() < 1:
        return []
    x_thresh = p.proj_threshold_frac * col_sums.max()
    x_runs = _find_runs(col_sums > x_thresh)

    boxes: list[tuple[int, int, int, int, float]] = []
    for x0, x1 in x_runs:
        bw = x1 - x0 + 1
        if bw < p.min_bar_w:
            continue

        slice_ = mask[:, x0 : x1 + 1]
        row_sums = slice_.sum(axis=1).astype(np.float64)
        if row_sums.max() < 1:
            continue
        y_thresh = p.proj_threshold_frac * row_sums.max()
        y_runs = _find_runs(row_sums > y_thresh)

        for y0, y1 in y_runs:
            bh = y1 - y0 + 1
            if bh < p.min_bar_h:
                continue
            if max(bw / bh, bh / bw) > p.max_aspect:
                continue
            sub = mask[y0 : y1 + 1, x0 : x1 + 1]
            filled = int((sub > 0).sum())
            area = bw * bh
            if area <= 0:
                continue
            fill_ratio = filled / area
            if fill_ratio < p.min_fill_ratio:
                continue
            if filled < p.min_area_floor:
                continue
            boxes.append((int(x0), int(y0), int(bw), int(bh), float(fill_ratio)))

    if not boxes:
        return []

    # Dynamic relative-size filter: drop boxes whose filled-pixel area is
    # tiny relative to the biggest box from this same mask.
    filled_areas = [int(round(b[4] * b[2] * b[3])) for b in boxes]
    max_filled = max(filled_areas)
    dyn = max(p.min_area_floor, p.min_area_fraction * max_filled)
    return [b for b, a in zip(boxes, filled_areas) if a >= dyn]


def score_mask(boxes: list[tuple[int, int, int, int, float]], img_shape: tuple[int, int]) -> float:
    """Score a per-mask detection result.

    Higher = more, well-formed rectangles. Heavy penalty if a single box
    covers a huge fraction of the image (the "merged into one giant blob"
    pathology that connected-components OR'd over all masks tends to hit).
    """
    if not boxes:
        return 0.0
    h, w = img_shape
    img_area = h * w
    biggest_box_area = max(bw * bh for _, _, bw, bh, _ in boxes)
    if biggest_box_area > 0.35 * img_area:
        return 0.0
    avg_fill = sum(f for _, _, _, _, f in boxes) / len(boxes)
    return len(boxes) * avg_fill


# ---------------------------------------------------------------------------
# Noise filter + corner detection
# ---------------------------------------------------------------------------


def filter_by_largest(mask: np.ndarray, keep_frac: float) -> np.ndarray:
    """Drop connected components smaller than ``keep_frac`` * the largest blob.

    ``keep_frac`` of 0.0 returns the mask unchanged. 1.0 keeps only the very
    largest blob.
    """
    if keep_frac <= 0.0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    threshold = keep_frac * float(areas.max())
    keep_ids = np.flatnonzero(areas >= threshold) + 1
    if keep_ids.size == 0:
        return np.zeros_like(mask)
    return np.where(np.isin(labels, keep_ids), 255, 0).astype(np.uint8)


def detect_corners(image_gray: np.ndarray, p: Params) -> np.ndarray:
    """Run Shi-Tomasi corner detection. Returns Nx2 int array of (x, y)."""
    if not p.corner_enabled or p.corner_max <= 0:
        return np.empty((0, 2), dtype=np.int32)
    pts = cv2.goodFeaturesToTrack(
        image_gray,
        maxCorners=int(p.corner_max),
        qualityLevel=float(p.corner_quality),
        minDistance=int(p.corner_min_dist),
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.int32)
    return pts.reshape(-1, 2).round().astype(np.int32)


# ---------------------------------------------------------------------------
# Oriented corner kernels (Option B: 4-quadrant signed convolution)
# ---------------------------------------------------------------------------
#
# Each kernel encodes a specific corner geometry as a sign pattern across
# four quadrants of an NxN window. We compare a binary mask (1 = bar,
# 0 = background) against the pattern by setting:
#
#   weight = +n_neg / n_total   for "expect bar"        quadrants
#   weight = -n_pos / n_total   for "expect background" quadrants
#
# That choice makes the kernel zero-sum: a uniform window (all bar or all
# background) gives zero response, so only sharp local geometry contributes.
# Higher response = stronger match to the target geometry.
#
#   TL_convex   target: BR=1, others=0     (convex top-left  of a bar)
#   TR_convex   target: BL=1, others=0     (convex top-right of a bar)
#   TL_concave  target: TL+BL+BR=1, TR=0   (concave corner, notch toward TR)
#   TR_concave  target: TR+BL+BR=1, TL=0   (concave corner, notch toward TL)


def make_corner_kernels(size: int) -> dict[str, np.ndarray]:
    """Build the four oriented corner kernels as zero-sum float32 arrays."""
    size = max(4, int(size))
    if size % 2:
        size += 1
    half = size // 2
    targets = {
        "TL_convex":  np.array([[0, 0], [0, 1]], dtype=np.int8),
        "TR_convex":  np.array([[0, 0], [1, 0]], dtype=np.int8),
        "TL_concave": np.array([[1, 0], [1, 1]], dtype=np.int8),
        "TR_concave": np.array([[0, 1], [1, 1]], dtype=np.int8),
    }
    kernels: dict[str, np.ndarray] = {}
    for name, t in targets.items():
        n_pos = int(t.sum())
        n_neg = 4 - n_pos
        pos_w = n_neg
        neg_w = -n_pos
        k = np.zeros((size, size), dtype=np.float32)
        for qy in range(2):
            for qx in range(2):
                y0, y1 = qy * half, qy * half + half
                x0, x1 = qx * half, qx * half + half
                k[y0:y1, x0:x1] = pos_w if t[qy, qx] else neg_w
        kernels[name] = k
    return kernels


def _peaks_from_response(
    resp: np.ndarray,
    abs_threshold: float,
    min_dist: int,
) -> np.ndarray:
    """Extract peak coordinates from a 2D response map.

    Pipeline:
      1. Build a binary "above threshold" mask using ``abs_threshold``.
      2. Find connected components of that mask -- each blob is one corner
         candidate (this kills the plateau-along-edges pathology that a naive
         dilate-equal-self NMS suffers from).
      3. For each blob, keep its argmax pixel.
      4. Greedy NMS: sort by response descending, accept each point if it is
         at least ``min_dist`` away from every already-accepted point.
    """
    if abs_threshold <= 0.0 or resp.max() < abs_threshold:
        return np.empty((0, 2), dtype=np.int32)

    above = (resp >= abs_threshold).astype(np.uint8)
    num, labels, _, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    if num <= 1:
        return np.empty((0, 2), dtype=np.int32)

    candidates: list[tuple[float, int, int]] = []
    for blob_id in range(1, num):
        ys, xs = np.where(labels == blob_id)
        if ys.size == 0:
            continue
        vals = resp[ys, xs]
        idx = int(vals.argmax())
        candidates.append((float(vals[idx]), int(xs[idx]), int(ys[idx])))

    candidates.sort(key=lambda c: c[0], reverse=True)

    accepted: list[tuple[int, int]] = []
    md2 = max(1, int(min_dist)) ** 2
    for _, x, y in candidates:
        if all((x - ax) ** 2 + (y - ay) ** 2 >= md2 for ax, ay in accepted):
            accepted.append((x, y))

    if not accepted:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(accepted, dtype=np.int32)


def detect_oriented_corners(
    source: np.ndarray,
    kernels: dict[str, np.ndarray],
    threshold_frac: float,
    min_dist: int,
) -> dict[str, np.ndarray]:
    """Apply each corner kernel; return ``{type: Nx2 int array of (x, y)}``.

    The input is normalized to ``[0, 1]`` (uint8 ``0..255`` -> float32 ``0..1``)
    so that binary masks and grayscale images share the same response scale.
    ``threshold_frac`` is then interpreted as a fraction of each kernel's
    *theoretical* maximum response (the sum of its positive weights, i.e. what
    a perfect geometric match scores on a normalized input). This keeps the
    threshold meaningful even when a given image contains no instance of one
    of the corner types.
    """
    if source.size == 0:
        return {name: np.empty((0, 2), dtype=np.int32) for name in kernels}
    if source.dtype == np.uint8:
        src_f = source.astype(np.float32) / 255.0
    else:
        src_f = source.astype(np.float32)
        if src_f.max() > 1.5:
            src_f /= 255.0

    out: dict[str, np.ndarray] = {}
    for name, k in kernels.items():
        resp = cv2.filter2D(src_f, ddepth=cv2.CV_32F, kernel=k)
        theoretical_max = float(k[k > 0].sum())
        abs_thresh = threshold_frac * theoretical_max
        out[name] = _peaks_from_response(resp, abs_thresh, min_dist)
    return out


def pick_corner_source(
    name: str,
    gray: np.ndarray,
    per_mask: dict[str, dict],
    winner: str,
) -> np.ndarray:
    """Return the uint8 image the corner kernels should run on.

    The kernels assume "bar pixels = HIGH values". For binary masks
    (Mask A/B/C/filled/Combined/winner) that is already the case. For grayscale
    we auto-invert when the background appears light (typical IBCS chart) so
    bars become the high-value pixels.
    """
    if name == "grayscale":
        if gray.mean() > 127:
            return cv2.bitwise_not(gray)
        return gray.copy()
    if name == "winner mask":
        return per_mask[winner]["mask"]
    mask_lookup = {
        "Mask A": "A",
        "Mask B": "B",
        "Mask C": "C",
        "Mask filled": "filled",
        "Combined": "combined",
    }
    key = mask_lookup.get(name)
    if key is None or key not in per_mask:
        return per_mask[winner]["mask"]
    return per_mask[key]["mask"]


# ---------------------------------------------------------------------------
# Bounding boxes from corners
# ---------------------------------------------------------------------------


def find_axes_by_corners(
    points: np.ndarray,
    img_h: int,
    y_tolerance: int,
    min_count: int,
) -> list[tuple[int, int]]:
    """Locate horizontal axes by 1-D clustering corner Y coordinates.

    Sort the Ys; greedily start a new cluster whenever the gap to the previous
    Y exceeds ``y_tolerance``. A cluster with at least ``min_count`` members
    becomes an axis candidate centered on the mean of its Ys.

    Returns ``[(y, support_count)]`` sorted top-to-bottom.
    """
    if points.size == 0 or img_h <= 0 or min_count <= 0:
        return []
    ys = np.sort(points[:, 1].astype(np.int32)).tolist()
    clusters: list[list[int]] = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] <= int(y_tolerance):
            clusters[-1].append(int(y))
        else:
            clusters.append([int(y)])

    peaks = [
        (int(round(sum(c) / len(c))), len(c))
        for c in clusters
        if len(c) >= int(min_count)
    ]
    peaks.sort(key=lambda p: p[0])
    return peaks


def find_axes_by_morph(
    edges: np.ndarray,
    kernel_w: int,
    abs_score_floor: int,
    rel_score_floor: float,
) -> list[tuple[int, int]]:
    """Find horizontal strokes via morphological opening + row-sum peaks.

    Applies ``MORPH_OPEN`` with a 1x``kernel_w`` horizontal kernel. Only pixels
    that belong to a horizontal run at least ``kernel_w`` long survive.
    Row sums of the result peak exactly on horizontal strokes (axis lines,
    chart frame edges, gridlines).

    A row is treated as a peak when its sum exceeds both ``abs_score_floor``
    pixels and ``rel_score_floor * max_row_sum``. Contiguous runs of qualifying
    rows are collapsed to their argmax. Returns ``[(y, row_sum)]``.
    """
    if edges.size == 0 or kernel_w < 4:
        return []
    h = edges.shape[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel_w), 1))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    row_sum = horiz.sum(axis=1).astype(np.int32)
    peak_val = int(row_sum.max())
    if peak_val <= 0:
        return []
    threshold = max(int(abs_score_floor), int(rel_score_floor * peak_val))
    above = row_sum >= threshold

    peaks: list[tuple[int, int]] = []
    i = 0
    while i < h:
        if above[i]:
            j = i
            while j < h and above[j]:
                j += 1
            band = row_sum[i:j]
            offset = int(np.argmax(band))
            peaks.append((int(i + offset), int(band[offset])))
            i = j
        else:
            i += 1
    return peaks


def find_axes_by_hough(
    edges: np.ndarray,
    min_line_length: int,
    threshold: int,
    max_line_gap: int,
    angle_tol_deg: float,
    y_tolerance: int,
) -> list[tuple[int, int]]:
    """Find horizontal axes via probabilistic Hough transform.

    Runs ``cv2.HoughLinesP`` and keeps segments whose endpoints differ by less
    than ``angle_tol_deg`` from horizontal. Their midpoint Ys are clustered by
    ``y_tolerance``; cluster centers are returned with their member count.
    """
    if edges.size == 0 or min_line_length < 4:
        return []
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )
    if lines is None:
        return []
    angle_tol = float(angle_tol_deg)
    ys: list[int] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = abs(int(x2) - int(x1))
        dy = abs(int(y2) - int(y1))
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle <= angle_tol:
            ys.append((int(y1) + int(y2)) // 2)
    if not ys:
        return []
    ys.sort()
    clusters: list[list[int]] = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] <= int(y_tolerance):
            clusters[-1].append(int(y))
        else:
            clusters.append([int(y)])
    return [(int(round(sum(c) / len(c))), len(c)) for c in clusters]


def fuse_axes(
    img_h: int,
    y_tolerance: int,
    min_total_support: float,
    max_axes: int,
    sources: list[tuple[str, list[tuple[int, int]], float]],
) -> list[tuple[int, float, str]]:
    """Combine axis candidates from multiple detectors into a single ranking.

    Each ``sources`` entry is ``(name, peaks, weight)`` where ``peaks`` is
    ``[(y, support)]``. Within a source the support values are normalized to
    their own max so that detectors on wildly different scales (corner counts
    vs. row sums vs. line counts) contribute comparably.

    Peaks within ``y_tolerance`` across sources are merged via weighted mean.
    The fused score equals the sum of contributing weights. Returns
    ``[(y, fused_score, "src1+src2+...")]`` sorted top-to-bottom, capped to
    ``max_axes`` strongest peaks and to ``min_total_support`` floor.
    """
    candidates: list[tuple[int, float, str]] = []
    for name, peaks, weight in sources:
        if not peaks or weight <= 0:
            continue
        max_sup = max(s for _, s in peaks) or 1
        for y, sup in peaks:
            candidates.append((int(y), float(weight) * float(sup) / float(max_sup), name))
    if not candidates:
        return []
    candidates.sort(key=lambda c: c[0])

    clusters: list[list[tuple[int, float, str]]] = [[candidates[0]]]
    for y, w, src in candidates[1:]:
        if y - clusters[-1][-1][0] <= int(y_tolerance):
            clusters[-1].append((y, w, src))
        else:
            clusters.append([(y, w, src)])

    fused: list[tuple[int, float, str]] = []
    for cl in clusters:
        total_w = sum(w for _, w, _ in cl)
        if total_w < float(min_total_support):
            continue
        y_avg = sum(y * w for y, w, _ in cl) / total_w
        srcs = sorted({s for _, _, s in cl})
        fused.append((int(round(y_avg)), float(total_w), "+".join(srcs)))

    fused.sort(key=lambda r: r[1], reverse=True)
    fused = fused[: max(1, int(max_axes))]
    fused.sort(key=lambda r: r[0])
    return fused


def infer_bar_floor(
    mask: np.ndarray,
    x1: int,
    x2: int,
    y_top: int,
    min_active_frac: float,
    max_search: int,
    initial_slack: int = 3,
) -> int | None:
    """Walk down a column strip from ``y_top`` to find the bar's bottom.

    For each row, compute the fraction of active pixels in the column range
    ``[x1, x2)``. Returns the last Y where this fraction stays above
    ``min_active_frac``, allowing a few initial rows of slack so antialiased
    corner pixels do not abort the walk before the bar fill is reached.
    Returns ``None`` if no usable bar fill is found within ``max_search`` px.
    """
    h, w = mask.shape[:2]
    x1 = max(0, int(x1))
    x2 = min(w, int(x2))
    y_top = max(0, int(y_top))
    width = x2 - x1
    if width <= 0 or y_top >= h:
        return None
    threshold = max(1, int(float(min_active_frac) * width))
    end_y = min(h, y_top + max(int(max_search), 1))
    strip = (mask[y_top:end_y, x1:x2] > 0).sum(axis=1)

    start = 0
    for i, c in enumerate(strip[: max(0, int(initial_slack))]):
        if c >= threshold:
            start = i
            break
    last_active = -1
    for i in range(start, len(strip)):
        if strip[i] >= threshold:
            last_active = i
        else:
            if last_active >= 0:
                break
    if last_active < 0:
        return None
    return y_top + int(last_active)


def build_bboxes_from_corners(
    tl_pts: np.ndarray,
    tr_pts: np.ndarray,
    axes_y: list[int],
    img_w: int,
    img_h: int,
    pair_y_tolerance: int,
    min_width: int,
    max_width_frac: float,
    min_height: int,
    max_height_frac: float,
    floor_mask: np.ndarray | None = None,
    use_bar_floor: bool = False,
    floor_active_frac: float = 0.5,
) -> list[tuple[int, int, int, int, str]]:
    """Pair TL_convex + TR_convex corners and close them off into a bbox.

    Closing strategy per bar (in order of preference):
    1. ``infer_bar_floor`` on ``floor_mask`` when ``use_bar_floor`` is set --
       walks the column strip down from the pair's top until the bar fill
       drops below ``floor_active_frac``.
    2. The nearest axis strictly below the pair's top in ``axes_y``.

    Bars whose implied height exceeds ``max_height_frac * img_h`` are rejected
    (avoids pairing chart titles with the bottom axis). The last tuple field
    records which strategy actually closed the box, for diagnostics.
    """
    if tl_pts.size == 0 or tr_pts.size == 0:
        return []

    max_width = max(int(min_width) + 1, int(max_width_frac * img_w))
    max_height = max(int(min_height) + 1, int(max_height_frac * img_h))
    axes_sorted = sorted(int(a) for a in axes_y)
    have_floor_mask = use_bar_floor and floor_mask is not None and floor_mask.size > 0

    order = tr_pts[:, 0].argsort()
    tr_sorted = tr_pts[order]
    tr_used = np.zeros(len(tr_sorted), dtype=bool)
    tr_xs = tr_sorted[:, 0].astype(np.int32)

    bboxes: list[tuple[int, int, int, int, str]] = []
    for tl in tl_pts:
        tl_x, tl_y = int(tl[0]), int(tl[1])
        lo_x = tl_x + int(min_width)
        hi_x = tl_x + max_width
        lo_i = int(np.searchsorted(tr_xs, lo_x, side="left"))
        hi_i = int(np.searchsorted(tr_xs, hi_x, side="right"))

        best: tuple[float, int, int, int] | None = None
        for i in range(lo_i, hi_i):
            if tr_used[i]:
                continue
            tr_x = int(tr_sorted[i, 0])
            tr_y = int(tr_sorted[i, 1])
            dy = abs(tr_y - tl_y)
            if dy > int(pair_y_tolerance):
                continue
            score = (tr_x - tl_x) + dy * 3
            if best is None or score < best[0]:
                best = (score, i, tr_x, tr_y)
        if best is None:
            continue
        _, tr_i, tr_x, tr_y = best
        tr_used[tr_i] = True

        y_top = (tl_y + tr_y) // 2
        nearest_axis_below = next(
            (a for a in axes_sorted if a > y_top + int(min_height)), None
        )
        y_bot: int | None = None
        source = ""

        if have_floor_mask:
            assert floor_mask is not None
            search_cap = int(max_height)
            if nearest_axis_below is not None:
                search_cap = min(search_cap, nearest_axis_below - y_top + 1)
            search_cap = max(int(min_height) + 1, search_cap)
            y_bot_floor = infer_bar_floor(
                floor_mask,
                tl_x,
                tr_x,
                y_top,
                min_active_frac=float(floor_active_frac),
                max_search=search_cap,
            )
            if y_bot_floor is not None and y_bot_floor - y_top >= int(min_height):
                y_bot = y_bot_floor
                source = "floor"

        if y_bot is None and nearest_axis_below is not None:
            y_bot = nearest_axis_below
            source = "axis"

        if y_bot is None:
            continue

        x, w, h = tl_x, tr_x - tl_x, y_bot - y_top
        if w < int(min_width) or h < int(min_height) or h > max_height:
            continue
        bboxes.append((x, y_top, w, h, source))

    return bboxes


def align_corner_bboxes_to_lowest_bottom(
    bboxes: list[tuple[int, int, int, int, str]],
    axes_y: list[int],
    img_h: int,
    *,
    plot_top_frac: float = 0.12,
) -> list[tuple[int, int, int, int, str]]:
    """Extend each corner bbox downward to the chart's lowest shared bottom edge.

    Bars in a column chart share one baseline. After per-bar closing (axis or
    floor walk), bottoms often stop too high. Take the lowest bottom among bar
    boxes and detected x-axes, then pull every in-plot box down to that Y.
    """
    if not bboxes:
        return bboxes

    plot_y_min = int(plot_top_frac * img_h)
    plot_boxes = [(x, y, w, h, s) for x, y, w, h, s in bboxes if y >= plot_y_min]
    if not plot_boxes:
        return bboxes

    candidates = [y + h for _, y, _, h, _ in plot_boxes]
    min_bar_top = min(y for _, y, _, _, _ in plot_boxes)
    if axes_y:
        candidates.extend(int(a) for a in axes_y if int(a) > min_bar_top)

    y_floor = min(max(candidates), img_h - 1)

    aligned: list[tuple[int, int, int, int, str]] = []
    for x, y, w, h, src in bboxes:
        if y < plot_y_min:
            aligned.append((x, y, w, h, src))
            continue
        new_h = y_floor - y
        if new_h <= h:
            aligned.append((x, y, w, h, src))
            continue
        tag = f"{src}+align" if src else "align"
        aligned.append((x, y, w, new_h, tag))
    return aligned


# ---------------------------------------------------------------------------
# Mask building + full pipeline
# ---------------------------------------------------------------------------


def _build_masks(gray: np.ndarray, p: Params) -> dict[str, np.ndarray]:
    _, mask_a = cv2.threshold(gray, p.black_max, 255, cv2.THRESH_BINARY_INV)
    mask_b = cv2.inRange(gray, p.grey_low, p.grey_high)

    edges = cv2.Canny(gray, p.canny_low, p.canny_high)
    d1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    d2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    hatch = cv2.bitwise_or(
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, d1, iterations=3),
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, d2, iterations=3),
    )
    if p.hatch_close_iters > 0:
        mask_c = cv2.morphologyEx(
            hatch, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=p.hatch_close_iters,
        )
    else:
        mask_c = hatch

    closed_edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < p.min_area_floor:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if not (4 <= len(approx) <= 8):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if area / max(w * h, 1) < p.min_fill_ratio:
            continue
        kept.append(c)
    mask_filled = np.zeros_like(gray)
    cv2.drawContours(mask_filled, kept, -1, 255, thickness=-1)

    combined = cv2.bitwise_or(
        cv2.bitwise_or(mask_a, mask_b),
        cv2.bitwise_or(mask_c, mask_filled),
    )

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned: dict[str, np.ndarray] = {}
    for name, m in {
        "A": mask_a, "B": mask_b, "C": mask_c,
        "filled": mask_filled, "combined": combined,
    }.items():
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3, iterations=1)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k3, iterations=1)
        m2 = filter_by_largest(m2, p.noise_keep_frac)
        cleaned[name] = m2
    return cleaned


def run_pipeline(bgr: np.ndarray, p: Params) -> dict:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if p.invert_input:
        gray = cv2.bitwise_not(gray)

    masks = _build_masks(gray, p)

    per_mask: dict[str, dict] = {}
    for name in MASK_NAMES:
        m = masks[name]
        boxes = projection_bars(m, p)
        per_mask[name] = {
            "mask": m,
            "boxes": boxes,
            "score": score_mask(boxes, m.shape),
        }

    # Winner = highest score. Primary masks (A, B, C, filled) outrank the
    # combined fallback at equal score. Within primary masks the tiebreak
    # prefers earlier in the priority list.
    def rank_key(name: str) -> tuple[float, int, int]:
        is_fallback = 1 if name == "combined" else 0
        return (per_mask[name]["score"], -is_fallback, -MASK_NAMES.index(name))

    winner = max(MASK_NAMES, key=rank_key)
    if per_mask[winner]["score"] <= 0.0:
        winner = "combined" if per_mask["combined"]["score"] > 0 else winner

    winner_boxes = per_mask[winner]["boxes"]
    winner_mask = per_mask[winner]["mask"]

    corner_source_img = gray if p.corner_source == "grayscale" else winner_mask
    corners = detect_corners(corner_source_img, p)

    if p.oriented_enabled:
        kernels = make_corner_kernels(p.oriented_kernel_size)
        oriented_source_img = pick_corner_source(p.oriented_source, gray, per_mask, winner)
        oriented_corners = detect_oriented_corners(
            oriented_source_img, kernels, p.oriented_threshold_frac, p.oriented_min_dist,
        )
    else:
        oriented_source_img = winner_mask
        oriented_corners = {name: np.empty((0, 2), dtype=np.int32) for name in ORIENTED_TYPES}

    img_h, img_w = gray.shape[:2]
    edges_axis = cv2.Canny(gray, p.canny_low, p.canny_high)

    if p.bbox_from_corners_enabled and p.oriented_enabled:
        axis_points = np.concatenate(
            [oriented_corners[t] for t in p.bbox_axis_types if t in oriented_corners and oriented_corners[t].size],
            axis=0,
        ) if any(oriented_corners[t].size for t in p.bbox_axis_types) else np.empty((0, 2), dtype=np.int32)

        corner_peaks = (
            find_axes_by_corners(axis_points, img_h, p.bbox_axis_y_tolerance, p.bbox_axis_min_count)
            if p.axis_use_corners else []
        )
        morph_kernel_w = max(8, int(p.axis_morph_kernel_w_frac * img_w))
        morph_peaks = (
            find_axes_by_morph(
                edges_axis,
                kernel_w=morph_kernel_w,
                abs_score_floor=p.axis_morph_abs_floor,
                rel_score_floor=p.axis_morph_rel_floor,
            )
            if p.axis_use_morph else []
        )
        hough_min_len = max(8, int(p.axis_hough_min_len_frac * img_w))
        hough_peaks = (
            find_axes_by_hough(
                edges_axis,
                min_line_length=hough_min_len,
                threshold=int(p.axis_hough_threshold),
                max_line_gap=int(p.axis_hough_max_gap),
                angle_tol_deg=float(p.axis_hough_angle_tol_deg),
                y_tolerance=p.bbox_axis_y_tolerance,
            )
            if p.axis_use_hough else []
        )
        axes_peaks = fuse_axes(
            img_h=img_h,
            y_tolerance=p.bbox_axis_y_tolerance,
            min_total_support=p.axis_min_total_support,
            max_axes=p.bbox_axis_max_count,
            sources=[
                ("corner", corner_peaks, p.axis_corner_weight),
                ("morph", morph_peaks, p.axis_morph_weight),
                ("hough", hough_peaks, p.axis_hough_weight),
            ],
        )
        axes_y = [y for y, _, _ in axes_peaks]

        if p.floor_enabled:
            floor_mask = pick_corner_source(p.floor_source, gray, per_mask, winner)
        else:
            floor_mask = None
        corner_bboxes = build_bboxes_from_corners(
            oriented_corners["TL_convex"],
            oriented_corners["TR_convex"],
            axes_y,
            img_w,
            img_h,
            p.bbox_pair_y_tolerance,
            p.bbox_min_width,
            p.bbox_max_width_frac,
            p.bbox_min_height,
            p.bbox_max_height_frac,
            floor_mask=floor_mask,
            use_bar_floor=p.floor_enabled,
            floor_active_frac=p.floor_active_frac,
        )
        corner_bboxes = align_corner_bboxes_to_lowest_bottom(
            corner_bboxes, axes_y, img_h,
        )
    else:
        corner_peaks = []
        morph_peaks = []
        hough_peaks = []
        axes_peaks = []
        corner_bboxes = []

    overlay = rgb.copy()
    for x, y, w, h, _ in winner_boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 0), 2)
    for cx, cy in corners:
        cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 255, 255), thickness=-1)
        cv2.circle(overlay, (int(cx), int(cy)), 3, (220, 30, 30), thickness=-1)
    for name in ORIENTED_TYPES:
        color = ORIENTED_COLORS_BGR[name]
        for cx, cy in oriented_corners[name]:
            cv2.circle(overlay, (int(cx), int(cy)), 6, (255, 255, 255), thickness=1)
            cv2.circle(overlay, (int(cx), int(cy)), 4, color, thickness=-1)
    for ay, _support, _src in axes_peaks:
        cv2.line(overlay, (0, ay), (img_w - 1, ay), (255, 80, 220), 1, cv2.LINE_AA)
    for x, y, w, h, _src in corner_bboxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 140, 0), 2)

    return {
        "rgb": rgb,
        "gray": gray,
        "per_mask": per_mask,
        "winner": winner,
        "winner_boxes": winner_boxes,
        "corners": corners,
        "oriented_corners": oriented_corners,
        "oriented_source_img": oriented_source_img,
        "axes_peaks": axes_peaks,
        "axis_signals": {"corner": corner_peaks, "morph": morph_peaks, "hough": hough_peaks},
        "corner_bboxes": corner_bboxes,
        "overlay": overlay,
    }


def app_default_params() -> Params:
    """Streamlit app sidebar defaults (scaling-detection/app.py)."""
    return Params(
        invert_input=False,
        black_max=60,
        grey_low=80,
        grey_high=190,
        canny_low=50,
        canny_high=150,
        hatch_close_iters=2,
        noise_keep_frac=0.0,
        proj_threshold_frac=0.10,
        min_area_floor=100,
        min_area_fraction=0.05,
        min_fill_ratio=0.55,
        min_bar_w=4,
        min_bar_h=4,
        max_aspect=25.0,
        corner_enabled=False,
        corner_source="winner mask",
        corner_max=300,
        corner_quality=0.01,
        corner_min_dist=6,
        oriented_enabled=True,
        oriented_source="grayscale",
        oriented_kernel_size=16,
        oriented_threshold_frac=0.4,
        oriented_min_dist=6,
        bbox_from_corners_enabled=True,
        bbox_axis_types=("TR_convex", "TR_concave", "TL_concave", "TL_convex"),
        bbox_axis_y_tolerance=4,
        bbox_axis_min_count=3,
        bbox_axis_max_count=5,
        bbox_pair_y_tolerance=6,
        bbox_min_width=6,
        bbox_max_width_frac=0.30,
        bbox_min_height=8,
        bbox_max_height_frac=0.60,
        axis_use_corners=True,
        axis_use_morph=True,
        axis_use_hough=True,
        axis_corner_weight=1.5,
        axis_morph_weight=1.5,
        axis_hough_weight=1.5,
        axis_min_total_support=0.3,
        axis_morph_kernel_w_frac=0.15,
        axis_morph_abs_floor=200,
        axis_morph_rel_floor=0.30,
        axis_hough_min_len_frac=0.20,
        axis_hough_threshold=80,
        axis_hough_max_gap=20,
        axis_hough_angle_tol_deg=2.0,
        floor_enabled=True,
        floor_source="winner mask",
        floor_active_frac=0.5,
    )


def default_params() -> Params:
    """Legacy defaults for benchmarks; prefer app_default_params() for CLI."""
    return Params(
        invert_input=False,
        black_max=60,
        grey_low=80,
        grey_high=190,
        canny_low=50,
        canny_high=150,
        hatch_close_iters=2,
        noise_keep_frac=0.0,
        proj_threshold_frac=0.10,
        min_area_floor=100,
        min_area_fraction=0.05,
        min_fill_ratio=0.55,
        min_bar_w=4,
        min_bar_h=4,
        max_aspect=25.0,
        corner_enabled=False,
        corner_source="winner mask",
        corner_max=300,
        corner_quality=0.01,
        corner_min_dist=6,
        oriented_enabled=True,
        oriented_source="Mask A",
        oriented_kernel_size=16,
        oriented_threshold_frac=0.85,
        oriented_min_dist=6,
        bbox_from_corners_enabled=True,
        bbox_axis_types=("TR_convex", "TR_concave", "TL_concave", "TL_convex"),
        bbox_axis_y_tolerance=8,
        bbox_axis_min_count=3,
        bbox_axis_max_count=4,
        bbox_pair_y_tolerance=8,
        bbox_min_width=6,
        bbox_max_width_frac=0.30,
        bbox_min_height=8,
        bbox_max_height_frac=0.60,
        axis_use_corners=True,
        axis_use_morph=True,
        axis_use_hough=True,
        axis_corner_weight=1.5,
        axis_morph_weight=1.0,
        axis_hough_weight=0.8,
        axis_min_total_support=0.5,
        axis_morph_kernel_w_frac=0.15,
        axis_morph_abs_floor=200,
        axis_morph_rel_floor=0.25,
        axis_hough_min_len_frac=0.15,
        axis_hough_threshold=80,
        axis_hough_max_gap=10,
        axis_hough_angle_tol_deg=5.0,
        floor_enabled=True,
        floor_source="winner mask",
        floor_active_frac=0.5,
    )


def eval_params() -> Params:
    """Params for overlap evaluation: grayscale kernels, threshold 0.45."""
    p = default_params()
    p.oriented_source = "grayscale"
    p.oriented_threshold_frac = 0.45
    p.oriented_enabled = True
    p.bbox_from_corners_enabled = True
    # Axis-based closing is more reliable than floor-walk on synthetic B&W charts.
    p.floor_enabled = False
    p.bbox_axis_min_count = 2
    p.oriented_min_dist = 4
    p.axis_min_total_support = 0.35
    return p
