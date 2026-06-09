# OpenCV bar detection pipeline: report

## Summary

The code under `scaling-detection/` is a classical computer vision pipeline, not a trained neural network. It takes a chart image (usually a crop) and tries to find the horizontal baseline, the top corners of each bar, and a bounding box per bar suitable for height measurement. That work lines up with Models 2 and 5 in the [five-model IBCS scaling architecture](./ibcs-scaling-ml-pipeline-report.md). Label reading and zero-axis checks are out of scope here.

The pipeline is tuned for IBCS-style bar charts: black/grey/hatched fills, light backgrounds, shared category baseline. A Streamlit app exposes every parameter for manual tuning on images in `Dataset/`. Automated scores today come from procedurally generated black-and-white charts with known bar boxes.

## What it does

Input is a BGR image. Output is masks, candidate boxes, fused axis lines, corner-derived bar boxes, and a single RGB overlay for review.

### Step 1: Grayscale and masks

The image is converted to grayscale (optional invert for dark themes). Five binary masks are built:

| Mask | Role |
|------|------|
| A | Dark pixels (black bars) via threshold |
| B | Mid-grey bars via intensity range |
| C | Hatched regions from Canny edges and diagonal morph close |
| filled | Filled rectangles from closed edge contours |
| combined | OR of all of the above as fallback |

Each mask gets light morph close/open and optional connected-component filtering to drop speckle.

### Step 2: Projection bars (per mask)

On each mask, column sums find vertical runs of ink; within each run, row sums find bar-shaped blobs. Filters drop slivers, extreme aspect ratios, and low fill ratio. Masks are scored; the winner (preferring A, B, C, filled over combined) drives a set of green boxes on the overlay. This path is fast and works on simple fills but can merge adjacent bars or grab chrome on busy dashboards.

### Step 3: Oriented corners

Four small convolution kernels (TL/TR convex and concave) score local corner geometry on a chosen source (grayscale or a mask). Peaks become corner points. These points support axis finding and bar width, not full bar height by themselves.

### Step 4: Horizontal axis fusion

When corner-based boxes are enabled, three detectors propose horizontal lines:

- Cluster Y coordinates from oriented corner points
- Morphological horizontal open on Canny edges (row-sum peaks)
- Probabilistic Hough on near-horizontal segments

Weighted fusion merges peaks within a few pixels. The result is the category baseline (magenta lines on the overlay). Model 2 in the scaling architecture maps to this block.

### Step 5: Corner bounding boxes

TL_convex corners pair with TR_convex corners at similar height to define bar width. The bottom is closed either by walking down the mask column until fill drops (floor walk) or by snapping to the nearest fused axis below the top. Boxes are then aligned to the lowest shared bottom among in-plot bars so one baseline fits the whole series. Orange rectangles on the overlay are these boxes. They are what we measure for bar height (Model 5 geometry).

Optional Shi-Tomasi corners exist for experimentation; the default eval path relies on oriented kernels and axis fusion.

## How it is run

| Entry point | Purpose |
|-------------|---------|
| `scaling-detection/pipeline.py` | Core `run_pipeline()` and `Params` dataclass |
| `scaling-detection/app.py` | `streamlit run scaling-detection/app.py` for live tuning |
| `scaling-detection/benchmark_overlap.py` | Batch IoU eval on synthetic charts |
| `scaling-detection/benchmark_overlap.py --save-overlays` | Writes sample overlays to `eval_outputs/` |
| `pytest tests/test_corner_bbox_overlap.py` | CI gate on 80-chart batch |

Default sidebar parameters live in `default_params()`. Benchmarks and tests use `eval_params()` (grayscale corner source, lower corner threshold, axis-based closing instead of floor walk on synthetic data).

## How evaluation went

We score orange corner boxes against ground truth on synthetic vertical bar charts (`synthetic_bw.py`). Matplotlib renders 3–12 categories; even-index bars are black (targets), odd-index bars are white outlines (decoys). Optional grid and title add noise. Matching uses greedy IoU at 0.5 per chart; dataset metrics macro-average precision, recall, F1, and mean IoU over matched pairs.

### Numbers (reproduced locally)

| Run | Charts | Precision | Recall | F1 | Mean IoU | F1 ≥ 0.5 |
|-----|--------|-------------|--------|-----|----------|----------|
| Pytest gate (`eval_params`, seed 42) | 80 | (varies) | ≥ 0.45 required | ≥ 0.50 required | — | ≥ 50% of charts |
| Benchmark CLI (`eval_params`, seed 0) | 200 | 0.865 | 0.856 | 0.846 | 0.917 | 194 / 200 |

On clean synthetic B&W charts, corner pairing plus axis alignment is strong: most charts get usable boxes and high overlap with labels. Failures tend to be odd bar counts, heavy grid, or decoy white bars confusing pairing.

### What the tests do not cover

- Real IBCS dashboards (grey plan bars, hatch forecast, multi-series, labels, chrome)
- Axis detection accuracy in isolation (only bar box IoU is gated)
- Full-page screenshots without a prior crop ([YOLO separation report](./yolo-dashboard-chart-detection-report.md) covers that gap)
- Scaling compliance (label vs height ratios)

Manual review uses images under `Dataset/` (Compliant / non-compliant folders) in the Streamlit app. Overlays in `scaling-detection/eval_outputs/` are examples from `--save-overlays` (green = GT, orange = prediction on synthetic data).

## Strengths

- No training step; behavior is explainable from masks, geometry, and thresholds.
- Handles multiple IBCS fill styles via separate masks and a winner-takes-best scoring rule.
- Axis fusion combines three weak signals instead of betting on one detector.
- Shared baseline alignment matches how column charts are drawn.
- Fast enough for interactive tuning and batch benchmarks on hundreds of synthetic charts.

## Weaknesses

- Parameters are image-dependent; dashboard photos often need cropping first.
- Projection path and corner path can disagree; production logic should prefer one strategy per chart type.
- Hatched and overlapping series still confuse masks on real slides.
- Grey-only or non-bar chart types are outside the design center.
- Floor walk vs axis snap behaves differently on synthetic vs photo charts (`eval_params` disables floor walk for benchmarks).

## Relation to the rest of the project

```text
  Dashboard screenshot
        │
        ▼
  YOLO chart separation (train.py)     ← crop per chart; still weak on dense dashboards
        │
        ▼
  OpenCV pipeline (scaling-detection/) ← baseline + bar boxes
        │
        ▼
  (planned) label OCR + zero-axis      ← scaling verdict
```

The OpenCV stack is the geometry layer. It turned in good numbers on controlled synthetic bars. Moving it to production IBCS slides means better crops, mask thresholds per theme, and eventually pairing with label detection so measured heights can be checked against stated values.

## Suggested next steps

1. Run the Streamlit app on a fixed set of compliant and non-compliant crops and log failure modes (axis too high, merged bars, missed hatch series).
2. Fine-tune `eval_params` on a small hand-labeled set of real chart crops, not only synthetic BW.
3. Report axis Y error separately from bar IoU so Model 2 quality is visible in metrics.
4. Wire the pipeline output format into the scaling check once label detection exists.

## Files worth knowing

| Path | Contents |
|------|----------|
| `scaling-detection/pipeline.py` | Full implementation |
| `scaling-detection/metrics.py` | IoU matching and dataset aggregates |
| `scaling-detection/synthetic_bw.py` | Labeled synthetic charts for eval |
| `scaling-detection/render_pipeline_diagram.py` | Flowchart PNG for docs |
| `tests/test_corner_bbox_overlap.py` | Regression thresholds |
