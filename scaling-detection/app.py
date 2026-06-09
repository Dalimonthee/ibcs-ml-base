"""Streamlit frontend for the IBCS bar-detection pipeline.

Run with:

    streamlit run scaling-detection/app.py

The sidebar lets you pick any image under ``Dataset/`` and tune every knob
of the three-step OpenCV pipeline live.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pipeline import (
    DATASET_DIR,
    FLOOR_SOURCES,
    IMAGE_EXTS,
    MASK_CAPTIONS,
    MASK_NAMES,
    ORIENTED_SOURCES,
    ORIENTED_TYPES,
    Params,
    REPO_ROOT,
    run_pipeline,
)

try:
    from streamlit_cropper import st_cropper  # type: ignore
    HAS_CROPPER = True
except ImportError:
    HAS_CROPPER = False

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_images(root_str: str) -> list[str]:
    root = Path(root_str)
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


@st.cache_data(show_spinner=False)
def load_bgr(path_str: str) -> np.ndarray | None:
    return cv2.imread(path_str, cv2.IMREAD_COLOR)


def sidebar_controls(image_choices: list[str]) -> tuple[str, Params, dict]:
    with st.sidebar:
        st.header("Image")
        default_idx = (
            image_choices.index("Compliant/102.png")
            if "Compliant/102.png" in image_choices
            else 0
        )
        rel_path = st.selectbox(
            f"Pick a chart ({len(image_choices)} available)",
            image_choices,
            index=default_idx,
        )

        crop_enabled = st.toggle(
            "Crop image (real-time)",
            value=False,
            help=(
                "Drag the pink box on the image to focus the pipeline on a "
                "single chart or region. The full pipeline re-runs on every "
                "drag." if HAS_CROPPER
                else "Install `streamlit-cropper` to enable interactive cropping."
            ),
            disabled=not HAS_CROPPER,
        )
        crop_aspect_label = st.selectbox(
            "Crop aspect ratio",
            options=["Free", "1:1", "4:3", "16:9", "3:2"],
            index=0,
            disabled=not crop_enabled or not HAS_CROPPER,
        )
        crop_box_color = st.color_picker(
            "Crop box color",
            value="#FF1493",
            disabled=not crop_enabled or not HAS_CROPPER,
        )
        aspect_lookup = {
            "Free": None,
            "1:1": (1, 1),
            "4:3": (4, 3),
            "16:9": (16, 9),
            "3:2": (3, 2),
        }
        crop_settings = {
            "enabled": bool(crop_enabled and HAS_CROPPER),
            "aspect_ratio": aspect_lookup[crop_aspect_label],
            "box_color": crop_box_color,
        }

        st.header("Preprocessing")
        invert_input = st.toggle(
            "Invert grayscale (INVERT_INPUT)",
            value=False,
            help="Flip dark/light before any masking. Turn on for dark-themed "
                 "charts or when hatching texture is overwhelming Mask A.",
        )

        st.header("Mask A / B intensity")
        black_max = st.slider("BLACK_MAX  (Mask A: pixels darker than this)", 0, 255, 60)
        grey_low, grey_high = st.slider("GREY_RANGE  (Mask B)", 0, 255, (80, 190))

        st.header("Edge / hatch detection")
        canny_low = st.slider("CANNY_LOW", 0, 255, 50)
        canny_high = st.slider("CANNY_HIGH", 0, 255, 150)
        hatch_close_iters = st.slider(
            "HATCH_CLOSE_ITERS  (Mask C rectangular closing)",
            0, 5, 2,
            help="0 = keep Mask C as raw diagonal-closed edges. Higher fuses "
                 "adjacent hatched bars into a single blob.",
        )

        st.header("Noise filter (per mask)")
        noise_keep_frac = st.slider(
            "NOISE_KEEP_FRAC",
            0.0, 1.0, 0.0, step=0.01,
            help="Keep only connected components whose area is at least this "
                 "fraction of the LARGEST blob in the same mask. 0.0 = off, "
                 "1.0 = only the single biggest blob survives. Useful when a "
                 "mask is full of speckle/text noise.",
        )

        st.header("Per-mask projection finder")
        proj_threshold_frac = st.slider(
            "PROJ_THRESHOLD_FRAC",
            0.0, 1.0, 0.10, step=0.01,
            help="A column/row is 'active' when its projection sum exceeds "
                 "this fraction of the max sum in that mask. Lower = more "
                 "sensitive, higher = stricter.",
        )

        st.header("Corner detection (Shi-Tomasi)")
        corner_enabled = st.toggle("Show corner dots", value=False)
        corner_source = st.radio(
            "Corner source",
            options=["grayscale", "winner mask"],
            index=1,
            horizontal=True,
            help="grayscale = corners of the raw image (catches text/axes too). "
                 "winner mask = corners only inside the chosen mask's blobs.",
        )
        corner_max = st.slider("Max corners", 0, 2000, 300, step=10)
        corner_quality = st.slider(
            "Quality level", 0.001, 0.5, 0.01, step=0.005, format="%.3f",
            help="Minimum corner 'strength' relative to the best corner. Lower = more dots.",
        )
        corner_min_dist = st.slider(
            "Min distance between corners (px)", 1, 50, 6,
            help="Suppresses clusters of dots at the same physical corner.",
        )

        st.header("Oriented corners (Option B kernels)")
        st.caption(
            "Four signed 4-quadrant kernels. "
            "Green = TL-convex, Cyan = TR-convex, Yellow = TL-concave, "
            "Magenta = TR-concave."
        )
        oriented_enabled = st.toggle("Enable oriented corner detection", value=True)
        oriented_source = st.selectbox(
            "Source",
            options=ORIENTED_SOURCES,
            index=ORIENTED_SOURCES.index("Mask A"),
            help="Which image to run the kernels on. Binary masks (A/B/C/"
                 "filled/Combined/winner) give the cleanest results. "
                 "'grayscale' is auto-inverted when the background is light so "
                 "bars become high-valued.",
        )
        oriented_kernel_size = st.slider(
            "Kernel size (px, even)",
            4, 64, 16, step=2,
            help="Window size of the 4-quadrant kernel. Smaller = more local "
                 "(robust to adjacent bars) but noisier; larger = stronger "
                 "signal but vulnerable to neighbouring bars. Pick a value "
                 "noticeably smaller than your typical bar width.",
        )
        oriented_threshold_frac = st.slider(
            "Response threshold (fraction of peak)",
            0.0, 1.0, 0.85, step=0.05,
            help="A pixel is a corner if its kernel response >= this fraction "
                 "of the strongest response. 0.67 = a straight edge starts "
                 "leaking through, so keep at 0.80+ to isolate true corners.",
        )
        oriented_min_dist = st.slider(
            "NMS min distance (px)",
            1, 50, 6,
            help="Non-max-suppression radius - suppresses clusters at the same "
                 "physical corner.",
        )

        st.header("Bounding box from corners")
        st.caption(
            "Detect horizontal x-axes from multiple signals (corner clusters, "
            "morphological horizontal strokes, Hough lines), then pair "
            "TL-convex + TR-convex corners. Each bar is closed by walking down "
            "its column strip (preferred) or by the nearest axis below."
        )
        bbox_from_corners_enabled = st.toggle(
            "Build bounding boxes from corners", value=True,
        )

        st.subheader("Axis detection signals")
        axis_use_corners = st.toggle("Use corner clustering", value=True)
        bbox_axis_type_choices = st.multiselect(
            "Corner types used for axis detection",
            options=ORIENTED_TYPES,
            default=["TR_convex", "TR_concave", "TL_concave", "TL_convex"],
            help="Concave + label-text corners cluster on the axis most "
                 "reliably; all four types makes detection more robust.",
            disabled=not axis_use_corners,
        )
        axis_corner_weight = st.slider(
            "Corner-cluster weight",
            0.0, 3.0, 1.5, step=0.1,
            help="Bias toward the corner signal: it knows about bars "
                 "specifically, whereas morph/hough fire on any horizontal "
                 "stroke (chart frames, gridlines, etc.).",
            disabled=not axis_use_corners,
        )

        axis_use_morph = st.toggle(
            "Use horizontal morphological opening",
            value=True,
            help="Best at finding visible axis strokes even when no corners "
                 "fire on the axis row. Applies a 1xN open kernel to the Canny "
                 "edges and picks rows with strong remaining response.",
        )
        axis_morph_kernel_w_frac = st.slider(
            "Morph kernel width (fraction of image width)",
            0.05, 0.5, 0.15, step=0.01,
            help="The kernel keeps only horizontal strokes at least this wide. "
                 "Long enough to ignore tick marks and bar tops, short enough "
                 "to catch a partial axis.",
            disabled=not axis_use_morph,
        )
        axis_morph_abs_floor = st.slider(
            "Morph absolute row-sum floor",
            0, 5000, 200,
            help="Row sums below this many surviving pixels are ignored.",
            disabled=not axis_use_morph,
        )
        axis_morph_rel_floor = st.slider(
            "Morph relative row-sum floor",
            0.0, 1.0, 0.30, step=0.05,
            help="Row sums below this fraction of the maximum row sum are "
                 "ignored. Higher = pickier.",
            disabled=not axis_use_morph,
        )
        axis_morph_weight = st.slider(
            "Morph weight",
            0.0, 3.0, 1.5, step=0.1,
            disabled=not axis_use_morph,
        )

        axis_use_hough = st.toggle(
            "Use probabilistic Hough lines",
            value=True,
            help="Direct line-segment detection. Robust on clean charts with "
                 "thin axis strokes.",
        )
        axis_hough_min_len_frac = st.slider(
            "Hough min line length (fraction of image width)",
            0.05, 1.0, 0.20, step=0.05,
            disabled=not axis_use_hough,
        )
        axis_hough_threshold = st.slider(
            "Hough threshold (accumulator votes)",
            10, 500, 80, step=5,
            disabled=not axis_use_hough,
        )
        axis_hough_max_gap = st.slider(
            "Hough max gap (px)",
            0, 100, 20,
            disabled=not axis_use_hough,
        )
        axis_hough_angle_tol_deg = st.slider(
            "Hough angle tolerance (degrees from horizontal)",
            0.0, 10.0, 2.0, step=0.5,
            disabled=not axis_use_hough,
        )
        axis_hough_weight = st.slider(
            "Hough weight",
            0.0, 3.0, 1.5, step=0.1,
            disabled=not axis_use_hough,
        )

        st.subheader("Axis fusion + pairing")
        bbox_axis_y_tolerance = st.slider(
            "Axis Y tolerance (px)",
            1, 20, 4,
            help="Vertical thickness for considering signals 'on the same row' "
                 "during fusion and corner clustering.",
        )
        bbox_axis_min_count = st.slider(
            "Min corners per cluster (corner signal only)",
            2, 30, 3,
            help="Corner clusters with fewer members than this are dropped "
                 "before fusion.",
        )
        axis_min_total_support = st.slider(
            "Min fused support",
            0.0, 5.0, 0.3, step=0.05,
            help="Fused peaks below this combined weight are rejected. Higher "
                 "= require agreement between multiple detectors.",
        )
        bbox_axis_max_count = st.slider(
            "Max axes per image (multi-chart support)",
            1, 12, 5,
            help="Several charts on the same image -> several x-axes. Keep "
                 "this generous so chart frames don't crowd out true axes.",
        )
        bbox_pair_y_tolerance = st.slider(
            "TL/TR pairing Y tolerance (px)",
            1, 30, 6,
            help="Max vertical difference between a TL-convex and the "
                 "TR-convex it pairs with (i.e. how level the bar's top must be).",
        )

        st.subheader("Per-bar floor (preferred close)")
        floor_enabled = st.toggle(
            "Walk down each bar to find its bottom",
            value=True,
            help="For each TL/TR pair, walk down the column strip until the "
                 "bar fill drops below the active threshold. Independent of "
                 "the global axis Y. Falls back to the axis when the walk "
                 "yields no usable result.",
        )
        floor_source = st.selectbox(
            "Floor source mask",
            options=FLOOR_SOURCES,
            index=FLOOR_SOURCES.index("winner mask"),
            help="Which binary mask to walk down. The winner mask is usually "
                 "right, but on hatched or outlined charts a specific mask "
                 "may be much cleaner.",
            disabled=not floor_enabled,
        )
        floor_active_frac = st.slider(
            "Floor min active fraction",
            0.1, 1.0, 0.5, step=0.05,
            help="Minimum fraction of the column strip that must be active "
                 "for a row to count as part of the bar. Raise on noisy masks.",
            disabled=not floor_enabled,
        )

        st.subheader("Bar size filters")
        bbox_min_width = st.slider("Bar min width (px)", 1, 100, 6)
        bbox_max_width_frac = st.slider(
            "Bar max width (fraction of image width)",
            0.05, 1.0, 0.30, step=0.05,
            help="Upper bound when searching for a TR partner to the right "
                 "of a TL corner.",
        )
        bbox_min_height = st.slider("Bar min height (px)", 1, 200, 8)
        bbox_max_height_frac = st.slider(
            "Bar max height (fraction of image height)",
            0.05, 1.0, 0.60, step=0.05,
            help="Rejects pairings whose implied height spans most of the "
                 "image (e.g. a chart title TL paired with the x-axis below).",
        )

        st.header("Dynamic size + rectangularity")
        min_area_floor = st.slider("MIN_AREA_FLOOR  (px, absolute)", 10, 5000, 100, step=10)
        min_area_fraction = st.slider(
            "MIN_AREA_FRACTION  (relative to largest box from same mask)",
            0.0, 1.0, 0.05, step=0.01,
        )
        min_fill_ratio = st.slider(
            "MIN_FILL_RATIO  (rectangularity)",
            0.0, 1.0, 0.55, step=0.05,
            help="filled / bbox_area. 1.0 = perfect rectangle.",
        )
        min_bar_w = st.slider("MIN_BAR_W", 1, 100, 4)
        min_bar_h = st.slider("MIN_BAR_H", 1, 100, 4)
        max_aspect = st.slider("MAX_ASPECT", 1.0, 100.0, 25.0, step=1.0)

    params = Params(
        invert_input=invert_input,
        black_max=black_max,
        grey_low=grey_low,
        grey_high=grey_high,
        canny_low=canny_low,
        canny_high=canny_high,
        hatch_close_iters=hatch_close_iters,
        noise_keep_frac=noise_keep_frac,
        proj_threshold_frac=proj_threshold_frac,
        min_area_floor=min_area_floor,
        min_area_fraction=min_area_fraction,
        min_fill_ratio=min_fill_ratio,
        min_bar_w=min_bar_w,
        min_bar_h=min_bar_h,
        max_aspect=max_aspect,
        corner_enabled=corner_enabled,
        corner_source=corner_source,
        corner_max=corner_max,
        corner_quality=corner_quality,
        corner_min_dist=corner_min_dist,
        oriented_enabled=oriented_enabled,
        oriented_source=oriented_source,
        oriented_kernel_size=oriented_kernel_size,
        oriented_threshold_frac=oriented_threshold_frac,
        oriented_min_dist=oriented_min_dist,
        bbox_from_corners_enabled=bbox_from_corners_enabled,
        bbox_axis_types=tuple(bbox_axis_type_choices),
        bbox_axis_y_tolerance=bbox_axis_y_tolerance,
        bbox_axis_min_count=bbox_axis_min_count,
        bbox_axis_max_count=bbox_axis_max_count,
        bbox_pair_y_tolerance=bbox_pair_y_tolerance,
        bbox_min_width=bbox_min_width,
        bbox_max_width_frac=bbox_max_width_frac,
        bbox_min_height=bbox_min_height,
        bbox_max_height_frac=bbox_max_height_frac,
        axis_use_corners=axis_use_corners,
        axis_use_morph=axis_use_morph,
        axis_use_hough=axis_use_hough,
        axis_corner_weight=axis_corner_weight,
        axis_morph_weight=axis_morph_weight,
        axis_hough_weight=axis_hough_weight,
        axis_min_total_support=axis_min_total_support,
        axis_morph_kernel_w_frac=axis_morph_kernel_w_frac,
        axis_morph_abs_floor=axis_morph_abs_floor,
        axis_morph_rel_floor=axis_morph_rel_floor,
        axis_hough_min_len_frac=axis_hough_min_len_frac,
        axis_hough_threshold=axis_hough_threshold,
        axis_hough_max_gap=axis_hough_max_gap,
        axis_hough_angle_tol_deg=axis_hough_angle_tol_deg,
        floor_enabled=floor_enabled,
        floor_source=floor_source,
        floor_active_frac=floor_active_frac,
    )
    return rel_path, params, crop_settings


def render_mask_overlay(rgb: np.ndarray, boxes: list[tuple], color=(0, 180, 255)) -> np.ndarray:
    """Return a copy of rgb with the given boxes drawn on it (thin, distinct)."""
    out = rgb.copy()
    for x, y, w, h, *_ in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
    return out


def main() -> None:
    st.set_page_config(page_title="IBCS bar detection", layout="wide")
    st.title("IBCS bar detection - per-mask projection explorer")
    st.caption(
        "Each mask runs its own X-then-Y projection bar finder. The mask with "
        "the highest score wins; the others (including the combined fallback) "
        "are ignored unless the winner found nothing."
    )

    image_choices = list_images(str(DATASET_DIR))
    if not image_choices:
        st.error(
            f"No images found under {DATASET_DIR}. "
            "Expected the repo's Dataset/ folder to sit one level above this script."
        )
        st.stop()

    rel_path, params, crop_settings = sidebar_controls(image_choices)
    abs_path = DATASET_DIR / rel_path
    bgr_full = load_bgr(str(abs_path))
    if bgr_full is None:
        st.error(f"Could not read image: {abs_path}")
        st.stop()

    if crop_settings["enabled"]:
        st.subheader(f"Crop  -  {rel_path}")
        st.caption(
            "Drag the box on the image. The full pipeline re-runs in real time "
            "on each crop change."
        )
        rgb_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2RGB)
        pil_full = Image.fromarray(rgb_full)
        cropped_pil = st_cropper(
            pil_full,
            realtime_update=True,
            box_color=crop_settings["box_color"],
            aspect_ratio=crop_settings["aspect_ratio"],
            return_type="image",
            key=f"cropper::{rel_path}",
        )
        bgr = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
        ch, cw = bgr.shape[:2]
        oh, ow = bgr_full.shape[:2]
        st.caption(
            f"crop: {cw} x {ch} px  (full: {ow} x {oh} px,  "
            f"{(cw * ch) / max(1, ow * oh):.0%} of original area)"
        )
        st.divider()
    elif not HAS_CROPPER:
        st.info(
            "Run `pip install streamlit-cropper` to enable the real-time "
            "cropping widget."
        )
        bgr = bgr_full
    else:
        bgr = bgr_full

    if bgr.size == 0 or bgr.shape[0] < 4 or bgr.shape[1] < 4:
        st.warning("Crop is too small to run the pipeline; drag a larger box.")
        st.stop()

    out = run_pipeline(bgr, params)
    winner = out["winner"]
    winner_data = out["per_mask"][winner]

    col_orig, col_det = st.columns(2)
    with col_orig:
        st.subheader(
            "Original (cropped)" if crop_settings["enabled"] else f"Original  -  {rel_path}"
        )
        st.image(out["rgb"], use_container_width=True)
    with col_det:
        st.subheader(
            f"Detected: {len(out['winner_boxes'])} bars  "
            f"(winner: Mask {winner})"
        )
        st.image(out["overlay"], use_container_width=True)
        oriented_summary = "  ".join(
            f"{n.replace('_', '-')}={len(out['oriented_corners'][n])}"
            for n in ORIENTED_TYPES
        )
        st.caption(
            f"winner score: {winner_data['score']:.2f}  |  "
            f"shi-tomasi: {len(out['corners'])}  |  "
            f"oriented (source: {params.oriented_source}): {oriented_summary}"
        )
        if params.bbox_from_corners_enabled:
            axes_str = (
                ", ".join(
                    f"y={y} ({src}, w={w:.2f})" for y, w, src in out["axes_peaks"]
                )
                if out["axes_peaks"] else "none"
            )
            close_counts: dict[str, int] = {}
            for *_, src in out["corner_bboxes"]:
                close_counts[src] = close_counts.get(src, 0) + 1
            close_str = (
                ", ".join(f"{k}={v}" for k, v in close_counts.items())
                if close_counts else "none"
            )
            st.caption(
                f"corner bboxes (orange): {len(out['corner_bboxes'])} ({close_str})  |  "
                f"detected x-axes (pink): {axes_str}"
            )
            sig = out.get("axis_signals", {})
            sig_summary = "  ".join(
                f"{k}={len(v)}" for k, v in sig.items()
            ) if sig else ""
            if sig_summary:
                st.caption(f"axis signal raw peaks: {sig_summary}")
        st.caption(
            "other masks: " + "  ".join(
                f"{n}={len(out['per_mask'][n]['boxes'])}@{out['per_mask'][n]['score']:.2f}"
                for n in MASK_NAMES if n != winner
            )
        )

    st.divider()
    st.subheader("Per-mask detection (each mask projects independently)")
    cols = st.columns(len(MASK_NAMES))
    for col, name in zip(cols, MASK_NAMES):
        info = out["per_mask"][name]
        is_winner = name == winner
        col.image(
            render_mask_overlay(
                cv2.cvtColor(info["mask"], cv2.COLOR_GRAY2RGB),
                info["boxes"],
                color=(0, 220, 0) if is_winner else (255, 140, 0),
            ),
            caption=(
                f"{'WINNER - ' if is_winner else ''}"
                f"{MASK_CAPTIONS[name]}  |  "
                f"{len(info['boxes'])} boxes  |  "
                f"score={info['score']:.2f}"
            ),
            use_container_width=True,
            clamp=True,
        )

    with st.expander("Working grayscale (after optional inversion)"):
        st.image(out["gray"], use_container_width=True, clamp=True)

    if params.oriented_enabled:
        with st.expander(
            f"Oriented-corner source image  -  '{params.oriented_source}' "
            "(this is what the 4 kernels actually convolve over)"
        ):
            st.image(out["oriented_source_img"], use_container_width=True, clamp=True)

    st.divider()
    st.subheader(f"Projection profiles  -  winner: Mask {winner}")
    winner_mask = winner_data["mask"]
    px_col, py_col = st.columns(2)
    with px_col:
        st.line_chart(
            {"column sum": winner_mask.sum(axis=0).astype(np.int64)},
            height=200,
        )
        st.caption("X projection - peaks correspond to bar columns")
    with py_col:
        st.line_chart(
            {"row sum": winner_mask.sum(axis=1).astype(np.int64)},
            height=200,
        )
        st.caption("Y projection - peaks correspond to bar rows")

    with st.expander(f"Winner bbox details ({len(out['winner_boxes'])} boxes)"):
        if out["winner_boxes"]:
            st.dataframe(
                {
                    "x": [b[0] for b in out["winner_boxes"]],
                    "y": [b[1] for b in out["winner_boxes"]],
                    "w": [b[2] for b in out["winner_boxes"]],
                    "h": [b[3] for b in out["winner_boxes"]],
                    "fill_ratio": [round(b[4], 3) for b in out["winner_boxes"]],
                    "filled_px": [int(round(b[4] * b[2] * b[3])) for b in out["winner_boxes"]],
                },
                use_container_width=True,
            )
        else:
            st.info(
                "No bars survived in any mask. Try lowering "
                "PROJ_THRESHOLD_FRAC / MIN_FILL_RATIO / MIN_AREA_FLOOR, "
                "or toggling INVERT_INPUT."
            )


if __name__ == "__main__":
    main()