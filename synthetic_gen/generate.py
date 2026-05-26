"""CLI entrypoint for synthetic bar chart dataset generation."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import yaml
from PIL import Image, ImageDraw

from .chart_factory import create_chart
from .composer import compose, compose_multi
from .layouts import pick_chart_count, pick_layout
from .styling import random_style
from .yolo_writer import ensure_dirs, get_sample_paths, write_data_yaml, write_sample


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_charts_per_image(raw: dict | None) -> dict[int, float]:
    if not raw:
        return {1: 0.40, 2: 0.25, 3: 0.25, 4: 0.10}
    return {int(k): float(v) for k, v in raw.items()}


def _draw_bbox_overlay(image: Image.Image,
                       bboxes: list[tuple[float, float, float, float]]) -> Image.Image:
    """Draw YOLO bboxes on a copy of the image for visual QA."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i, (xc, yc, bw, bh) in enumerate(bboxes):
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=2)
        draw.text((x1 + 2, y1 + 2), f"chart {i + 1}", fill=(0, 120, 255))
    return out


def _generate_sample(rng: random.Random,
                     config: dict,
                     charts_per_image: dict[int, float],
                     chart_weights: dict[str, float] | None) -> tuple:
    """Create one synthetic sample: (canvas, bboxes, metadata dict)."""
    canvas_size = tuple(config.get("image_size", [640, 640]))
    gap_px = config.get("gap_px", 24)
    min_slot_fill = config.get("min_slot_fill", 0.85)

    n_charts = pick_chart_count(rng, charts_per_image)

    if n_charts == 1:
        fig, _ax, style, chart_type = create_chart(rng, weights=chart_weights)
        canvas_img, bbox = compose(
            fig, rng, canvas_size=canvas_size, dpi=style.dpi,
        )
        return canvas_img, [bbox], {
            "n_charts": 1,
            "layout": None,
            "chart_types": [chart_type],
        }

    shared_style = random_style(rng)
    figures = []
    chart_types = []
    dpi = shared_style.dpi

    for _ in range(n_charts):
        fig, _ax, style, chart_type = create_chart(
            rng, weights=chart_weights, shared_style=shared_style,
        )
        figures.append(fig)
        chart_types.append(chart_type)
        dpi = style.dpi

    layout = pick_layout(rng, n_charts)
    canvas_img, bboxes = compose_multi(
        figures, rng, layout,
        canvas_size=canvas_size,
        dpi=dpi,
        gap_px=gap_px,
        min_slot_fill=min_slot_fill,
    )
    return canvas_img, bboxes, {
        "n_charts": n_charts,
        "layout": layout,
        "chart_types": chart_types,
    }


def generate_preview(config: dict, n_preview: int, output_dir: Path) -> None:
    """Write preview images with bbox overlays for visual QA."""
    seed = config.get("seed", 42)
    charts_per_image = _parse_charts_per_image(config.get("charts_per_image"))
    chart_weights = config.get("chart_types")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {n_preview} preview images to {output_dir}")

    counts: dict[int, int] = {}
    for i in range(n_preview):
        rng = random.Random(seed + 100000 + i)
        canvas_img, bboxes, meta = _generate_sample(
            rng, config, charts_per_image, chart_weights,
        )
        counts[meta["n_charts"]] = counts.get(meta["n_charts"], 0) + 1

        overlay = _draw_bbox_overlay(canvas_img, bboxes)
        layout_suffix = meta["layout"] or "single"
        out_path = output_dir / f"preview_{i:03d}_{meta['n_charts']}charts_{layout_suffix}.png"
        overlay.save(str(out_path))

    print("Preview chart-count distribution:")
    for n in sorted(counts):
        print(f"  {n} charts: {counts[n]}")


def generate_dataset(config: dict) -> None:
    total = config["total_images"]
    train_ratio = config.get("train_ratio", 0.8)
    seed = config.get("seed", 42)
    weights = config.get("chart_types", None)
    charts_per_image = _parse_charts_per_image(config.get("charts_per_image"))

    script_dir = Path(__file__).resolve().parent
    output_raw = config.get("output_dir", "../yolo_dataset")
    output_dir = (script_dir / output_raw).resolve()

    dirs = ensure_dirs(output_dir)
    write_data_yaml(output_dir)

    n_train = int(total * train_ratio)

    type_counts: dict[str, int] = {}
    chart_count_dist: dict[int, int] = {}
    layout_counts: dict[str, int] = {}

    print(f"Generating {total} synthetic bar chart images...")
    print(f"  Output: {output_dir}")
    print(f"  Train: {n_train} | Val: {total - n_train}")
    print(f"  Canvas: {config.get('image_size', [640, 640])}")
    print(f"  Charts per image: {charts_per_image}")
    print(f"  Seed: {seed}")
    print()

    t_start = time.time()

    for i in range(total):
        rng = random.Random(seed + i)

        split = "train" if i < n_train else "val"
        img_path, lbl_path = get_sample_paths(dirs, i, split)

        canvas_img, bboxes, meta = _generate_sample(
            rng, config, charts_per_image, weights,
        )

        for ct in meta["chart_types"]:
            type_counts[ct] = type_counts.get(ct, 0) + 1

        n_charts = meta["n_charts"]
        chart_count_dist[n_charts] = chart_count_dist.get(n_charts, 0) + 1
        if meta["layout"]:
            layout_counts[meta["layout"]] = layout_counts.get(meta["layout"], 0) + 1

        write_sample(canvas_img, bboxes, img_path, lbl_path)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            last = meta["layout"] or meta["chart_types"][0]
            print(f"  [{i + 1:>{len(str(total))}}/{total}] "
                  f"{rate:.1f} img/s | ETA {eta:.0f}s | last: {n_charts}ch {last}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({total / elapsed:.1f} img/s)")

    print(f"\nCharts per image distribution:")
    for n in sorted(chart_count_dist):
        c = chart_count_dist[n]
        print(f"  {n} chart(s): {c:>5} ({c / total * 100:.1f}%)")

    if layout_counts:
        print(f"\nMulti-chart layout distribution:")
        for layout, count in sorted(layout_counts.items()):
            print(f"  {layout:<25} {count:>5}")

    print(f"\nChart type distribution (across all charts):")
    total_charts = sum(type_counts.values())
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct:<25} {count:>5} ({count / total_charts * 100:.1f}%)")

    print(f"\nDataset written to: {output_dir}")
    print(f"  data.yaml: {output_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bar chart dataset for YOLOv8"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--total", type=int, default=None,
        help="Override total number of images to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--preview", type=int, default=None, metavar="N",
        help="Generate N preview images with bbox overlays (no full dataset)",
    )
    parser.add_argument(
        "--preview-dir", type=str, default=None,
        help="Output directory for preview images (default: preview/)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.total is not None:
        config["total_images"] = args.total
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output is not None:
        config["output_dir"] = args.output

    if args.preview is not None:
        script_dir = Path(__file__).resolve().parent
        preview_dir = Path(args.preview_dir) if args.preview_dir else script_dir / "preview"
        generate_preview(config, args.preview, preview_dir)
        return

    generate_dataset(config)


if __name__ == "__main__":
    main()
