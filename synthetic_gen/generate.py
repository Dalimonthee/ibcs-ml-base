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

from .chart_factory import create_chart
from .composer import compose
from .yolo_writer import ensure_dirs, get_sample_paths, write_data_yaml, write_sample


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_dataset(config: dict) -> None:
    total = config["total_images"]
    train_ratio = config.get("train_ratio", 0.8)
    canvas_size = tuple(config.get("image_size", [640, 640]))
    seed = config.get("seed", 42)
    weights = config.get("chart_types", None)

    script_dir = Path(__file__).resolve().parent
    output_raw = config.get("output_dir", "../yolo_dataset")
    output_dir = (script_dir / output_raw).resolve()

    dirs = ensure_dirs(output_dir)
    write_data_yaml(output_dir)

    n_train = int(total * train_ratio)

    type_counts: dict[str, int] = {}

    print(f"Generating {total} synthetic bar chart images...")
    print(f"  Output: {output_dir}")
    print(f"  Train: {n_train} | Val: {total - n_train}")
    print(f"  Canvas: {canvas_size[0]}x{canvas_size[1]}")
    print(f"  Seed: {seed}")
    print()

    t_start = time.time()

    for i in range(total):
        rng = random.Random(seed + i)

        split = "train" if i < n_train else "val"
        img_path, lbl_path = get_sample_paths(dirs, i, split)

        fig, ax, style, chart_type = create_chart(rng, weights=weights)

        type_counts[chart_type] = type_counts.get(chart_type, 0) + 1

        canvas_img, bbox = compose(fig, rng, canvas_size=canvas_size,
                                   dpi=style.dpi)

        write_sample(canvas_img, bbox, img_path, lbl_path)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i + 1:>{len(str(total))}}/{total}] "
                  f"{rate:.1f} img/s | ETA {eta:.0f}s | last: {chart_type}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({total / elapsed:.1f} img/s)")
    print(f"\nChart type distribution:")
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct:<25} {count:>5} ({count / total * 100:.1f}%)")
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
    args = parser.parse_args()

    config = load_config(args.config)

    if args.total is not None:
        config["total_images"] = args.total
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output is not None:
        config["output_dir"] = args.output

    generate_dataset(config)


if __name__ == "__main__":
    main()
