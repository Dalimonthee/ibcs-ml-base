"""Write images and YOLO-format annotation files."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from PIL import Image


CLASS_ID = 0
CLASS_NAME = "bar_chart"


def ensure_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create the YOLO dataset directory structure and return paths."""
    base = Path(output_dir)
    paths = {
        "images_train": base / "images" / "train",
        "images_val": base / "images" / "val",
        "labels_train": base / "labels" / "train",
        "labels_val": base / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _as_bbox_list(
    bboxes: tuple[float, float, float, float]
    | list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    if isinstance(bboxes, tuple):
        return [bboxes]
    return list(bboxes)


def write_sample(image: Image.Image,
                 bboxes: tuple[float, float, float, float]
                 | list[tuple[float, float, float, float]],
                 image_path: Path,
                 label_path: Path) -> None:
    """Save a single image and its YOLO annotation file(s)."""
    image.save(str(image_path), format="PNG")

    bbox_list = _as_bbox_list(bboxes)
    lines = [
        f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        for x_center, y_center, w, h in bbox_list
    ]

    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_data_yaml(output_dir: str | Path) -> None:
    """Write the data.yaml file for YOLOv8 training."""
    base = Path(output_dir)
    data = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [CLASS_NAME],
    }
    yaml_path = base / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_sample_paths(dirs: dict[str, Path], index: int,
                     split: str) -> tuple[Path, Path]:
    """Return (image_path, label_path) for a given sample index and split."""
    name = f"img_{index:05d}"
    img_dir = dirs[f"images_{split}"]
    lbl_dir = dirs[f"labels_{split}"]
    return img_dir / f"{name}.png", lbl_dir / f"{name}.txt"
