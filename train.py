"""YOLOv8 training, validation, and prediction pipeline for bar chart detection."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

# Python 3.14 on macOS may lack the _lzma C extension even when xz is
# installed, because pyenv compiled CPython before xz was available.
# torchvision unconditionally imports lzma at startup, and ultralytics
# calls tarfile.is_tarfile() which probes xz decompression.  We inject a
# stub whose (De)Compressor classes raise immediately -- identical to the
# behavior when a file simply isn't lzma-compressed.
if "_lzma" not in sys.modules:
    try:
        import _lzma  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("_lzma")

        class _LZMAError(Exception):
            pass

        class _LZMACompressor:
            def __init__(self, *a, **kw):
                raise _LZMAError("_lzma stub: compression not available")
            def compress(self, data): raise _LZMAError("stub")
            def flush(self): return b""

        class _LZMADecompressor:
            def __init__(self, *a, **kw):
                raise _LZMAError("_lzma stub: decompression not available")
            def decompress(self, data, max_length=-1): raise _LZMAError("stub")

        _stub.LZMACompressor = _LZMACompressor
        _stub.LZMADecompressor = _LZMADecompressor
        _stub.LZMAError = _LZMAError
        for _attr, _val in [
            ("FORMAT_AUTO", 0), ("FORMAT_XZ", 1), ("FORMAT_ALONE", 2),
            ("FORMAT_RAW", 3), ("CHECK_NONE", 0), ("CHECK_CRC32", 1),
            ("CHECK_CRC64", 4), ("CHECK_SHA256", 10), ("CHECK_ID_MAX", 15),
            ("CHECK_UNKNOWN", 16), ("MF_HC3", 3), ("MF_HC4", 4),
            ("MF_BT2", 18), ("MF_BT3", 19), ("MF_BT4", 20),
            ("MODE_FAST", 1), ("MODE_NORMAL", 2), ("PRESET_DEFAULT", 6),
            ("PRESET_EXTREME", 0),
        ]:
            setattr(_stub, _attr, _val)
        _stub._encode_filter_properties = lambda *a: b""
        _stub._decode_filter_properties = lambda *a: {}
        _stub.is_check_supported = lambda *a: True
        sys.modules["_lzma"] = _stub

# Also patch tarfile.is_tarfile -- in Python 3.14 it probes xz
# decompression via our stub and the resulting error isn't caught because
# _LZMAError is not a subclass of TarError.
import tarfile as _tarfile
_orig_is_tarfile = _tarfile.is_tarfile
def _safe_is_tarfile(name):
    try:
        return _orig_is_tarfile(name)
    except Exception:
        return False
_tarfile.is_tarfile = _safe_is_tarfile

from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
DATA_YAML = PROJECT_ROOT / "yolo_dataset" / "data.yaml"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def collect_images(base_dir: Path) -> dict[str, list[Path]]:
    """Collect all image files from Dataset/, grouped by subfolder."""
    groups: dict[str, list[Path]] = {}
    for subfolder in sorted(base_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        images = [
            f for f in sorted(subfolder.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            groups[subfolder.name] = images
    return groups


def train(args) -> YOLO:
    """Phase 1: Train YOLOv8 on the synthetic dataset."""
    print("=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    return model


def validate(model: YOLO, args) -> None:
    """Phase 2: Validate on the validation set and print metrics."""
    print("\n" + "=" * 60)
    print("PHASE 2: VALIDATION")
    print("=" * 60)

    metrics = model.val(data=str(args.data), imgsz=args.imgsz)

    print("\n--- Validation Results ---")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


def predict(model: YOLO, args) -> None:
    """Phase 3: Run prediction on real Dataset/ images and save annotated outputs."""
    print("\n" + "=" * 60)
    print("PHASE 3: PREDICTION ON REAL IMAGES")
    print("=" * 60)

    image_groups = collect_images(DATASET_DIR)
    if not image_groups:
        print(f"  No images found in {DATASET_DIR}")
        return

    total_images = sum(len(imgs) for imgs in image_groups.values())
    print(f"  Found {total_images} images across {len(image_groups)} folders\n")

    total_detections = 0
    images_with_detections = 0
    all_confidences: list[float] = []

    for group_name, images in image_groups.items():
        out_dir = PREDICTIONS_DIR / group_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Processing {group_name}/ ({len(images)} images)...")

        results = model.predict(
            source=[str(img) for img in images],
            conf=args.conf,
            imgsz=args.imgsz,
            save=False,
            verbose=False,
        )

        for img_path, result in zip(images, results):
            n_boxes = len(result.boxes)
            total_detections += n_boxes
            if n_boxes > 0:
                images_with_detections += 1
                for box in result.boxes:
                    all_confidences.append(float(box.conf[0]))

            annotated = result.plot()

            out_path = out_dir / img_path.name
            Image.fromarray(annotated[..., ::-1]).save(str(out_path))

        print(f"    -> Saved {len(images)} annotated images to {out_dir}")

    print("\n--- Prediction Summary ---")
    print(f"  Total images processed:    {total_images}")
    print(f"  Images with detections:    {images_with_detections} "
          f"({images_with_detections / total_images * 100:.1f}%)")
    print(f"  Total bounding boxes:      {total_detections}")
    if all_confidences:
        avg_conf = sum(all_confidences) / len(all_confidences)
        print(f"  Average confidence:        {avg_conf:.4f}")
        print(f"  Min confidence:            {min(all_confidences):.4f}")
        print(f"  Max confidence:            {max(all_confidences):.4f}")
    print(f"\n  Annotated images saved to: {PREDICTIONS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 bar chart detection: train, validate, predict"
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model variant (default: yolov8n.pt)")
    parser.add_argument("--data", type=str, default=str(DATA_YAML),
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training, only run prediction")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained weights (used with --predict-only)")
    args = parser.parse_args()

    if args.predict_only:
        weights = args.weights
        if weights is None:
            default = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
            if default.exists():
                weights = str(default)
            else:
                print(f"ERROR: No weights found at {default}")
                print("  Provide --weights or run training first.")
                return
        print(f"Loading model from: {weights}")
        model = YOLO(weights)
        validate(model, args)
        predict(model, args)
    else:
        model = train(args)
        validate(model, args)
        predict(model, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
