#!/usr/bin/env python3
"""Build a COCO-indexed YOLO copy of the KITTI val split for Ultralytics mAP.

The problem this solves: exp8 passes the string "kitti.yaml" to model.val(),
which resolves to Ultralytics' bundled KITTI config -- a different dataset with
a 9-class taxonomy, scored against a COCO-pretrained 80-class model. The result
is a class-index mismatch, not a measurement.

This script writes a parallel dataset whose labels are YOLO-format boxes carrying
COCO class indices, so the pretrained model is scored against compatible labels.

    ~/rams/data/kitti_yolo/images/val/   symlinks to the original PNGs
    ~/rams/data/kitti_yolo/labels/val/   converted .txt labels
    <package>/results/exp8_kitti_local.yaml

IMPORTANT INTERPRETATION LIMITS -- state these in the paper:
  * This is a ZERO-SHOT cross-dataset evaluation. A COCO-pretrained detector is
    being scored on KITTI. Numbers are not comparable to the official KITTI
    benchmark, which uses KITTI-trained models and its own protocol.
  * KITTI DontCare regions are dropped from ground truth, not treated as ignore
    regions. Ultralytics has no ignore-region support, so detections landing in
    those areas still count as false positives and depress mAP and precision.
  * Cyclist maps to person: the KITTI Cyclist box covers rider plus bicycle,
    while COCO separates them. This is a judgement call, and it is the mapping
    most favourable to neither side. Document it.

Usage:
    python build_kitti_yolo.py
    python build_kitti_yolo.py --kitti-root ~/rams/data/kitti --out ~/rams/data/kitti_yolo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# KITTI class -> COCO class index (80-class COCO ordering).
# Unmapped classes (Misc, DontCare) are dropped entirely.
KITTI_TO_COCO = {
    "car": 2,             # car
    "van": 2,             # car
    "truck": 7,           # truck
    "pedestrian": 0,      # person
    "person_sitting": 0,  # person
    "cyclist": 0,         # person  (see interpretation limits above)
    "tram": 6,            # train
}
DROPPED = {"misc", "dontcare"}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def convert(kitti_root: Path, out_root: Path) -> tuple[int, int, dict[str, int]]:
    src_images = kitti_root / "images/val"
    src_labels = kitti_root / "labels/val"
    dst_images = out_root / "images/val"
    dst_labels = out_root / "labels/val"

    if not src_images.is_dir():
        sys.exit(f"ERROR: {src_images} not found")
    if not src_labels.is_dir():
        sys.exit(f"ERROR: {src_labels} not found")

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(src_images.glob("*.png"))
    if not images:
        sys.exit(f"ERROR: no PNG files in {src_images}")

    n_boxes = 0
    class_counts: dict[str, int] = {}
    skipped_missing = 0

    for image_path in images:
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped_missing += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped_missing += 1
            continue
        height, width = image.shape[:2]

        lines: list[str] = []
        for raw in label_path.read_text().splitlines():
            parts = raw.split()
            if len(parts) < 15:
                continue
            name = parts[0].lower()
            if name in DROPPED:
                class_counts[name] = class_counts.get(name, 0) + 1
                continue
            coco_index = KITTI_TO_COCO.get(name)
            if coco_index is None:
                class_counts[f"unmapped:{name}"] = class_counts.get(f"unmapped:{name}", 0) + 1
                continue

            x1, y1, x2, y2 = (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
            x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 1.0 or box_h <= 1.0:
                continue

            xc = (x1 + box_w / 2.0) / width
            yc = (y1 + box_h / 2.0) / height
            lines.append(f"{coco_index} {xc:.6f} {yc:.6f} {box_w / width:.6f} {box_h / height:.6f}")
            class_counts[name] = class_counts.get(name, 0) + 1
            n_boxes += 1

        (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        link = dst_images / image_path.name
        if not link.exists():
            try:
                link.symlink_to(image_path.resolve())
            except OSError:
                from shutil import copy2
                copy2(image_path, link)

    if skipped_missing:
        print(f"  warning: skipped {skipped_missing} images (missing label or unreadable)")
    return len(images) - skipped_missing, n_boxes, class_counts


def write_yaml(out_root: Path, yaml_path: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_NAMES))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"# Generated by build_kitti_yolo.py -- KITTI val split, COCO class indices.\n"
        f"# Zero-shot cross-dataset evaluation. Not comparable to the official KITTI benchmark.\n"
        f"path: {out_root.resolve()}\n"
        f"train: images/val\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, default=Path.home() / "rams/data/kitti")
    parser.add_argument("--out", type=Path, default=Path.home() / "rams/data/kitti_yolo")
    parser.add_argument("--yaml", type=Path, default=Path("results/exp8_kitti_local.yaml"))
    args = parser.parse_args()

    kitti_root = args.kitti_root.expanduser()
    out_root = args.out.expanduser()

    print(f"Converting {kitti_root} -> {out_root}\n")
    n_images, n_boxes, counts = convert(kitti_root, out_root)
    write_yaml(out_root, args.yaml.expanduser())

    print(f"  images converted : {n_images}")
    print(f"  boxes written    : {n_boxes}")
    print("  class breakdown  :")
    for name in sorted(counts):
        marker = "  (dropped)" if name in DROPPED or name.startswith("unmapped:") else ""
        print(f"      {name:<22} {counts[name]:>6}{marker}")
    print(f"\n  dataset yaml     : {args.yaml.expanduser().resolve()}")

    if n_images != 1500:
        print(f"\nWARNING: expected 1500 images, converted {n_images}")
        return 1
    print("\nDone. Export RAMS_KITTI_YAML to this yaml before re-running the accuracy phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
