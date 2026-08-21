"""Policy-level KITTI accuracy evaluation for the revised RAMS letter.

Unlike the older per-tier experiment, this replays labelled KITTI frames through
``RAMSController.infer()``.  The resulting figures therefore reflect the
detector tier actually selected by each policy under the stated resource load.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.run import LOAD_PROFILES, LoadInjector, VRU_CLASSES, wilson_interval
from experiments.exp8_accuracy_per_tier import iou, parse_kitti_label
from rams.controller import RAMSController
from rams.policy import CANONICAL_POLICY_NAMES, PAPER_POLICY_LABELS

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
OBSTACLE_CLASSES = {"car", "van", "truck", "tram", "bus"}


def image_paths(images: Path, labels: Path, limit: int | None) -> list[tuple[Path, Path]]:
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    pairs = [(path, labels / f"{path.stem}.txt") for path in paths]
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No KITTI image files found")
    return pairs


def match_subset(gt: list[dict], pred: list[dict], classes: set[str]) -> tuple[int, int, int]:
    """Greedy IoU matching for a class subset (localization-level obstacle metric)."""
    gt_items = [item for item in gt if item["class"] in classes]
    pred_items = [item for item in pred if item["class"] in classes]
    matched: set[int] = set()
    tp = fp = 0
    for candidate in pred_items:
        best_iou, best_index = 0.0, -1
        for index, truth in enumerate(gt_items):
            if index in matched:
                continue
            score = iou(candidate["bbox"], truth["bbox"])
            if score > best_iou:
                best_iou, best_index = score, index
        if best_index >= 0 and best_iou >= 0.5:
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt_items) - len(matched)


def metric(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    recall_ci = wilson_interval(tp, tp + fn) if tp + fn else (0.0, 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "false_negative_rate": round(1.0 - recall, 4),
        "recall_ci95": [round(recall_ci[0], 4), round(recall_ci[1], 4)],
    }


def evaluate(policy: str, profile: str, pairs: list[tuple[Path, Path]], simulate: bool) -> list[dict]:
    injector = LoadInjector(LOAD_PROFILES[profile])
    records: list[dict] = []
    try:
        with RAMSController(simulate=simulate, policy=policy) as controller:
            # Keep all tiers warm before applying the measured load profile.
            injector.start()
            time.sleep(1.0)
            for index, (image_path, label_path) in enumerate(pairs):
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                gt = parse_kitti_label(label_path)
                result = controller.infer(frame)
                detections = []
                for det in result.get("detections", []):
                    bbox = det.get("xyxy") or det.get("bbox")
                    if bbox and len(bbox) >= 4:
                        detections.append({
                            "class": str(det.get("class", "")).lower(),
                            "bbox": [float(value) for value in bbox[:4]],
                        })
                v_tp, v_fp, v_fn = match_subset(gt, detections, VRU_CLASSES)
                o_tp, o_fp, o_fn = match_subset(gt, detections, OBSTACLE_CLASSES)
                records.append({
                    "image": image_path.name,
                    "policy": policy,
                    "policy_label": PAPER_POLICY_LABELS.get(policy, policy),
                    "load_profile": profile,
                    "tier": result["tier"],
                    "latency_ms": round(float(result["latency_ms"]), 3),
                    "pressure": result.get("pressure"),
                    "load_injector": "process_steady_v3",
                    "load_worker_processes": injector.worker_count,
                    "backend": result.get("backend", "unknown"),
                    "execution_providers": result.get("execution_providers", []),
                    "coreml_provider_options": result.get("coreml_provider_options", {}),
                    "vru_tp": v_tp, "vru_fp": v_fp, "vru_fn": v_fn,
                    "obstacle_tp": o_tp, "obstacle_fp": o_fp, "obstacle_fn": o_fn,
                    "frame_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                })
                if (index + 1) % 100 == 0:
                    print(f"[{policy}/{profile}] {index + 1}/{len(pairs)}")
    finally:
        injector.stop()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Policy-level KITTI VRU and obstacle accuracy")
    parser.add_argument("--images", required=True, help="KITTI validation image directory")
    parser.add_argument("--labels", required=True, help="KITTI validation label directory")
    parser.add_argument("--policies", default=",".join(CANONICAL_POLICY_NAMES))
    parser.add_argument("--profiles", default="moderate",
                        help="comma-separated load profiles; moderate is the paper accuracy condition")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--simulate", action="store_true", default=False,
                        help="diagnostic only; simulated output is not paper evidence")
    args = parser.parse_args()

    images, labels = Path(args.images), Path(args.labels)
    if not images.is_dir() or not labels.is_dir():
        parser.error("--images and --labels must be existing directories")
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = set(profiles) - set(LOAD_PROFILES)
    if unknown:
        parser.error(f"Unknown load profile(s): {', '.join(sorted(unknown))}")
    pairs = image_paths(images, labels, args.max_images)
    random.Random(args.seed).shuffle(pairs)

    records: list[dict] = []
    for profile in profiles:
        for policy in policies:
            records.extend(evaluate(policy, profile, pairs, args.simulate))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["policy"], record["load_profile"])].append(record)
    summary_groups = []
    for (policy, profile), group in sorted(grouped.items()):
        vru = metric(sum(r["vru_tp"] for r in group), sum(r["vru_fp"] for r in group), sum(r["vru_fn"] for r in group))
        obstacle = metric(sum(r["obstacle_tp"] for r in group), sum(r["obstacle_fp"] for r in group), sum(r["obstacle_fn"] for r in group))
        summary_groups.append({
            "policy": policy,
            "policy_label": PAPER_POLICY_LABELS.get(policy, policy),
            "load_profile": profile,
            "n_frames": len(group),
            "vru": vru,
            "non_vru_obstacle": obstacle,
            "tier_counts": {tier: sum(r["tier"] == tier for r in group) for tier in ("NANO", "SMALL", "MEDIUM")},
        })

    run_id = f"exp11_policy_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path, json_path, manifest_path = RESULTS / f"{run_id}.csv", RESULTS / f"{run_id}.json", RESULTS / f"{run_id}_manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps({"groups": summary_groups, "simulated": args.simulate}, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "seed": args.seed, "n_frames": len(pairs), "frames": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path, _ in pairs
        ], "policies": policies, "profiles": profiles, "simulated": args.simulate,
    }, indent=2), encoding="utf-8")
    print(f"Raw records: {csv_path}\nSummary: {json_path}\nReplay manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
