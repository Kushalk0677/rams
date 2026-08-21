"""Real KITTI retention analysis for confidence sensitivity and sustained VRUs.

Ground truth is used only after inference to score the controller.  It is not
fed into policy selection, so this experiment does not create an oracle policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.run import LOAD_PROFILES, LoadInjector, VRU_CLASSES
from experiments.exp8_accuracy_per_tier import parse_kitti_label
from rams.controller import RAMSController
from rams.policy import PAPER_POLICY_LABELS, make_policy

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_pairs(images: Path, labels: Path, limit: int | None) -> list[tuple[Path, Path]]:
    pairs = [(image, labels / f"{image.stem}.txt") for image in sorted(images.glob("*.png"))]
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No KITTI PNG images found")
    return pairs


def longest_true_streak(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def retention_lock_active(policy) -> bool:
    """Return whether a VRU-retention policy currently has an active lock.

    The retention policies store the last qualifying detection rather than an
    absolute ``_lock_until`` timestamp.  Query their policy-specific active
    state so the experiment measures the deployed controller contract.
    """
    active = getattr(policy, "_vru_active", None)
    if callable(active):
        return bool(active())

    lock_tier = getattr(policy, "_vru_lock_tier", None)
    if callable(lock_tier):
        return lock_tier() is not None

    return False


def evaluate(policy_name: str, min_conf: float, profile: str,
             pairs: list[tuple[Path, Path]]) -> list[dict]:
    policy = make_policy(policy_name, min_conf=min_conf)
    injector = LoadInjector(LOAD_PROFILES[profile])
    records: list[dict] = []
    gt_streak = 0
    try:
        with RAMSController(simulate=False, policy=policy) as controller:
            # Load begins only after all model tiers are constructed.
            injector.start()
            time.sleep(1.0)
            for index, (image_path, label_path) in enumerate(pairs):
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                gt = parse_kitti_label(label_path)
                gt_vru = any(item["class"] in VRU_CLASSES for item in gt)
                gt_streak = gt_streak + 1 if gt_vru else 0
                lock_before = retention_lock_active(policy)
                result = controller.infer(frame)
                detected_vru = any(str(item.get("class", "")).lower() in VRU_CLASSES
                                   for item in result.get("detections", []))
                lock_after = retention_lock_active(policy)
                records.append({
                    "image": image_path.name,
                    "frame_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "policy": policy_name,
                    "policy_label": PAPER_POLICY_LABELS.get(policy_name, policy_name),
                    "min_conf": min_conf,
                    "load_profile": profile,
                    "gt_vru": gt_vru,
                    "detected_vru": detected_vru,
                    "gt_vru_streak": gt_streak,
                    "lock_before": lock_before,
                    "lock_after": lock_after,
                    "tier": result["tier"],
                    "latency_ms": round(float(result["latency_ms"]), 3),
                    "pressure": result.get("pressure"),
                    "load_injector": "process_steady_v3",
                    "load_worker_processes": injector.worker_count,
                    "backend": result.get("backend", "unknown"),
                })
                if (index + 1) % 100 == 0:
                    print(f"[{policy_name}, conf={min_conf:.2f}] {index + 1}/{len(pairs)}")
    finally:
        injector.stop()
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["policy"], record["min_conf"], record["load_profile"])].append(record)
    output = []
    for (policy, min_conf, profile), group in sorted(groups.items()):
        gt = [record for record in group if record["gt_vru"]]
        detected_given_gt = sum(record["detected_vru"] for record in gt)
        lock_given_detection = [record for record in group if record["detected_vru"]]
        retained_given_gt = sum(record["tier"] != "NANO" for record in gt)
        output.append({
            "policy": policy,
            "policy_label": PAPER_POLICY_LABELS.get(policy, policy),
            "min_conf": min_conf,
            "load_profile": profile,
            "n_frames": len(group),
            "n_gt_vru_frames": len(gt),
            "detected_vru_given_gt": detected_given_gt,
            "detector_trigger_recall": round(detected_given_gt / len(gt), 4) if gt else None,
            "retained_non_nano_given_gt": round(retained_given_gt / len(gt), 4) if gt else None,
            "lock_after_detection_rate": round(sum(record["lock_after"] for record in lock_given_detection) /
                                               len(lock_given_detection), 4) if lock_given_detection else None,
            "max_gt_vru_streak": max((record["gt_vru_streak"] for record in group), default=0),
            "max_lock_streak": longest_true_streak([record["lock_after"] for record in group]),
            "tier_counts": {tier: sum(record["tier"] == tier for record in group)
                            for tier in ("NANO", "SMALL", "MEDIUM")},
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="VRU retention sensitivity on real KITTI replay")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--policies", default="safety,safety2")
    parser.add_argument("--min-confs", default="0.25,0.40")
    parser.add_argument("--profile", choices=sorted(LOAD_PROFILES), default="moderate")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    pairs = load_pairs(Path(args.images), Path(args.labels), args.max_images)
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    min_confs = [float(value.strip()) for value in args.min_confs.split(",") if value.strip()]
    records = [record for policy in policies for min_conf in min_confs
               for record in evaluate(policy, min_conf, args.profile, pairs)]
    summary = summarize(records)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS / f"exp12_retention_sensitivity_{stamp}.csv"
    json_path = RESULTS / f"exp12_retention_sensitivity_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps({"groups": summary, "n_frames": len(pairs)}, indent=2), encoding="utf-8")
    print(f"Raw records: {csv_path}\nSummary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
