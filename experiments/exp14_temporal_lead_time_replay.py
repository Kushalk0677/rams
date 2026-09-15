"""KITTI-raw temporal lead-time replay for the i7-1165G7 ONNX route.

This experiment asks one narrow question: before a ground-truth VRU first
appears, does two-level VRU retention already select a stronger tier than the
threshold policy under the *same fixed calibrated pressure trace*?  It is not
a safety, prediction, recall, or energy experiment.

The controller is replayed sequentially from the beginning of each raw drive.
The retention clock is driven by the camera timestamps, so its 0.5-s hold has
the same temporal meaning as the five-frame look-back window.  Ground truth is
used only after inference to select and score pre-registered windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2

# Support the documented direct invocation
# ``python experiments/exp14_temporal_lead_time_replay.py`` as well as module
# invocation.  Python otherwise adds only ``experiments/`` to sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.exp13_temporal_lead_time import _timestamp_seconds
from rams.controller import RAMSController
from rams.models import Tier
from rams.policy import SafetyTwoLevelPolicy, ThresholdPolicy


@dataclass
class ReplayClock:
    """Mutable camera-time clock supplied to the retention policy."""

    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def _tier_name(value: object) -> str:
    return value.name if isinstance(value, Tier) else str(value)


def _load_calibration(path: Path) -> tuple[float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    proposal = payload.get("proposal", {})
    profiles = payload.get("profiles", {})
    try:
        lo = float(proposal["lo_thresh"])
        hi = float(proposal["hi_thresh"])
        pressure = float(profiles["heavy"]["pressure"]["p50"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Calibration record is missing i7 threshold/pressure fields: {path}") from exc
    return lo, hi, pressure


def _unique_events(events: Iterable[dict]) -> list[dict]:
    """Collapse same-frame multi-VRU entries into one dependent event cluster."""
    grouped: dict[tuple[str, int], dict] = {}
    for event in events:
        key = (str(event["sequence"]), int(event["entry_frame"]))
        previous = grouped.get(key)
        if previous is None:
            item = dict(event)
            item["object_types"] = [str(item.pop("object_type", "unknown"))]
            grouped[key] = item
        else:
            previous["object_types"].append(str(event.get("object_type", "unknown")))
    return sorted(grouped.values(), key=lambda item: (item["sequence"], item["entry_frame"]))


def pair_events_to_controls(events: list[dict], candidates: list[dict]) -> list[dict]:
    """Choose one deterministic, unused same-sequence control per entry event."""
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_sequence[str(candidate["sequence"])].append(candidate)
    for sequence in by_sequence:
        by_sequence[sequence].sort(key=lambda item: int(item["control_frame"]))

    used: set[tuple[str, int]] = set()
    pairs: list[dict] = []
    for event in events:
        sequence = str(event["sequence"])
        entry = int(event["entry_frame"])
        window = int(event["window_frames"])
        eligible = [
            candidate for candidate in by_sequence[sequence]
            if (sequence, int(candidate["control_frame"])) not in used
            and abs(int(candidate["control_frame"]) - entry) >= window
        ]
        if not eligible:
            raise ValueError(f"No unused same-sequence control available for {sequence} frame {entry}")
        # Stable rotation prevents early controls from being overused across events.
        chosen = eligible[len(pairs) % len(eligible)]
        used.add((sequence, int(chosen["control_frame"])))
        pairs.append({"event": event, "control": chosen})
    return pairs


def _load_sequence_frames(raw_root: Path, sequence: str) -> tuple[list[Path], list[float]]:
    date = sequence[:10]
    drive = raw_root / date / sequence / "image_02"
    paths = sorted((drive / "data").glob("*.png"))
    timestamps = [_timestamp_seconds(line) for line in (drive / "timestamps.txt").read_text(encoding="utf-8").splitlines()]
    if not paths or len(paths) != len(timestamps) or any(value is None for value in timestamps):
        raise ValueError(f"Invalid KITTI raw left-camera data for {sequence}")
    return paths, [float(value) for value in timestamps]


def _targets_for_pairs(pairs: list[dict]) -> dict[str, set[int]]:
    targets: dict[str, set[int]] = defaultdict(set)
    for pair in pairs:
        for kind in ("event", "control"):
            item = pair[kind]
            targets[str(item["sequence"])].update(int(frame) for frame in item["pre_frames"])
    return targets


def _max_required_frame(pairs: list[dict], sequence: str) -> int:
    frames = []
    for pair in pairs:
        for kind in ("event", "control"):
            item = pair[kind]
            if item["sequence"] == sequence:
                frames.extend(int(frame) for frame in item["pre_frames"])
    return max(frames)


def replay_policy(
    name: str,
    controller: RAMSController,
    clock: ReplayClock,
    raw_root: Path,
    pairs: list[dict],
    pressure: float,
    smoke: bool,
) -> dict[tuple[str, int], dict]:
    targets = _targets_for_pairs(pairs)
    records: dict[tuple[str, int], dict] = {}
    controller.start()
    try:
        for sequence in sorted(targets):
            paths, timestamps = _load_sequence_frames(raw_root, sequence)
            upper = _max_required_frame(pairs, sequence)
            if upper >= len(paths):
                raise ValueError(f"Target frame {upper} exceeds {sequence} length {len(paths)}")
            controller.policy.reset()
            controller._current_tier = Tier.SMALL
            for index in range(upper + 1):
                frame = cv2.imread(str(paths[index]))
                if frame is None:
                    raise ValueError(f"Unreadable frame: {paths[index]}")
                clock.value = timestamps[index]
                controller.set_pressure_override(pressure)
                result = controller.infer(frame)
                key = (sequence, index)
                if key in {(sequence, frame_index) for frame_index in targets[sequence]}:
                    records[key] = {
                        "sequence": sequence,
                        "frame": index,
                        "timestamp": timestamps[index],
                        "frame_sha256": hashlib.sha256(paths[index].read_bytes()).hexdigest(),
                        "policy": name,
                        "tier": _tier_name(result["tier"]),
                        "pressure": result["pressure"],
                        "backend": result.get("backend", "unknown"),
                        "n_detections": len(result.get("detections", [])),
                    }
            if smoke:
                break
    finally:
        controller.stop()
    return records


def _bootstrap_difference(pairs: list[dict], seed: int, samples: int = 10_000) -> tuple[float, float]:
    if len(pairs) < 2:
        return (float("nan"), float("nan"))
    randomizer = random.Random(seed)
    differences = []
    for _ in range(samples):
        selected = [pairs[randomizer.randrange(len(pairs))] for _ in pairs]
        differences.append(sum(item["event_rate"] - item["control_rate"] for item in selected) / len(selected))
    differences.sort()
    return differences[int(0.025 * samples)], differences[int(0.975 * samples) - 1]


def score_pairs(pairs: list[dict], threshold: dict[tuple[str, int], dict], safety2: dict[tuple[str, int], dict], seed: int) -> tuple[list[dict], dict]:
    scored: list[dict] = []
    all_frame_records: list[dict] = []
    for pair_index, pair in enumerate(pairs):
        output = {"pair_id": pair_index}
        for kind in ("event", "control"):
            item = pair[kind]
            sequence = str(item["sequence"])
            frames = [int(frame) for frame in item["pre_frames"]]
            elevated = []
            for frame in frames:
                baseline = threshold[(sequence, frame)]
                retained = safety2[(sequence, frame)]
                is_elevated = Tier[retained["tier"]] > Tier[baseline["tier"]]
                elevated.append(is_elevated)
                all_frame_records.append({
                    "pair_id": pair_index, "window": kind, "sequence": sequence, "frame": frame,
                    "threshold_tier": baseline["tier"], "retention_tier": retained["tier"],
                    "elevated": is_elevated, "pressure": retained["pressure"],
                    "frame_sha256": retained["frame_sha256"], "backend": retained["backend"],
                })
            output[f"{kind}_sequence"] = sequence
            output[f"{kind}_anchor_frame"] = int(item.get("entry_frame", item.get("control_frame")))
            output[f"{kind}_rate"] = sum(elevated) / len(elevated)
        scored.append(output)
    event_frames = [row for row in all_frame_records if row["window"] == "event"]
    control_frames = [row for row in all_frame_records if row["window"] == "control"]
    ci_low, ci_high = _bootstrap_difference(scored, seed)
    summary = {
        "n_pairs": len(scored),
        "n_event_pre_frames": len(event_frames),
        "n_control_pre_frames": len(control_frames),
        "pre_entry_elevation_rate": sum(row["elevated"] for row in event_frames) / len(event_frames),
        "control_elevation_rate": sum(row["elevated"] for row in control_frames) / len(control_frames),
        "paired_rate_difference": sum(row["event_rate"] - row["control_rate"] for row in scored) / len(scored),
        "paired_block_bootstrap_ci95": [ci_low, ci_high],
    }
    return all_frame_records, {"pairs": scored, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="KITTI-raw temporal lead-time controller replay")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pressure", type=float, default=None,
                        help="Fixed replay pressure. Defaults to the saved heavy-profile median.")
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--smoke", action="store_true", help="Run one paired event/control window only.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["transition_manifest"]
    if manifest["status"] != "ready":
        raise ValueError("Transition manifest is not ready")
    events = _unique_events(manifest["eligible_transition_events"])
    pairs = pair_events_to_controls(events, manifest["matched_control_candidates"])
    if args.smoke:
        pairs = pairs[:1]
    lo, hi, calibrated_pressure = _load_calibration(args.calibration)
    pressure = calibrated_pressure if args.pressure is None else args.pressure
    clock = ReplayClock()
    common = {"lo_thresh": lo, "hi_thresh": hi, "hysteresis_window": 3}
    threshold_controller = RAMSController(simulate=False, policy=ThresholdPolicy(**common))
    safety2_controller = RAMSController(
        simulate=False,
        policy=SafetyTwoLevelPolicy(**common, proximity_window_s=args.window_s,
                                    min_conf=args.min_conf, near_area_fraction=0.02, clock=clock),
    )
    threshold = replay_policy("threshold", threshold_controller, clock, args.raw_root, pairs, pressure, args.smoke)
    safety2 = replay_policy("safety2", safety2_controller, clock, args.raw_root, pairs, pressure, args.smoke)
    frames, scored = score_pairs(pairs, threshold, safety2, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"exp14_temporal_lead_time_{stamp}.csv"
    json_path = args.output_dir / f"exp14_temporal_lead_time_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames)
    payload = {
        "analysis": "KITTI raw pre-entry elevation; reactive carry-over only, not VRU anticipation",
        "smoke": args.smoke,
        "raw_root": str(args.raw_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "calibration": str(args.calibration.resolve()),
        "thresholds": {"lo": lo, "hi": hi, "hysteresis_window": 3},
        "fixed_pressure": pressure,
        "pressure_source": "saved heavy-profile median from the specified calibration" if args.pressure is None else "CLI override",
        "window_seconds": args.window_s,
        "min_conf": args.min_conf,
        "unique_entry_events_available": len(events),
        **scored,
        "records_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Records: {csv_path}\nSummary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
