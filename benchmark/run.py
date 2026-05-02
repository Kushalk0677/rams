"""
RAMS Benchmark Harness
Runs each switching policy for N inference calls under a synthetic
load profile, then writes per-run results to results/.

Usage:
    python -m benchmark.run --n 200 --policy all --simulate
    python -m benchmark.run --n 100 --policy safety --no-simulate
    python -m benchmark.run --n 200 --policy all --no-simulate --frames D:/data/val2017
    python -m benchmark.run --n 200 --policy all --no-simulate --frames D:/data/data_object_image_2/training/image_2
"""

import argparse
import csv
import json
import logging
import os
import platform
import random
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.controller import RAMSController
from rams.policy import POLICIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VRU_CLASSES = {"person", "pedestrian", "cyclist", "bicycle", "motorbike", "motorcycle", "rider"}


# ---------------------------------------------------------------------------
# Frame loader — cycles through real images if --frames provided
# ---------------------------------------------------------------------------

def load_frame_paths(frames_dir: str) -> list[Path]:
    """Return sorted list of jpg/png paths from frames_dir."""
    p = Path(frames_dir)
    if not p.exists():
        raise FileNotFoundError(f"--frames directory not found: {frames_dir}")
    paths = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")))
    if not paths:
        raise ValueError(f"No jpg/png images found in {frames_dir}")
    logger.info("Loaded %d frames from %s", len(paths), frames_dir)
    return paths


def read_frame(path: Path):
    """Read an image from disk as a numpy array (BGR, uint8)."""
    import cv2
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not read image: {path}")
    return frame


# ---------------------------------------------------------------------------
# Synthetic load injector (burns CPU on background threads)
# ---------------------------------------------------------------------------

class LoadInjector:
    """Spawns worker threads to simulate concurrent system load."""

    def __init__(self, intensity: float = 0.0):
        """intensity in [0, 1]. 0 = idle, 1 = full saturation."""
        self.intensity = intensity
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _burn(self):
        while not self._stop.is_set():
            _ = sum(i * i for i in range(1000))
            time.sleep(max(0.0, (1.0 - self.intensity) * 0.001))

    def start(self):
        n = max(0, int(self.intensity * 4))
        for _ in range(n):
            t = threading.Thread(target=self._burn, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Load profiles
# ---------------------------------------------------------------------------

LOAD_PROFILES = {
    "idle":     0.00,
    "light":    0.25,
    "moderate": 0.50,
    "heavy":    0.75,
    "burst":    1.00,
}


# ---------------------------------------------------------------------------
# Single policy run
# ---------------------------------------------------------------------------

def run_policy(
    policy_name: str,
    n_inferences: int,
    load_intensity: float,
    simulate: bool,
    profile_label: str,
    frame_paths: list = None,
) -> list[dict]:

    injector = LoadInjector(intensity=load_intensity)
    injector.start()

    records = []
    frame_idx = 0

    try:
        with RAMSController(simulate=simulate, policy=policy_name) as ctrl:
            time.sleep(0.5)  # let monitor warm up

            for i in range(n_inferences):

                # Pick next frame (cycle through if fewer frames than n)
                frame = None
                frame_name = None
                if frame_paths:
                    path = frame_paths[frame_idx % len(frame_paths)]
                    frame_idx += 1
                    frame_name = path.name
                    try:
                        frame = read_frame(path)
                    except IOError as e:
                        logger.warning("Skipping frame: %s", e)
                        frame = None

                result = ctrl.infer(frame=frame)

                # Detect VRU in real detections
                detections = result.get("detections", [])
                vru_detected = any(
                    str(d.get("class", "")).lower() in VRU_CLASSES
                    for d in detections
                )

                record = {
                    "run_idx":        i,
                    "policy":         policy_name,
                    "load_profile":   profile_label,
                    "load_intensity": load_intensity,
                    "tier":           result["tier"],
                    "latency_ms":     round(result["latency_ms"], 3),
                    "pressure":       result.get("pressure"),
                    "cpu_pct":        result.get("cpu_pct"),
                    "mem_pct":        result.get("mem_pct"),
                    "cpu_temp":       result.get("cpu_temp"),
                    "backend":        result.get("backend", "unknown"),
                    "simulated":      result.get("simulated", True),
                    "n_detections":   len(detections),
                    "vru_detected":   vru_detected,
                    "frame":          frame_name or "null",
                }
                records.append(record)

                if (i + 1) % 20 == 0:
                    logger.info(
                        "  [%s / %s] %d/%d  tier=%-6s  latency=%.1f ms  "
                        "R=%.3f  dets=%d  vru=%s  backend=%s",
                        policy_name, profile_label, i + 1, n_inferences,
                        result["tier"], result["latency_ms"],
                        result.get("pressure", 0),
                        len(detections), vru_detected,
                        result.get("backend", "?"),
                    )

    finally:
        injector.stop()

    return records


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(all_records: list[dict], run_id: str):
    csv_path  = RESULTS_DIR / f"{run_id}.csv"
    json_path = RESULTS_DIR / f"{run_id}.json"

    # CSV
    if all_records:
        keys = list(all_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_records)
        logger.info("Results -> %s", csv_path)

    # JSON summary
    summary = compute_summary(all_records)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", json_path)

    return summary


def compute_summary(records: list[dict]) -> dict:
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for r in records:
        key = (r["policy"], r["load_profile"])
        groups[key].append(r)

    summary: dict = {
        "host":     socket.gethostname(),
        "platform": platform.platform(),
        "python":   sys.version,
        "groups":   [],
    }

    for (policy, profile), recs in sorted(groups.items()):
        latencies    = [r["latency_ms"] for r in recs]
        tiers        = [r["tier"] for r in recs]
        tier_counts  = {t: tiers.count(t) for t in set(tiers)}
        vru_rate     = sum(1 for r in recs if r.get("vru_detected")) / len(recs)
        backends     = list({r.get("backend", "unknown") for r in recs})

        summary["groups"].append({
            "policy":        policy,
            "load_profile":  profile,
            "n":             len(recs),
            "latency_mean":  round(statistics.mean(latencies), 2),
            "latency_std":   round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
            "latency_p50":   round(statistics.median(latencies), 2),
            "latency_p95":   round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "tier_counts":   tier_counts,
            "vru_rate":      round(vru_rate, 4),
            "backends":      backends,
        })

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAMS Benchmark Harness")
    parser.add_argument("--n",           type=int,  default=100,   help="Inferences per policy-profile combo")
    parser.add_argument("--policy",      type=str,  default="all", help="all | threshold | predictive | safety | adaptive | safety2")
    parser.add_argument("--profile",     type=str,  default="all", help="all | idle | light | moderate | heavy | burst")
    parser.add_argument("--frames",      type=str,  default=None,  help="Path to directory of jpg/png images for real inference")
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    args = parser.parse_args()

    # Load frame paths if provided
    frame_paths = None
    if args.frames:
        frame_paths = load_frame_paths(args.frames)
        logger.info("Real inference mode: %d frames available", len(frame_paths))
    elif not args.simulate:
        logger.warning("--no-simulate set but no --frames provided; using blank frames")

    policies = list(POLICIES.keys()) if args.policy == "all" else [args.policy]
    profiles = (
        list(LOAD_PROFILES.items()) if args.profile == "all"
        else [(args.profile, LOAD_PROFILES[args.profile])]
    )

    run_id = f"rams_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
    logger.info("=== RAMS Benchmark  run_id=%s ===", run_id)
    logger.info("Policies: %s  |  Profiles: %s  |  N=%d  |  simulate=%s  |  frames=%s",
                policies, [p for p, _ in profiles], args.n, args.simulate,
                args.frames or "None (blank frames)")

    all_records = []

    for policy_name in policies:
        for profile_label, intensity in profiles:
            logger.info("\n--- Policy: %s  Load: %s (%.0f%%) ---",
                        policy_name, profile_label, intensity * 100)
            records = run_policy(
                policy_name=policy_name,
                n_inferences=args.n,
                load_intensity=intensity,
                simulate=args.simulate,
                profile_label=profile_label,
                frame_paths=frame_paths,
            )
            all_records.extend(records)

    summary = save_results(all_records, run_id)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Policy':<14} {'Profile':<10} {'N':>5}  {'Mean ms':>8}  {'P95 ms':>8}  {'VRU%':>6}  Tiers")
    print("-" * 80)
    for g in summary["groups"]:
        tier_str = ", ".join(f"{k}:{v}" for k, v in sorted(g["tier_counts"].items()))
        print(
            f"{g['policy']:<14} {g['load_profile']:<10} {g['n']:>5}  "
            f"{g['latency_mean']:>8.1f}  {g['latency_p95']:>8.1f}  "
            f"{g['vru_rate']*100:>5.1f}%  {tier_str}"
        )
    print("=" * 80)
    print(f"\nBackends used: {set(b for g in summary['groups'] for b in g['backends'])}")


if __name__ == "__main__":
    main()