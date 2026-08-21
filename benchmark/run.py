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
import hashlib
import json
import logging
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.controller import RAMSController
from rams.energy import (PowerProfile, estimate_profile_energy,
                         estimate_tdp_energy, load_power_profile)
from rams.policy import CANONICAL_POLICY_NAMES, POLICIES
from benchmark.load_injector import ProcessLoadInjector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VRU_CLASSES = {"person", "pedestrian", "cyclist", "bicycle", "motorbike", "motorcycle", "rider"}
ENERGY_SENSITIVITY_SCALES = {"low": 0.75, "nominal": 1.0, "high": 1.25}
DEFAULT_BURST_PEAK_INTENSITY = 0.85
DEFAULT_BURST_ON_FRAMES = 20
DEFAULT_BURST_OFF_FRAMES = 80
DEFAULT_BURST_SETTLE_S = 0.20


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


def build_replay_manifest(frame_paths: list[Path], seed: int, run_id: str) -> dict:
    """Freeze the exact replay inputs used by a paper-facing benchmark."""
    return {
        "run_id": run_id,
        "seed": seed,
        "n_frames": len(frame_paths),
        "frames": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in frame_paths
        ],
        "warmup_s": 0.5,
        "timing": "controller end-to-end; CUDA synchronized by model wrapper when available",
    }


def read_frame(path: Path):
    """Read an image from disk as a numpy array (BGR, uint8)."""
    import cv2
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not read image: {path}")
    return frame


# ---------------------------------------------------------------------------
# Load injectors
# ---------------------------------------------------------------------------

# Backward-compatible imports for experiments and tests.  Both protocols use
# the same isolated process implementation; their record labels differ.
LoadInjector = ProcessLoadInjector
BurstLoadInjector = ProcessLoadInjector


def burst_target_intensity(frame_index: int, peak: float, on_frames: int, off_frames: int) -> float:
    """Return the requested on/off burst target for one replay frame."""
    period = on_frames + off_frames
    if on_frames < 1 or off_frames < 1 or period < 2:
        raise ValueError("burst on/off frame counts must both be at least one")
    return peak if frame_index % period < on_frames else 0.0


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
    tdp_watts: float | None = None,
    tdp_label: str = "user-supplied TDP",
    energy_profile: PowerProfile | None = None,
    block: int | None = None,
    burst_peak_intensity: float = DEFAULT_BURST_PEAK_INTENSITY,
    burst_on_frames: int = DEFAULT_BURST_ON_FRAMES,
    burst_off_frames: int = DEFAULT_BURST_OFF_FRAMES,
    burst_settle_s: float = DEFAULT_BURST_SETTLE_S,
    load_reserve_logical_cpus: int = 1,
) -> list[dict]:
    is_burst = profile_label == "burst"
    initial_target = (burst_target_intensity(0, burst_peak_intensity, burst_on_frames, burst_off_frames)
                      if is_burst else load_intensity)
    injector = (BurstLoadInjector(intensity=initial_target,
                                   reserve_logical_cpus=load_reserve_logical_cpus)
                if is_burst else LoadInjector(
                    intensity=initial_target,
                    reserve_logical_cpus=load_reserve_logical_cpus,
                ))
    injector_label = "process_isolated_burst_v2" if is_burst else "process_steady_v3"

    records = []
    frame_idx = 0

    if not is_burst:
        # Steady process load begins before the controller and remains
        # unchanged for the full replay block.
        injector.start()
    try:
        with RAMSController(simulate=simulate, policy=policy_name) as ctrl:
            # Burst workers begin after model loading because their target
            # changes within a replay; steady workers are already active.
            if is_burst:
                injector.start()
            time.sleep(burst_settle_s if is_burst else 1.0)
            active_target = initial_target

            for i in range(n_inferences):
                target_intensity = (burst_target_intensity(i, burst_peak_intensity,
                                                           burst_on_frames, burst_off_frames)
                                    if is_burst else load_intensity)
                if target_intensity != active_target:
                    injector.set_intensity(target_intensity)
                    active_target = target_intensity
                    time.sleep(burst_settle_s)

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
                    "block":          block,
                    "policy":         policy_name,
                    "load_profile":   profile_label,
                    "load_intensity": target_intensity,
                    "load_injector":  injector_label,
                    "load_worker_processes": injector.worker_count,
                    "burst_cycle_frame": (i % (burst_on_frames + burst_off_frames)
                                           if is_burst else None),
                    "tier":           result["tier"],
                    "latency_ms":     round(result["latency_ms"], 3),
                    "pressure":       result.get("pressure"),
                    "cpu_pct":        result.get("cpu_pct"),
                    "mem_pct":        result.get("mem_pct"),
                    "cpu_temp":       result.get("cpu_temp"),
                    "gpu_util_pct":   result.get("gpu_util_pct"),
                    "gpu_mem_frac":   result.get("gpu_mem_frac"),
                    "gpu_temp":       result.get("gpu_temp"),
                    "gpu_clock_mhz":  result.get("gpu_clock_mhz"),
                    "accelerator_source": result.get("accelerator_source", "none"),
                    "policy_ms":      round(result.get("policy_ms", 0.0), 3),
                    "inference_ms":   round(result.get("inference_ms", result["latency_ms"]), 3),
                    "preprocess_ms":  round(result.get("preprocess_ms", 0.0), 3),
                    "postprocess_ms": round(result.get("postprocess_ms", 0.0), 3),
                    "end_to_end_ms":  round(result.get("end_to_end_ms", result["latency_ms"]), 3),
                    "backend":        result.get("backend", "unknown"),
                    "execution_providers": ";".join(result.get("execution_providers", [])),
                    "coreml_provider_options": json.dumps(result.get("coreml_provider_options", {}), sort_keys=True),
                    "simulated":      result.get("simulated", True),
                    "n_detections":   len(detections),
                    "vru_detected":   vru_detected,
                    "frame":          frame_name or "null",
                }
                estimate = estimate_tdp_energy(result["end_to_end_ms"], tdp_watts, tdp_label)
                record["tdp_energy_estimate_j"] = round(estimate.joules, 6) if estimate else None
                record["tdp_watts"] = estimate.watts if estimate else None
                record["tdp_label"] = estimate.label if estimate else ""
                if energy_profile is not None:
                    estimates = {
                        label: estimate_profile_energy(
                            result["end_to_end_ms"], energy_profile,
                            cpu_percent=result.get("cpu_pct"),
                            memory_percent=result.get("mem_pct"),
                            gpu_utilization_percent=result.get("gpu_util_pct"),
                            dynamic_scale=scale,
                        )
                        for label, scale in ENERGY_SENSITIVITY_SCALES.items()
                    }
                    nominal = estimates["nominal"]
                    record.update({
                        "energy_profile_estimate_low_j": round(estimates["low"].estimated_energy_j, 6),
                        "energy_profile_estimate_j": round(nominal.estimated_energy_j, 6),
                        "energy_profile_estimate_high_j": round(estimates["high"].estimated_energy_j, 6),
                        "energy_profile_power_w": round(nominal.estimated_power_w, 4),
                        "energy_profile_name": nominal.profile_name,
                        "energy_profile_source": nominal.profile_source,
                        "energy_profile_is_measured": False,
                    })
                else:
                    record.update({
                        "energy_profile_estimate_low_j": None,
                        "energy_profile_estimate_j": None,
                        "energy_profile_estimate_high_j": None,
                        "energy_profile_power_w": None,
                        "energy_profile_name": "",
                        "energy_profile_source": "",
                        "energy_profile_is_measured": False,
                    })
                records.append(record)

                if (i + 1) % 20 == 0:
                    logger.info(
                        "  [%s / %s] %d/%d  tier=%-6s  latency=%.1f ms  "
                        "R=%.3f  target=%.2f  dets=%d  vru=%s  backend=%s",
                        policy_name, profile_label, i + 1, n_inferences,
                        result["tier"], result["latency_ms"],
                        result.get("pressure", 0), target_intensity,
                        len(detections), vru_detected,
                        result.get("backend", "?"),
                    )

    finally:
        injector.stop()

    return records


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(all_records: list[dict], run_id: str, manifest: dict | None = None):
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

    if manifest is not None:
        manifest_path = RESULTS_DIR / f"{run_id}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Replay manifest -> %s", manifest_path)

    return summary


def bootstrap_ci(values: list[float], seed: int = 0, n_resamples: int = 1_000) -> tuple[float, float]:
    """Deterministic percentile-bootstrap interval for a mean."""
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values)
                   for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a binary rate."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


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
        vru_count    = sum(1 for r in recs if r.get("vru_detected"))
        vru_rate     = vru_count / len(recs)
        backends     = list({r.get("backend", "unknown") for r in recs})
        estimates = [r["tdp_energy_estimate_j"] for r in recs
                     if r.get("tdp_energy_estimate_j") is not None]
        profile_energies = [r["energy_profile_estimate_j"] for r in recs
                            if r.get("energy_profile_estimate_j") is not None]
        cpu_samples = [float(r["cpu_pct"]) for r in recs if r.get("cpu_pct") is not None]
        target_samples = [float(r.get("load_intensity", 0.0)) for r in recs]
        # Only the transient-burst profile has meaningful on/off windows.
        # For steady profiles these fields remain empty rather than presenting
        # every loaded frame as a burst-on observation.
        burst_on = ([r for r in recs if float(r.get("load_intensity", 0.0)) > 0.0]
                    if profile == "burst" else [])
        burst_off = ([r for r in recs if float(r.get("load_intensity", 0.0)) == 0.0]
                     if profile == "burst" else [])
        block_values: dict[int, list[float]] = {}
        for record in recs:
            block = record.get("block")
            if block is not None:
                block_values.setdefault(int(block), []).append(float(record["latency_ms"]))
        block_means = [sum(values) / len(values) for _, values in sorted(block_values.items())]
        # In the paper protocol, a replay block is the independent unit.  Old
        # ad-hoc runs without multiple blocks retain their frame-level CI.
        ci_values = block_means if len(block_means) >= 2 else latencies
        latency_ci = bootstrap_ci(ci_values, seed=hash(key) & 0xFFFFFFFF)
        vru_ci = wilson_interval(vru_count, len(recs))

        summary["groups"].append({
            "policy":        policy,
            "load_profile":  profile,
            "n":             len(recs),
            "latency_mean":  round(statistics.mean(latencies), 2),
            "latency_std":   round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
            "latency_p50":   round(statistics.median(latencies), 2),
            "latency_p95":   round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "latency_ci95":  [round(latency_ci[0], 2), round(latency_ci[1], 2)],
            "ci_unit":       "block_mean" if len(block_means) >= 2 else "frame",
            "n_blocks":      len(block_means),
            "tier_counts":   tier_counts,
            "vru_rate":      round(vru_rate, 4),
            "vru_rate_ci95": [round(vru_ci[0], 4), round(vru_ci[1], 4)],
            "backends":      backends,
            "load_injector": recs[0].get("load_injector", "legacy"),
            "load_target_intensity_mean": round(statistics.mean(target_samples), 4),
            "cpu_pct_mean": round(statistics.mean(cpu_samples), 2) if cpu_samples else None,
            "burst_on_frames": len(burst_on),
            "burst_off_frames": len(burst_off),
            "burst_on_latency_mean": (round(statistics.mean([r["latency_ms"] for r in burst_on]), 2)
                                      if burst_on else None),
            "burst_off_latency_mean": (round(statistics.mean([r["latency_ms"] for r in burst_off]), 2)
                                       if burst_off else None),
            "tdp_energy_estimate_mean_j": (round(statistics.mean(estimates), 6)
                                           if estimates else None),
            "energy_profile_estimate_mean_j": (round(statistics.mean(profile_energies), 6)
                                               if profile_energies else None),
        })

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAMS Benchmark Harness")
    parser.add_argument("--n",           type=int,  default=100,
                        help="Frames per policy/profile/block")
    parser.add_argument("--policy",      type=str,  default="all", help="all | threshold | predictive | safety | adaptive | safety2")
    parser.add_argument("--profile",     type=str,  default="all", help="all | idle | light | moderate | heavy | burst")
    parser.add_argument("--frames",      type=str,  default=None,  help="Path to directory of jpg/png images for real inference")
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.add_argument("--paper-mode", action="store_true",
                        help="require real replay frames and write a hashed manifest")
    parser.add_argument("--seed", type=int, default=20260710,
                        help="deterministic replay order seed")
    parser.add_argument("--blocks", type=int, default=1,
                        help="Independent paired replay blocks; randomizes policy order in every block")
    parser.add_argument("--tdp-watts", type=float, default=None,
                        help="fixed power budget for P×t estimate; not a physical measurement")
    parser.add_argument("--tdp-label", type=str, default="user-supplied TDP",
                        help="reported source/power-mode label for --tdp-watts")
    parser.add_argument("--energy-profile", type=str, default=None,
                        help="JSON telemetry-conditioned TDP profile; see configs/energy_profile.example.json")
    parser.add_argument("--burst-peak-intensity", type=float, default=DEFAULT_BURST_PEAK_INTENSITY,
                        help="host-level target used during burst-on windows; default 0.85")
    parser.add_argument("--burst-on-frames", type=int, default=DEFAULT_BURST_ON_FRAMES,
                        help="frames per burst-on window; default 20")
    parser.add_argument("--burst-off-frames", type=int, default=DEFAULT_BURST_OFF_FRAMES,
                        help="frames per burst-recovery window; default 80")
    parser.add_argument("--burst-settle-s", type=float, default=DEFAULT_BURST_SETTLE_S,
                        help="monitor-settle delay after every burst transition; default 0.20")
    parser.add_argument("--load-reserve-logical-cpus", type=int, default=1,
                        help="logical CPUs reserved from the process load generator; default 1")
    args = parser.parse_args()

    # Load frame paths if provided
    frame_paths = None
    if args.frames:
        frame_paths = load_frame_paths(args.frames)
        random.Random(args.seed).shuffle(frame_paths)
        logger.info("Real inference mode: %d frames available", len(frame_paths))
    elif not args.simulate:
        logger.warning("--no-simulate set but no --frames provided; using blank frames")
    if args.blocks < 1:
        parser.error("--blocks must be at least 1")
    if not 0.0 < args.burst_peak_intensity <= 1.0:
        parser.error("--burst-peak-intensity must be in (0, 1]")
    if args.burst_on_frames < 1 or args.burst_off_frames < 1:
        parser.error("--burst-on-frames and --burst-off-frames must both be at least 1")
    if args.burst_settle_s < 0:
        parser.error("--burst-settle-s must be non-negative")
    if args.paper_mode and (args.simulate or not frame_paths):
        parser.error("--paper-mode requires --no-simulate and a non-empty --frames directory")
    power_profile = load_power_profile(args.energy_profile) if args.energy_profile else None

    policies = list(CANONICAL_POLICY_NAMES) if args.policy == "all" else [args.policy]
    profiles = (
        list(LOAD_PROFILES.items()) if args.profile == "all"
        else [(args.profile, LOAD_PROFILES[args.profile])]
    )

    run_id = f"rams_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
    logger.info("=== RAMS Benchmark  run_id=%s ===", run_id)
    logger.info("Policies: %s  |  Profiles: %s  |  N/block=%d  |  blocks=%d  |  simulate=%s  |  frames=%s",
                policies, [p for p, _ in profiles], args.n, args.blocks, args.simulate,
                args.frames or "None (blank frames)")

    all_records = []

    for block in range(args.blocks):
        block_frames = frame_paths
        if frame_paths:
            offset = (block * args.n) % len(frame_paths)
            block_frames = frame_paths[offset:] + frame_paths[:offset]
        for profile_index, (profile_label, intensity) in enumerate(profiles):
            policy_order = list(policies)
            if args.blocks > 1:
                random.Random(args.seed + block * 1009 + profile_index).shuffle(policy_order)
            for policy_name in policy_order:
                logger.info("\n--- Block %d/%d  Policy: %s  Load: %s (%.0f%%) ---",
                            block + 1, args.blocks, policy_name, profile_label, intensity * 100)
                records = run_policy(
                    policy_name=policy_name,
                    n_inferences=args.n,
                    load_intensity=intensity,
                    simulate=args.simulate,
                    profile_label=profile_label,
                    frame_paths=block_frames,
                    tdp_watts=args.tdp_watts,
                    tdp_label=args.tdp_label,
                    energy_profile=power_profile,
                    block=block,
                    burst_peak_intensity=args.burst_peak_intensity,
                    burst_on_frames=args.burst_on_frames,
                    burst_off_frames=args.burst_off_frames,
                    burst_settle_s=args.burst_settle_s,
                    load_reserve_logical_cpus=args.load_reserve_logical_cpus,
                )
                all_records.extend(records)

    manifest = build_replay_manifest(frame_paths, args.seed, run_id) if args.paper_mode else None
    if manifest is not None:
        manifest["blocks"] = args.blocks
        manifest["frames_per_method_profile_block"] = args.n
        manifest["policy_order_randomized_per_block"] = args.blocks > 1
        manifest["load_protocol"] = {
            "steady_profiles": {
                "injector": "process_steady_v3",
                "profiles": ["idle", "light", "moderate", "heavy"],
            },
            "burst_profile": {
                "injector": "process_isolated_burst_v2",
                "reserved_logical_cpus": args.load_reserve_logical_cpus,
                "burst_peak_intensity": args.burst_peak_intensity,
                "burst_on_frames": args.burst_on_frames,
                "burst_off_frames": args.burst_off_frames,
                "burst_settle_s": args.burst_settle_s,
            },
        }
    summary = save_results(all_records, run_id, manifest=manifest)

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
