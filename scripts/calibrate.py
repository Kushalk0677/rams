"""Calibrate RAMS thresholds with real replay and measured host pressure.

This tool deliberately does not tune against accuracy.  It probes the same
real inference path and steady-load injector used by Phase 2, holding the
controller at SMALL so the measured pressure is not an artefact of one
candidate policy.  It then proposes one common base threshold pair for every
policy.  The proposal requires separation between the stable upper light-load
band and the stable lower heavy-load band; transient high-pressure light
samples remain meaningful inputs to the controller and are handled by policy
hysteresis rather than discarded during calibration.

Examples
--------
Dry run, preserving the current configuration::

    .\\.venv\\Scripts\\python.exe scripts\\calibrate.py --frames D:\\data\\kitti\\images\\val

Apply only after inspecting the saved report::

    .\\.venv\\Scripts\\python.exe scripts\\calibrate.py --frames D:\\data\\kitti\\images\\val --apply
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.run import LOAD_PROFILES, LoadInjector, load_frame_paths, read_frame
from rams.controller import RAMSController
from rams.models import Tier
from rams.policy import FixedTierPolicy


# Allow the cross-core worker processes to reach their requested duty cycle
# before recording telemetry.  This is deliberately shared with the runtime
# protocol's one-second steady-load settling interval.
STEADY_SETTLE_S = 1.0


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile from no values")
    index = (len(ordered) - 1) * quantile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "p10": round(percentile(values, 0.10), 4),
        "p25": round(percentile(values, 0.25), 4),
        "p50": round(percentile(values, 0.50), 4),
        "p75": round(percentile(values, 0.75), 4),
        "p90": round(percentile(values, 0.90), 4),
    }


def default_frames() -> Path:
    configured = os.environ.get("RAMS_DATA_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend((Path("D:/data"), ROOT.parent / "data", ROOT / "data"))
    for root in candidates:
        frames = root / "kitti" / "images" / "val"
        if frames.is_dir():
            return frames
    return candidates[0] / "kitti" / "images" / "val" if candidates else ROOT / "data" / "kitti" / "images" / "val"


def probe_profile(
    profile: str,
    frame_paths: list[Path],
    n_frames: int,
    warmup_frames: int,
) -> dict:
    """Measure pressure during real fixed-SMALL replay under one load profile."""
    intensity = LOAD_PROFILES[profile]
    pressures: list[float] = []
    cpu_values: list[float] = []
    latencies: list[float] = []
    print(f"\n[{profile}] fixed-SMALL real replay: {n_frames} measured frames at target {intensity:.0%}")

    with LoadInjector(intensity):
        with RAMSController(simulate=False, policy=FixedTierPolicy(Tier.SMALL)) as controller:
            time.sleep(STEADY_SETTLE_S)
            for index in range(warmup_frames + n_frames):
                frame = read_frame(frame_paths[index % len(frame_paths)])
                result = controller.infer(frame)
                if index < warmup_frames:
                    continue
                pressures.append(float(result["pressure"]))
                cpu_values.append(float(result.get("cpu_pct", 0.0)))
                latencies.append(float(result["end_to_end_ms"]))
                if (index - warmup_frames + 1) % 25 == 0:
                    print(f"  {index - warmup_frames + 1:>3}/{n_frames}: R={pressures[-1]:.3f}, "
                          f"CPU={cpu_values[-1]:.1f}%, latency={latencies[-1]:.1f} ms")

    return {
        "profile": profile,
        "target_intensity": intensity,
        "pressure": describe(pressures),
        "cpu_percent": describe(cpu_values),
        "end_to_end_ms": describe(latencies),
    }


def propose_thresholds(profiles: dict[str, dict], minimum_separation: float) -> dict:
    """Return a common base pair or a rejection reason.

Idle/light operation should normally occupy SMALL, while sustained heavy
pressure should enter NANO.  The steady workload uses independent processes
and a duty cycle scaled to a whole-host target.  Calibration nevertheless
uses observed telemetry, not the requested target: it separates the light and
heavy medians, records tail overlap explicitly, and relies on the existing
three-sample hysteresis to reject transient crossings.  The resulting pair is
determined entirely from pressure, before any accuracy result is examined.
    """
    normal = [profiles[name]["pressure"] for name in ("idle", "light") if name in profiles]
    if not normal or "heavy" not in profiles:
        return {"valid": False, "reason": "idle, light, and heavy probes are required"}

    normal_floor = min(float(item["p10"]) for item in normal)
    light_median = float(profiles["light"]["pressure"]["p50"])
    heavy_median = float(profiles["heavy"]["pressure"]["p50"])
    light_upper = float(profiles["light"]["pressure"]["p75"])
    heavy_floor = float(profiles["heavy"]["pressure"]["p10"])
    separation = heavy_median - light_median
    tail_overlap = light_upper - heavy_floor
    if separation < minimum_separation:
        return {
            "valid": False,
            "reason": (
                "heavy pressure does not separate from the central light band: "
                f"heavy p50={heavy_median:.3f}, light p50={light_median:.3f}, "
                f"gap={separation:.3f} < required {minimum_separation:.3f}"
            ),
            "light_p50": round(light_median, 3),
            "heavy_p50": round(heavy_median, 3),
            "light_p75": round(light_upper, 3),
            "heavy_p10": round(heavy_floor, 3),
            "separation": round(separation, 3),
            "tail_overlap": round(tail_overlap, 3),
        }

    # Keeping lo below the normal p10 puts at least 90% of normal replay in
    # SMALL, absent hysteresis.  hi bisects the central light/heavy boundary.
    lo = max(0.05, min(0.90, normal_floor - 0.01))
    hi = max(lo + 0.08, min(0.95, (light_median + heavy_median) / 2.0))
    if hi >= 0.95 and heavy_median <= light_median:
        return {"valid": False, "reason": "no usable NANO threshold within the pressure range"}
    return {
        "valid": True,
        "lo_thresh": round(lo, 3),
        "hi_thresh": round(hi, 3),
        "normal_p10": round(normal_floor, 3),
        "light_p50": round(light_median, 3),
        "heavy_p50": round(heavy_median, 3),
        "light_p75": round(light_upper, 3),
        "heavy_p10": round(heavy_floor, 3),
        "separation": round(separation, 3),
        "tail_overlap": round(tail_overlap, 3),
        "target_behavior": "idle/light SMALL; heavy may enter NANO; VRU policies may retain a stronger tier after detection",
    }


def apply_thresholds(config_path: Path, lo: float, hi: float, snapshot_path: Path) -> None:
    """Update every policy's shared base thresholds while preserving YAML comments."""
    shutil.copy2(config_path, snapshot_path)
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    required = {"threshold", "predictive", "adaptive", "safety", "safety2"}
    found: set[str] = set()
    active: str | None = None
    output: list[str] = []
    for line in lines:
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            active = line.strip()[:-1]
            found.add(active)
        if active in required and line.lstrip().startswith("lo_thresh:"):
            output.append(f"    lo_thresh: {lo}\n")
        elif active in required and line.lstrip().startswith("hi_thresh:"):
            output.append(f"    hi_thresh: {hi}\n")
        else:
            output.append(line)

    if "adaptive" not in found:
        insertion = next((index for index, line in enumerate(output) if line.startswith("  safety:")), None)
        if insertion is None:
            raise RuntimeError("could not add adaptive policy thresholds: safety block not found")
        output[insertion:insertion] = [
            "  adaptive:\n",
            f"    lo_thresh: {lo}\n",
            f"    hi_thresh: {hi}\n",
            "\n",
        ]
    missing = {"threshold", "predictive", "safety", "safety2"} - found
    if missing:
        raise RuntimeError(f"could not update required policy blocks: {sorted(missing)}")
    config_path.write_text("".join(output), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate comparable RAMS thresholds from real fixed-SMALL replay")
    parser.add_argument("--frames", type=Path, default=None, help="KITTI image directory; defaults to RAMS_DATA_ROOT or D:/data")
    parser.add_argument("--profiles", default="idle,light,moderate,heavy", help="comma-separated steady profiles")
    parser.add_argument("--n", type=int, default=100, help="measured real frames per profile")
    parser.add_argument("--warmup", type=int, default=10, help="unrecorded warm-up frames per profile")
    parser.add_argument("--minimum-separation", type=float, default=0.03,
                        help="minimum heavy-p50 minus light-p50 pressure gap needed to apply")
    parser.add_argument("--apply", action="store_true", help="write the validated proposal to configs/default.yaml")
    args = parser.parse_args()

    frames_dir = (args.frames or default_frames()).expanduser()
    if args.n < 25 or args.warmup < 0:
        parser.error("--n must be at least 25 and --warmup must be non-negative")
    names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = [name for name in names if name not in LOAD_PROFILES or name == "burst"]
    if unknown:
        parser.error(f"unknown or unsupported steady profile(s): {unknown}")
    if not {"idle", "light", "heavy"}.issubset(names):
        parser.error("--profiles must include idle, light, and heavy")
    frame_paths = load_frame_paths(str(frames_dir))
    print("RAMS replay calibration")
    print(f"Frames: {frames_dir} ({len(frame_paths)} available), backend: {os.environ.get('RAMS_BACKEND', 'onnx')}")
    print("The active YAML is unchanged unless --apply is supplied.")

    profiles = {
        name: probe_profile(name, frame_paths, args.n, args.warmup)
        for name in names
    }
    proposal = propose_thresholds(profiles, args.minimum_separation)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / f"calibration_{stamp}_replay.json"
    report = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "backend": os.environ.get("RAMS_BACKEND", "onnx"),
        "frames": str(frames_dir),
        "n_per_profile": args.n,
        "warmup_frames": args.warmup,
        "steady_settle_s": STEADY_SETTLE_S,
        "load_injector": "process_steady_v3",
        "profiles": profiles,
        "proposal": proposal,
        "applied": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if proposal.get("valid") and args.apply:
        config_path = ROOT / "configs" / "default.yaml"
        snapshots = results_dir / "calibration_snapshots"
        snapshots.mkdir(exist_ok=True)
        snapshot_path = snapshots / f"default_before_replay_calibration_{stamp}.yaml"
        apply_thresholds(config_path, float(proposal["lo_thresh"]), float(proposal["hi_thresh"]), snapshot_path)
        report["applied"] = True
        report["config_path"] = str(config_path)
        report["config_snapshot_before"] = str(snapshot_path)
        print(f"\nApplied common thresholds: lo={proposal['lo_thresh']}, hi={proposal['hi_thresh']}")
        print(f"Previous configuration saved to: {snapshot_path}")
    elif args.apply:
        print(f"\nNOT APPLIED: {proposal['reason']}")
    else:
        print("\nDry run only. Re-run with --apply only if the proposal is valid.")

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration report: {report_path}")
    if not proposal.get("valid"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
