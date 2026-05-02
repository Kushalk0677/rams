"""
RAMS Threshold Calibrator
=========================
Measures your machine's real idle baseline pressure R(t) and computes
policy thresholds that make sense for your hardware.

Problem:
    The default thresholds (lo=0.45, hi=0.72) were tuned for simulation.
    On real hardware — especially a laptop under Windows with background
    tasks, memory pressure, and ONNX inference consuming CPU — idle
    pressure can sit at 0.5-0.7, making the system oscillate between tiers
    constantly even under "light" load.

Solution:
    Measure R(t) at true idle (nothing running), then set thresholds
    relative to that baseline so MEDIUM is the default tier at idle,
    SMALL kicks in at moderate load, and NANO only at heavy load.

Usage:
    python scripts/calibrate.py                  # measure + print
    python scripts/calibrate.py --apply          # measure + write configs/default.yaml
    python scripts/calibrate.py --seconds 30     # longer measurement (more accurate)

Outputs:
    - Prints calibrated thresholds to console
    - With --apply: updates configs/default.yaml automatically
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.monitor import ResourceMonitor


# ---------------------------------------------------------------------------
# Calibration logic
# ---------------------------------------------------------------------------

def measure_baseline(seconds: int = 20) -> dict:
    """
    Sample R(t) for `seconds` seconds at true idle.
    Returns statistics dict.
    """
    print(f"\n{'='*60}")
    print("  RAMS Threshold Calibrator")
    print(f"{'='*60}")
    print(f"\n  Measuring baseline pressure for {seconds}s ...")
    print("  Keep the system as idle as possible (close other apps).")
    print("  Starting in 3 seconds ...\n")
    time.sleep(3)

    samples = []
    cpu_samples = []
    mem_samples = []

    with ResourceMonitor(hz=10.0) as monitor:
        time.sleep(0.5)  # warm up
        t_end = time.monotonic() + seconds
        i = 0
        while time.monotonic() < t_end:
            snap = monitor.snapshot
            if snap:
                samples.append(snap.pressure_index)
                cpu_samples.append(snap.cpu_percent)
                mem_samples.append(snap.memory_percent)
                if i % 20 == 0:
                    elapsed = seconds - (t_end - time.monotonic())
                    print(f"  [{elapsed:4.0f}s / {seconds}s]  "
                          f"R={snap.pressure_index:.3f}  "
                          f"CPU={snap.cpu_percent:.1f}%  "
                          f"MEM={snap.memory_percent:.1f}%  "
                          f"temp={snap.cpu_temp or 'N/A'}")
            time.sleep(0.1)
            i += 1

    if not samples:
        print("ERROR: No samples collected.")
        sys.exit(1)

    return {
        "n":          len(samples),
        "mean":       statistics.mean(samples),
        "std":        statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p50":        statistics.median(samples),
        "p75":        sorted(samples)[int(len(samples) * 0.75)],
        "p90":        sorted(samples)[int(len(samples) * 0.90)],
        "p95":        sorted(samples)[int(len(samples) * 0.95)],
        "max":        max(samples),
        "cpu_mean":   statistics.mean(cpu_samples),
        "mem_mean":   statistics.mean(mem_samples),
    }


def compute_thresholds(baseline: dict) -> dict:
    """
    Derive policy thresholds from baseline measurement.

    Strategy:
        - lo_thresh = baseline p90 + small margin
          (anything below this = system is idle/lightly loaded → MEDIUM)
        - hi_thresh = lo_thresh + headroom for moderate load
          (anything above this = system is heavily loaded → NANO)
        - Both clipped to [0.30, 0.95] to stay sane

    The gap between lo and hi (the SMALL band) is proportional to the
    baseline std — more volatile machines get a wider SMALL band.
    """
    # Idle ceiling: p90 of baseline + 5% margin
    idle_ceil = baseline["p90"] + 0.05

    # lo_thresh: just above idle ceiling (entering SMALL from MEDIUM)
    lo = round(min(0.75, max(0.30, idle_ceil)), 3)

    # hi_thresh: lo + gap. Gap widens with baseline volatility (std)
    gap = max(0.10, min(0.25, baseline["std"] * 4 + 0.12))
    hi  = round(min(0.95, lo + gap), 3)

    return {
        "lo_thresh": lo,
        "hi_thresh": hi,
        "gap":       round(hi - lo, 3),
    }


def print_report(baseline: dict, thresholds: dict):
    print(f"\n{'='*60}")
    print("  Baseline Measurement Results")
    print(f"{'='*60}")
    print(f"  Samples:      {baseline['n']}")
    print(f"  Mean R(t):    {baseline['mean']:.3f}")
    print(f"  Std R(t):     {baseline['std']:.3f}")
    print(f"  p50:          {baseline['p50']:.3f}")
    print(f"  p75:          {baseline['p75']:.3f}")
    print(f"  p90:          {baseline['p90']:.3f}")
    print(f"  p95:          {baseline['p95']:.3f}")
    print(f"  max:          {baseline['max']:.3f}")
    print(f"  CPU mean:     {baseline['cpu_mean']:.1f}%")
    print(f"  MEM mean:     {baseline['mem_mean']:.1f}%")

    print(f"\n{'='*60}")
    print("  Calibrated Thresholds")
    print(f"{'='*60}")
    print(f"  lo_thresh:    {thresholds['lo_thresh']}  "
          f"(R below this → MEDIUM tier)")
    print(f"  hi_thresh:    {thresholds['hi_thresh']}  "
          f"(R above this → NANO tier)")
    print(f"  SMALL band:   [{thresholds['lo_thresh']}, "
          f"{thresholds['hi_thresh']}]  "
          f"(gap = {thresholds['gap']})")
    print()
    print("  Previous defaults:  lo=0.45  hi=0.72")
    print(f"  New values:         lo={thresholds['lo_thresh']}  "
          f"hi={thresholds['hi_thresh']}")

    # Sanity warnings
    if baseline["mean"] > 0.60:
        print("\n  ⚠  HIGH BASELINE: Your machine has significant background")
        print("     load. Close browsers, cloud sync, antivirus scans etc.")
        print("     and re-run calibration for best results.")
    if thresholds["gap"] < 0.10:
        print("\n  ⚠  NARROW SMALL BAND: May cause frequent tier switching.")
        print("     Consider increasing hysteresis_window to 5.")


def apply_to_config(thresholds: dict, config_path: Path):
    """
    Write calibrated thresholds into configs/default.yaml.
    Updates lo_thresh and hi_thresh for threshold, predictive,
    safety, and safety2 policies.
    """
    if not config_path.exists():
        print(f"\nERROR: Config not found at {config_path}")
        sys.exit(1)

    text = config_path.read_text()

    lo = thresholds["lo_thresh"]
    hi = thresholds["hi_thresh"]
    # predictive thresholds sit slightly lower (EWMA smooths spikes)
    lo_pred = round(max(0.25, lo - 0.05), 3)
    hi_pred = round(max(lo_pred + 0.10, hi - 0.05), 3)

    import re

    def replace_thresh(block_name: str, src: str,
                       new_lo: float, new_hi: float) -> str:
        # Replace lo_thresh and hi_thresh within a named policy block
        # Works by finding the block and replacing the next two thresh lines
        pattern = (
            rf"(  {block_name}:.*?lo_thresh:\s*)\S+"
            rf"(.*?hi_thresh:\s*)\S+"
        )
        repl = rf"\g<1>{new_lo}\g<2>{new_hi}"
        return re.sub(pattern, repl, src, flags=re.DOTALL, count=1)

    text = replace_thresh("threshold",  text, lo,      hi)
    text = replace_thresh("predictive", text, lo_pred, hi_pred)
    text = replace_thresh("safety",     text, lo,      hi)

    # Also handle safety2 if present (same pattern as safety)
    text = replace_thresh("safety2",    text, lo,      hi)

    config_path.write_text(text)
    print(f"\n  ✓  Config updated: {config_path}")
    print(f"     threshold:  lo={lo}  hi={hi}")
    print(f"     predictive: lo={lo_pred}  hi={hi_pred}")
    print(f"     safety:     lo={lo}  hi={hi}")


def save_calibration_json(baseline: dict, thresholds: dict, out: Path):
    """Save calibration results for reproducibility / paper appendix."""
    import json, socket, platform
    data = {
        "host":       socket.gethostname(),
        "platform":   platform.platform(),
        "baseline":   baseline,
        "thresholds": thresholds,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.write_text(json.dumps(data, indent=2))
    print(f"  ✓  Calibration saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAMS Threshold Calibrator — measures idle baseline and "
                    "computes hardware-appropriate switching thresholds."
    )
    parser.add_argument(
        "--seconds", type=int, default=20,
        help="How long to sample idle baseline (default: 20s, recommend 30s+)"
    )
    parser.add_argument(
        "--apply", action="store_true", default=False,
        help="Write calibrated thresholds to configs/default.yaml"
    )
    args = parser.parse_args()

    # Measure
    baseline = measure_baseline(seconds=args.seconds)

    # Compute thresholds
    thresholds = compute_thresholds(baseline)

    # Report
    print_report(baseline, thresholds)

    # Save JSON always (for paper reproducibility)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    save_calibration_json(
        baseline, thresholds,
        results_dir / f"calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Apply to config if requested
    if args.apply:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        apply_to_config(thresholds, config_path)
        print(f"\n  Re-run the benchmark now with calibrated thresholds:")
        print(f"  python -m benchmark.run --n 200 --policy all --no-simulate --frames D:\\data\\val2017")
    else:
        print(f"\n  To apply these thresholds automatically, run:")
        print(f"  python scripts/calibrate.py --apply")

    print()


if __name__ == "__main__":
    main()
