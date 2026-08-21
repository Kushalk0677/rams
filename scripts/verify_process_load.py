"""Measure a RAMS process-load target before starting a paper run.

This is a setup diagnostic only.  It does not write calibration or paper
results.  Calibration remains the acceptance gate for a device/backend.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.load_injector import ProcessLoadInjector


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure RAMS process-loader CPU usage")
    parser.add_argument("--intensity", type=float, default=0.75)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if not 0.0 <= args.intensity <= 1.0 or args.settle_s < 0 or args.samples < 1:
        parser.error("intensity must be in [0, 1], settle-s must be non-negative, and samples must be positive")

    injector = ProcessLoadInjector(args.intensity)
    try:
        injector.start()
        time.sleep(args.settle_s)
        psutil.cpu_percent(interval=None)
        samples = [psutil.cpu_percent(interval=1.0) for _ in range(args.samples)]
    finally:
        injector.stop()
    print(f"workers={injector.worker_count} logical_cpus={injector.logical_cpus} "
          f"target={args.intensity * 100:.1f}% median_cpu={statistics.median(samples):.1f}% "
          f"samples={','.join(f'{value:.1f}' for value in samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
