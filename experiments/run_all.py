"""
experiments/run_all.py
Runs all RAMS experiments in sequence.

Usage
-----
    # Full run (simulation, default N)
    python experiments/run_all.py --simulate

    # Quick smoke-test
    python experiments/run_all.py --simulate --n 20

    # Real YOLOv8 inference (requires ultralytics + model weights)
    python experiments/run_all.py --no-simulate --n 200
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _header(title: str):
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}\n")


def main():
    parser = argparse.ArgumentParser(description="Run all RAMS experiments")
    parser.add_argument("--n",           type=int,  default=60,
                        help="Inferences per cell (default 60; use 20 for a quick test)")
    parser.add_argument("--simulate",    action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.add_argument("--skip",        nargs="*", default=[],
                        help="Experiment numbers to skip, e.g. --skip 4 6")
    args = parser.parse_args()

    skip = {int(s) for s in args.skip}
    t0   = time.monotonic()

    if 1 not in skip:
        _header("Experiment 1 — Policy Comparison")
        from experiments.exp1_policy_comparison import run as exp1
        exp1(n=args.n, simulate=args.simulate)

    if 2 not in skip:
        _header("Experiment 2 — Load Profile Sweep")
        from experiments.exp2_load_sweep import run as exp2
        exp2(n=args.n, simulate=args.simulate)

    if 3 not in skip:
        _header("Experiment 3 — Hysteresis Sensitivity")
        from experiments.exp3_hysteresis import run as exp3
        exp3(n=args.n, simulate=args.simulate)

    if 4 not in skip:
        _header("Experiment 4 — Safety Override Analysis")
        from experiments.exp4_safety_override import run as exp4
        exp4(n=args.n, simulate=args.simulate)

    if 5 not in skip:
        _header("Experiment 5 — Accuracy–Latency Pareto Frontier (NEW)")
        from experiments.exp5_pareto import run as exp5
        exp5(n=args.n, simulate=args.simulate)

    if 6 not in skip:
        _header("Experiment 6 — Transient Spike Response (NEW)")
        from experiments.exp6_transient import run as exp6
        exp6(simulate=args.simulate)

    if 7 not in skip:
        _header("Experiment 7 — Safety-Weighted Accuracy Score SWAS (NEW)")
        from experiments.exp7_swas import run as exp7
        exp7(n=args.n, simulate=args.simulate)

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 64}")
    print(f"  All experiments complete in {elapsed:.1f} s")
    print(f"  Results written to: results/")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
