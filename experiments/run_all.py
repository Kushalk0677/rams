"""
experiments/run_all.py
Runs all RAMS experiments in sequence.

Usage
-----
    # Simulation-only experiments (no real models needed)
    python experiments/run_all.py --simulate

    # Quick smoke-test
    python experiments/run_all.py --simulate --n 20

    # Real YOLOv8 inference
    python experiments/run_all.py --no-simulate --n 200 --frames /data/kitti/images/val

    # Skip hardware-dependent experiments (8, 9, 10) for a quick run
    python experiments/run_all.py --simulate --skip 8 9 10

    # Run only new ESL experiments
    python experiments/run_all.py --no-simulate --only 8 10 \\
        --frames /data/kitti/images/val \\
        --kitti-images /data/kitti/images/val \\
        --kitti-labels /data/kitti/labels/val

Experiment index
----------------
  1  Policy comparison (latency × policy × load)
  2  Load profile sweep (fine-grained intensity)
  3  Hysteresis sensitivity ablation
  4  Safety override analysis (VRU rate, proximity, confidence)
  5  Accuracy-latency Pareto (simulated accuracy proxy)
  6  Transient spike response
  7  Safety-Weighted Accuracy Score (SWAS)
  --- ESL additions ---
  8  Per-tier mAP and VRU recall on real KITTI/COCO data  [requires --kitti-images]
  9  Multi-device benchmark (run separately per device, then aggregate)
  10 Safety-latency Pareto with real VRU recall           [requires exp8 + benchmark JSON]
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
    parser.add_argument("--n",            type=int, default=60,
                        help="Inferences per cell (default 60; use 20 for a quick test)")
    parser.add_argument("--simulate",     action="store_true", default=True)
    parser.add_argument("--no-simulate",  dest="simulate", action="store_false")
    parser.add_argument("--frames",       type=str, default=None,
                        help="Image directory for real inference (exps 1-7, 9)")
    parser.add_argument("--skip",         nargs="*", default=[],
                        help="Experiment numbers to skip e.g. --skip 8 9 10")
    parser.add_argument("--only",         nargs="*", default=[],
                        help="Run only these experiment numbers e.g. --only 8 10")

    # Exp 8 args
    parser.add_argument("--kitti-images", type=str, default=None,
                        help="KITTI val image directory (exp 8)")
    parser.add_argument("--kitti-labels", type=str, default=None,
                        help="KITTI val label directory (exp 8)")
    parser.add_argument("--coco-images",  type=str, default=None,
                        help="COCO val image directory (exp 8, optional)")
    parser.add_argument("--coco-labels",  type=str, default=None,
                        help="COCO val label directory (exp 8, optional)")
    parser.add_argument("--max-images",   type=int, default=None,
                        help="Cap images for exp 8 (useful for smoke tests)")

    # Exp 10 args
    parser.add_argument("--exp8-json",      type=str, default=None,
                        help="exp8_accuracy_kitti.json path (exp 10)")
    parser.add_argument("--benchmark-json", type=str, default=None,
                        help="Benchmark summary JSON path (exp 10)")
    parser.add_argument("--sweep-proximity", action="store_true",
                        help="Run proximity window sweep in exp 10")

    args = parser.parse_args()

    skip = {int(s) for s in args.skip}
    only = {int(s) for s in args.only} if args.only else None

    def should_run(exp_num: int) -> bool:
        if only is not None:
            return exp_num in only
        return exp_num not in skip

    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Experiments 1-7: existing, simulation-compatible
    # ------------------------------------------------------------------

    if should_run(1):
        _header("Experiment 1 — Policy Comparison")
        from experiments.exp1_policy_comparison import run as exp1
        exp1(n=args.n, simulate=args.simulate)

    if should_run(2):
        _header("Experiment 2 — Load Profile Sweep")
        from experiments.exp2_load_sweep import run as exp2
        exp2(n=args.n, simulate=args.simulate)

    if should_run(3):
        _header("Experiment 3 — Hysteresis Sensitivity")
        from experiments.exp3_hysteresis import run as exp3
        exp3(n=args.n, simulate=args.simulate)

    if should_run(4):
        _header("Experiment 4 — Safety Override Analysis")
        from experiments.exp4_safety_override import run as exp4
        exp4(n=args.n, simulate=args.simulate)

    if should_run(5):
        _header("Experiment 5 — Accuracy–Latency Pareto (simulated proxy)")
        from experiments.exp5_pareto import run as exp5
        exp5(n=args.n, simulate=args.simulate)

    if should_run(6):
        _header("Experiment 6 — Transient Spike Response")
        from experiments.exp6_transient import run as exp6
        exp6(simulate=args.simulate)

    if should_run(7):
        _header("Experiment 7 — Safety-Weighted Accuracy Score (SWAS)")
        from experiments.exp7_swas import run as exp7
        exp7(n=args.n, simulate=args.simulate)

    # ------------------------------------------------------------------
    # Experiment 8: per-tier mAP and VRU recall (requires real data)
    # ------------------------------------------------------------------

    if should_run(8):
        _header("Experiment 8 — Per-Tier mAP and VRU Recall [ESL]")
        if not args.kitti_images or not args.kitti_labels:
            print("  SKIPPED: --kitti-images and --kitti-labels required for exp 8.")
            print("  Example:")
            print("    --kitti-images /data/kitti/images/val")
            print("    --kitti-labels /data/kitti/labels/val")
        else:
            import subprocess
            cmd = [
                sys.executable, "experiments/exp8_accuracy_per_tier.py",
                "--dataset", "kitti",
                "--images",  args.kitti_images,
                "--labels",  args.kitti_labels,
            ]
            if args.max_images:
                cmd += ["--max-images", str(args.max_images)]
            if args.coco_images and args.coco_labels:
                cmd += ["--also-coco",
                        "--coco-images", args.coco_images,
                        "--coco-labels", args.coco_labels]
            subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # Experiment 9: multi-device (print instructions, can't auto-run)
    # ------------------------------------------------------------------

    if should_run(9):
        _header("Experiment 9 — Multi-Device Benchmark [ESL]")
        print("  Exp 9 must be run independently on each device.")
        print("  On each device, run:")
        print()
        print("    # Jetson Orin (TensorRT)")
        print("    python experiments/exp9_multidevice.py \\")
        print("        --device jetson-orin --backend tensorrt \\")
        print(f"        --frames {args.frames or '/data/kitti/images/val'} --n {args.n}")
        print()
        print("    # RTX 6000")
        print("    python experiments/exp9_multidevice.py \\")
        print("        --device rtx6000 --backend onnx \\")
        print(f"        --frames {args.frames or '/data/kitti/images/val'} --n {args.n}")
        print()
        print("  Then aggregate:")
        print("    python experiments/exp9_aggregate.py")

    # ------------------------------------------------------------------
    # Experiment 10: safety-latency Pareto with real recall
    # ------------------------------------------------------------------

    if should_run(10):
        _header("Experiment 10 — Safety-Latency Pareto (Real VRU Recall) [ESL]")
        if not args.exp8_json or not args.benchmark_json:
            print("  SKIPPED: --exp8-json and --benchmark-json required for exp 10.")
            print("  Run exp 8 first, then:")
            print("    --exp8-json    results/exp8_accuracy_kitti.json")
            print("    --benchmark-json results/rams_<timestamp>_kitti_calibrated.json")
        else:
            import subprocess
            cmd = [
                sys.executable, "experiments/exp10_safety_pareto.py",
                "--exp8-json",      args.exp8_json,
                "--benchmark-json", args.benchmark_json,
                "--dataset",        "kitti",
            ]
            if args.sweep_proximity and args.frames:
                cmd += ["--sweep-proximity", "--frames", args.frames, "--n", str(args.n)]
            subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 64}")
    print(f"  All experiments complete in {elapsed:.1f} s")
    print(f"  Results written to: results/")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
