"""
Complete experiment orchestrator for Windows-friendly RAMS runs.

This does not replace experiments/run_all.py. It adds a stricter, more
practical orchestrator that:
  - runs each experiment in a fresh subprocess,
  - records exactly what was attempted,
  - skips gracefully when datasets are missing,
  - can also drive benchmark / exp9 / exp10 in one pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def run_cmd(label: str, cmd: list[str], manifest: list[dict]) -> bool:
    print("\n" + "=" * 78)
    print(f"[{label}] {' '.join(cmd)}")
    print("=" * 78)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT)
    ok = proc.returncode == 0
    manifest.append({
        "label": label,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "ok": ok,
    })
    return ok


def latest_file(pattern: str) -> str | None:
    matches = sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else None


def maybe_add_simulate(cmd: list[str], simulate: bool):
    cmd.append("--simulate" if simulate else "--no-simulate")


def main():
    parser = argparse.ArgumentParser(description="Complete RAMS run-all orchestrator")
    parser.add_argument("--n", type=int, default=20, help="Per-cell inference count for most experiments")
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--no-simulate", dest="simulate", action="store_false")
    parser.add_argument("--transient-total-n", type=int, default=60,
                        help="Total inference count for exp6 transient run")
    parser.add_argument("--frames", type=str, default=None,
                        help="Real/smoke image directory for benchmark and exp9")
    parser.add_argument("--kitti-images", type=str, default=None)
    parser.add_argument("--kitti-labels", type=str, default=None)
    parser.add_argument("--coco-images", type=str, default=None)
    parser.add_argument("--coco-labels", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", type=str, default="windows-host")
    parser.add_argument("--backend", type=str, default="onnx")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-exp9", action="store_true")
    parser.add_argument("--skip-exp10", action="store_true")
    args = parser.parse_args()

    manifest: list[dict] = []
    py = sys.executable

    # Experiments 1-5
    exp_scripts = [
        ("exp1", [py, "experiments/exp1_policy_comparison.py", "--n", str(args.n)]),
        ("exp2", [py, "experiments/exp2_load_sweep.py", "--n", str(args.n)]),
        ("exp3", [py, "experiments/exp3_hysteresis.py", "--n", str(args.n)]),
        ("exp4", [py, "experiments/exp4_safety_override.py", "--n", str(args.n)]),
        ("exp5", [py, "experiments/exp5_pareto.py", "--n", str(args.n)]),
    ]
    for label, cmd in exp_scripts:
        maybe_add_simulate(cmd, args.simulate)
        run_cmd(label, cmd, manifest)

    # Exp 6 transient
    run_cmd(
        "exp6",
        [py, "experiments/exp6_transient.py", "--total-n", str(args.transient_total_n),
         "--simulate" if args.simulate else "--no-simulate"],
        manifest,
    )

    # Exp 7 can be long; still run, but with same n the user requested
    run_cmd(
        "exp7",
        [py, "experiments/exp7_swas.py", "--n", str(args.n),
         "--simulate" if args.simulate else "--no-simulate"],
        manifest,
    )

    # Benchmark summary used by exp10
    benchmark_json = None
    if not args.skip_benchmark:
        bench_cmd = [py, "-m", "benchmark.run", "--n", str(args.n), "--policy", "all", "--profile", "all"]
        maybe_add_simulate(bench_cmd, args.simulate)
        if args.frames:
            bench_cmd += ["--frames", args.frames]
        if run_cmd("benchmark", bench_cmd, manifest):
            benchmark_json = latest_file("rams_*.json")

    # Exp 8 if KITTI paths exist
    exp8_json = None
    exp8_coco_json = None
    if args.kitti_images and args.kitti_labels:
        exp8_cmd = [
            py, "experiments/exp8_accuracy_per_tier.py",
            "--dataset", "kitti",
            "--images", args.kitti_images,
            "--labels", args.kitti_labels,
        ]
        if args.max_images:
            exp8_cmd += ["--max-images", str(args.max_images)]
        if args.simulate:
            exp8_cmd += ["--simulate"]
        if args.coco_images and args.coco_labels:
            exp8_cmd += ["--also-coco", "--coco-images", args.coco_images, "--coco-labels", args.coco_labels]
        if run_cmd("exp8", exp8_cmd, manifest):
            exp8_json = str(RESULTS / "exp8_accuracy_kitti.json")
            if args.coco_images and args.coco_labels:
                exp8_coco_json = str(RESULTS / "exp8_accuracy_coco.json")
    else:
        manifest.append({"label": "exp8", "skipped": True, "reason": "kitti paths not provided"})

    # Exp 9 local device run + aggregate
    if not args.skip_exp9 and args.frames:
        e9_cmd = [
            py, "experiments/exp9_multidevice.py",
            "--device", args.device,
            "--backend", args.backend,
            "--frames", args.frames,
            "--n", str(args.n),
        ]
        if args.simulate:
            e9_cmd += ["--simulate"]
        run_cmd("exp9_device", e9_cmd, manifest)
        run_cmd("exp9_aggregate", [py, "experiments/exp9_aggregate.py", "--results-dir", "results/multidevice"], manifest)
    elif not args.skip_exp9:
        manifest.append({"label": "exp9", "skipped": True, "reason": "frames not provided"})

    # Exp 10 only if both ingredients exist
    if not args.skip_exp10:
        exp8_json = exp8_json or latest_file("exp8_accuracy_kitti.json")
        exp8_coco_json = exp8_coco_json or latest_file("exp8_accuracy_coco.json")
        benchmark_json = benchmark_json or latest_file("rams_*.json")
        if exp8_json and benchmark_json:
            e10_cmd = [
                py, "experiments/exp10_safety_pareto.py",
                "--exp8-json", exp8_json,
                "--benchmark-json", benchmark_json,
                "--dataset", "kitti",
            ]
            run_cmd("exp10_kitti", e10_cmd, manifest)
        else:
            manifest.append({"label": "exp10_kitti", "skipped": True, "reason": "exp8 and/or benchmark json missing"})

        if args.coco_images and args.coco_labels:
            if exp8_coco_json and benchmark_json:
                e10_coco_cmd = [
                    py, "experiments/exp10_safety_pareto.py",
                    "--exp8-json", exp8_coco_json,
                    "--benchmark-json", benchmark_json,
                    "--dataset", "coco",
                ]
                run_cmd("exp10_coco", e10_coco_cmd, manifest)
            else:
                manifest.append({"label": "exp10_coco", "skipped": True, "reason": "COCO exp8 and/or benchmark json missing"})
        else:
            manifest.append({"label": "exp10_coco", "skipped": True, "reason": "coco paths not provided"})

    manifest_path = RESULTS / "complete_runall_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nDone.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
