"""
Complete RAMS experiment orchestrator for NVIDIA Jetson.

Runs experiments 1 through 10 in isolated subprocesses, using the Jetson
TensorRT path by default. The script also performs the Jetson-specific setup
steps that matter for reproducibility: optional TensorRT export, calibration
with --apply, benchmark generation for exp10, exp9 multidevice output, and a
JSON manifest of every command that ran.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def count_files(path: str | None, exts: set[str] | None = None) -> int:
    if not path:
        return 0
    root = Path(path).expanduser()
    if not root.exists():
        return 0
    if exts is None:
        return sum(1 for p in root.iterdir() if p.is_file())
    return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def latest_file(pattern: str) -> str | None:
    matches = sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else None


def append_simulate(cmd: list[str], simulate: bool) -> None:
    cmd.append("--simulate" if simulate else "--no-simulate")


def run_cmd(label: str, cmd: list[str], manifest: list[dict], required: bool = True) -> bool:
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
        "required": required,
    })
    return ok


def preflight(args: argparse.Namespace, manifest: list[dict]) -> bool:
    frames_n = count_files(args.frames, IMAGE_EXTS)
    kitti_img_n = count_files(args.kitti_images, IMAGE_EXTS)
    kitti_lbl_n = count_files(args.kitti_labels, {".txt"})
    coco_img_n = count_files(args.coco_images, IMAGE_EXTS)
    coco_lbl_n = count_files(args.coco_labels, {".txt"})

    engines = {
        "NANO": any(Path(name).exists() for name in ("yolov8n.engine", "yolov8n_imgsz320.engine")),
        "SMALL": any(Path(name).exists() for name in ("yolov8s.engine", "yolov8s_imgsz416.engine")),
        "MEDIUM": any(Path(name).exists() for name in ("yolov8m.engine", "yolov8m_imgsz640.engine")),
    }
    checks = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "frames": {"path": args.frames, "count": frames_n},
        "kitti_images": {"path": args.kitti_images, "count": kitti_img_n},
        "kitti_labels": {"path": args.kitti_labels, "count": kitti_lbl_n},
        "coco_images": {"path": args.coco_images, "count": coco_img_n},
        "coco_labels": {"path": args.coco_labels, "count": coco_lbl_n},
        "tensorrt_engines_present": engines,
    }
    manifest.append({"label": "preflight", "ok": True, "checks": checks})

    print("\nJetson preflight")
    print("  Run these before starting if you have not already:")
    print("    sudo nvpmodel -m 0")
    print("    sudo jetson_clocks")
    print(f"  KITTI images: {kitti_img_n}  labels: {kitti_lbl_n}")
    print(f"  Frames for benchmark/exp9: {frames_n}")
    print(f"  TensorRT engines present: {engines}")

    required_ok = frames_n > 0 and kitti_img_n > 0 and kitti_lbl_n > 0
    if args.require_full_kitti and (kitti_img_n < 1500 or kitti_lbl_n < 1500):
        required_ok = False
        print("  Expected at least 1500 KITTI val images and labels.")
    if args.backend == "tensorrt" and not args.skip_export and all(engines.values()):
        print("  Existing TensorRT engines will be reused.")
    if args.backend == "tensorrt" and args.skip_export and not all(engines.values()):
        required_ok = False
        print("  Missing TensorRT engines and --skip-export was set.")
    if not required_ok:
        print("  Preflight failed. Fix dataset/engine paths or pass smoke-test settings.")
    return required_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson complete RAMS run-all, experiments 1-10")
    parser.add_argument("--n", type=int, default=200,
                        help="Per-cell inference count for most experiments")
    parser.add_argument("--transient-total-n", type=int, default=240,
                        help="Total inference count per policy for exp6")
    parser.add_argument("--frames", type=str, default="~/rams/data/kitti/images/val",
                        help="Image directory for benchmark and exp9")
    parser.add_argument("--kitti-images", type=str, default="~/rams/data/kitti/images/val")
    parser.add_argument("--kitti-labels", type=str, default="~/rams/data/kitti/labels/val")
    parser.add_argument("--coco-images", type=str, default=None)
    parser.add_argument("--coco-labels", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap exp8 images for smoke runs; omit for full val")
    parser.add_argument("--device", type=str, default="jetson-orin")
    parser.add_argument("--backend", choices=["tensorrt", "onnx", "pytorch"], default="tensorrt")
    parser.add_argument("--simulate", action="store_true", default=False,
                        help="Force simulated inference across the suite")
    parser.add_argument("--calibration-seconds", type=int, default=30)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-exp9", action="store_true")
    parser.add_argument("--skip-exp10", action="store_true")
    parser.add_argument("--require-full-kitti", action="store_true", default=False,
                        help="Fail preflight unless 1500 KITTI val images/labels are present")
    args = parser.parse_args()

    args.frames = expand(args.frames)
    args.kitti_images = expand(args.kitti_images)
    args.kitti_labels = expand(args.kitti_labels)
    args.coco_images = expand(args.coco_images)
    args.coco_labels = expand(args.coco_labels)

    manifest: list[dict] = []
    py = sys.executable
    all_ok = preflight(args, manifest)
    if not all_ok:
        manifest_path = RESULTS / "jetson_complete_runall_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return 2

    if args.backend == "tensorrt" and not args.skip_export and not args.simulate:
        run_cmd(
            "tensorrt_export",
            [py, "experiments/exp9_multidevice.py", "--device", args.device,
             "--backend", "tensorrt", "--frames", args.frames, "--n", "1",
             "--policy", "threshold", "--profile", "idle", "--export-trt", "--export-only"],
            manifest,
        )

    if not args.skip_calibration and not args.simulate:
        run_cmd(
            "calibration",
            [py, "scripts/calibrate.py", "--seconds", str(args.calibration_seconds), "--apply"],
            manifest,
        )

    for label, script in [
        ("exp1", "experiments/exp1_policy_comparison.py"),
        ("exp2", "experiments/exp2_load_sweep.py"),
        ("exp3", "experiments/exp3_hysteresis.py"),
        ("exp4", "experiments/exp4_safety_override.py"),
        ("exp5", "experiments/exp5_pareto.py"),
    ]:
        cmd = [py, script, "--n", str(args.n)]
        append_simulate(cmd, args.simulate)
        all_ok = run_cmd(label, cmd, manifest) and all_ok

    all_ok = run_cmd(
        "exp6",
        [py, "experiments/exp6_transient.py", "--total-n", str(args.transient_total_n),
         "--simulate" if args.simulate else "--no-simulate"],
        manifest,
    ) and all_ok

    all_ok = run_cmd(
        "exp7",
        [py, "experiments/exp7_swas.py", "--n", str(args.n),
         "--simulate" if args.simulate else "--no-simulate"],
        manifest,
    ) and all_ok

    benchmark_json = None
    if not args.skip_benchmark:
        bench_cmd = [py, "-m", "benchmark.run", "--n", str(args.n), "--policy", "all",
                     "--profile", "all", "--frames", args.frames]
        append_simulate(bench_cmd, args.simulate)
        all_ok = run_cmd("benchmark_for_exp10", bench_cmd, manifest) and all_ok
        benchmark_json = latest_file("rams_*.json")

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
    all_ok = run_cmd("exp8", exp8_cmd, manifest) and all_ok

    if not args.skip_exp9:
        exp9_cmd = [
            py, "experiments/exp9_multidevice.py",
            "--device", args.device,
            "--backend", args.backend,
            "--frames", args.frames,
            "--n", str(args.n),
        ]
        if args.simulate:
            exp9_cmd += ["--simulate"]
        all_ok = run_cmd("exp9_device", exp9_cmd, manifest) and all_ok
        all_ok = run_cmd(
            "exp9_aggregate",
            [py, "experiments/exp9_aggregate.py", "--results-dir", "results/multidevice"],
            manifest,
        ) and all_ok

    if not args.skip_exp10:
        benchmark_json = benchmark_json or latest_file("rams_*.json")
        exp10_inputs = [("kitti", str(RESULTS / "exp8_accuracy_kitti.json"))]
        if args.coco_images and args.coco_labels:
            exp10_inputs.append(("coco", str(RESULTS / "exp8_accuracy_coco.json")))
        for dataset, exp8_json in exp10_inputs:
            if Path(exp8_json).exists() and benchmark_json:
                all_ok = run_cmd(
                    f"exp10_{dataset}",
                    [py, "experiments/exp10_safety_pareto.py",
                     "--exp8-json", exp8_json,
                     "--benchmark-json", benchmark_json,
                     "--dataset", dataset],
                    manifest,
                ) and all_ok
            else:
                manifest.append({
                    "label": f"exp10_{dataset}",
                    "skipped": True,
                    "reason": "exp8 and/or benchmark JSON missing",
                })

    manifest_path = RESULTS / "jetson_complete_runall_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nDone.")
    print(f"Manifest: {manifest_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
