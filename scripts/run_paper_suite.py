"""One-command evidence collection for the revised RAMS embedded-systems letter.

Default invocation is deliberately a real, full KITTI replay.  ``--smoke``
keeps the same stages but uses a small real replay, so it validates the exact
paths and backends without creating publishable measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def default_data_root() -> Path:
    """Return the first usable dataset root without hard-coding one machine.

    ``RAMS_DATA_ROOT`` is the portable override used by package operators.  On
    the current Windows workstation, datasets are intentionally stored under
    ``D:\\data`` rather than beside the source tree, so recognize that location
    when it contains the required KITTI replay layout.
    """
    candidates: list[Path] = []
    configured = os.environ.get("RAMS_DATA_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((Path("D:/data"), ROOT.parent / "data", ROOT / "data"))
    for candidate in candidates:
        if (candidate / "kitti" / "images" / "val").is_dir():
            return candidate
    # Preserve a deterministic diagnostic path when datasets have not yet
    # been installed.  Preflight reports the missing directories clearly.
    return candidates[0] if candidates else ROOT / "data"


def defaults(target: str, data_root: Path) -> tuple[str, str]:
    if target == "auto":
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Linux" and "aarch64" in machine:
            target = "jetson"
        elif system == "Darwin" and machine in {"arm64", "aarch64"}:
            target = "macos"
        else:
            target = "windows"
    if target == "jetson":
        base = Path.home() / "rams" / "data" / "kitti"
    else:
        base = data_root / "kitti"
    return target, str(base / "images" / "val")


def run(label: str, command: list[str], manifest: list[dict]) -> bool:
    print("\n" + "=" * 88)
    print(f"[{label}] {' '.join(command)}")
    print("=" * 88)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    entry = {
        "label": label, "command": command, "returncode": completed.returncode,
        "elapsed_s": round(time.monotonic() - started, 2), "ok": completed.returncode == 0,
    }
    manifest.append(entry)
    return entry["ok"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every paper-facing RAMS measurement on one device")
    parser.add_argument("--platform", choices=["auto", "windows", "macos", "jetson"], default="auto")
    parser.add_argument("--device", default=platform.node(), help="Stable label recorded for this hardware")
    parser.add_argument("--backend", choices=["onnx", "coreml", "tensorrt", "pytorch"], default=None)
    parser.add_argument("--data-root", default=None,
                        help="Dataset root containing kitti/ and coco/. Overrides RAMS_DATA_ROOT and auto-detection.")
    parser.add_argument("--frames", default=None, help="KITTI images used for latency replay")
    parser.add_argument("--kitti-images", default=None)
    parser.add_argument("--kitti-labels", default=None)
    parser.add_argument("--coco-images", default=None,
                        help="COCO val2017 images for supplementary tier-level context")
    parser.add_argument("--coco-labels", default=None,
                        help="YOLO-format COCO val2017 labels")
    parser.add_argument("--energy-profile", default=None,
                        help="Completed per-device telemetry/TDP JSON profile (required for a full run)")
    parser.add_argument("--blocks", type=int, default=10, help="Independent paired replay blocks in a full run")
    parser.add_argument("--n", type=int, default=200, help="Frames per method/profile/block in a full run")
    parser.add_argument("--burst-peak-intensity", type=float, default=0.85,
                        help="host-level target during transient burst windows")
    parser.add_argument("--burst-on-frames", type=int, default=20,
                        help="frames in each burst-on window")
    parser.add_argument("--burst-off-frames", type=int, default=80,
                        help="frames in each recovery window")
    parser.add_argument("--burst-settle-s", type=float, default=0.20,
                        help="monitor-settle delay after a burst transition")
    parser.add_argument("--load-reserve-logical-cpus", type=int, default=1,
                        help="logical CPUs reserved from generated background load")
    parser.add_argument("--accuracy-profiles", default="moderate",
                        help="Comma-separated loads for policy-level KITTI accuracy")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--phase", choices=["all", "calibration", "runtime", "runtime1", "runtime2", "runtime3", "runtime4", "accuracy", "retention"], default="all",
                        help="Run all stages, calibration, runtime work, accuracy, or VRU-retention sensitivity")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Small real replay (1 block x 5 frames; 20 labelled frames), not paper evidence")
    parser.add_argument("--simulate", action="store_true",
                        help="Diagnostic-only mode; never produces paper-facing results")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser() if args.data_root else default_data_root()
    target, default_frames = defaults(args.platform, data_root)
    frames = Path(args.frames or default_frames).expanduser()
    images = Path(args.kitti_images or frames).expanduser()
    labels = Path(args.kitti_labels or (frames.parents[1] / "labels" / "val")).expanduser()
    coco_images = Path(args.coco_images or (data_root / "coco" / "images" / "val2017")).expanduser()
    coco_labels = Path(args.coco_labels or (data_root / "coco" / "labels" / "val2017")).expanduser()
    backend = args.backend or ("tensorrt" if target == "jetson" else "onnx")
    if target == "windows" and backend != "onnx":
        parser.error(
            "The Windows validation protocol requires --backend onnx. "
            "Use the Jetson or macOS package for their supported backends."
        )
    # Every child stage inherits an explicit backend requirement.  This keeps
    # the recorded backend from drifting from the one actually benchmarked.
    os.environ["RAMS_BACKEND"] = backend
    n, blocks, max_images = (5, 1, 20) if args.smoke else (args.n, args.blocks, None)

    runtime_phases = {"runtime", "runtime1", "runtime2", "runtime3", "runtime4"}
    needs_datasets = args.phase in {"all", "accuracy", "retention"} or args.phase in runtime_phases
    if not args.simulate and needs_datasets:
        missing = [str(path) for path in (frames, images, labels, coco_images, coco_labels) if not path.is_dir()]
        if missing:
            parser.error("Missing required KITTI or COCO directories: " + ", ".join(missing))
    needs_energy_profile = args.phase == "all" or args.phase in runtime_phases
    if needs_energy_profile and not args.smoke and not args.simulate:
        if not args.energy_profile:
            parser.error("A full run requires --energy-profile; copy configs/energy_profile.example.json and document the device power mode.")
        if not Path(args.energy_profile).is_file():
            parser.error(f"Energy profile not found: {args.energy_profile}")
    if n < 1 or blocks < 1:
        parser.error("--n and --blocks must be positive")
    if not 0.0 < args.burst_peak_intensity <= 1.0:
        parser.error("--burst-peak-intensity must be in (0, 1]")
    if args.burst_on_frames < 1 or args.burst_off_frames < 1:
        parser.error("--burst-on-frames and --burst-off-frames must both be positive")
    if args.burst_settle_s < 0:
        parser.error("--burst-settle-s must be non-negative")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest: list[dict] = [{
        "label": "preflight", "ok": True, "platform_target": target, "device": args.device,
        "backend": backend, "data_root": str(data_root), "frames": str(frames), "kitti_images": str(images), "kitti_labels": str(labels),
        "coco_images": str(coco_images), "coco_labels": str(coco_labels),
        "n_per_block": n, "blocks": blocks, "smoke": args.smoke, "simulated": args.simulate,
        "phase": args.phase, "python": sys.version, "os": platform.platform(), "machine": platform.machine(),
        "energy_profile": args.energy_profile, "requested_backend_env": backend,
        "load_protocol": {
            "steady_profiles": {
                "injector": "process_steady_v3",
                "reserved_logical_cpus": args.load_reserve_logical_cpus,
            },
            "burst_profile": {
                "injector": "process_isolated_burst_v2",
                "reserved_logical_cpus": args.load_reserve_logical_cpus,
                "burst_peak_intensity": args.burst_peak_intensity,
                "burst_on_frames": args.burst_on_frames,
                "burst_off_frames": args.burst_off_frames,
                "burst_settle_s": args.burst_settle_s,
            },
        },
    }]
    py = sys.executable
    infer_mode = "--simulate" if args.simulate else "--no-simulate"
    all_ok = True

    # Calibration changes configs/default.yaml.  Preserve both the existing
    # configuration and the applied one so a paper run remains auditable and
    # the previous threshold setting can always be restored manually.
    if args.phase in {"all", "calibration"}:
        if not args.skip_calibration and not args.smoke:
            snapshot_dir = RESULTS / "calibration_snapshots"
            snapshot_dir.mkdir(exist_ok=True)
            config_path = ROOT / "configs" / "default.yaml"
            before_path = snapshot_dir / f"default_before_calibration_{run_stamp}.yaml"
            shutil.copy2(config_path, before_path)
            manifest.append({"label": "calibration_config_before", "path": str(before_path), "ok": True})
            previous_calibrations = {path.resolve() for path in RESULTS.glob("calibration_*.json")}
            calibration = [py, "scripts/calibrate.py", "--frames", str(frames),
                           "--n", str(n), "--warmup", "20", "--apply"]
            all_ok = run("calibration", calibration, manifest) and all_ok
            after_path = snapshot_dir / f"default_after_calibration_{run_stamp}.yaml"
            shutil.copy2(config_path, after_path)
            new_calibrations = [str(path) for path in RESULTS.glob("calibration_*.json")
                                if path.resolve() not in previous_calibrations]
            manifest.append({"label": "calibration_config_after", "path": str(after_path),
                             "calibration_records": new_calibrations, "ok": True})
        else:
            manifest.append({"label": "calibration", "skipped": True, "reason": "--skip-calibration or --smoke"})

    runtime_plan = {
        # Runtime 1 and 4 have no Pareto run.  The paired Pareto work is split
        # between runtime 2 (moderate) and runtime 3 (heavy).
        "runtime1": (["idle", "light"], []),
        "runtime2": (["moderate"], ["moderate"]),
        "runtime3": (["heavy"], ["heavy"]),
        "runtime4": (["burst"], []),
    }
    if args.phase == "all":
        selected_runtime_phases = list(runtime_plan)
    elif args.phase == "runtime":
        selected_runtime_phases = ["runtime"]
    elif args.phase in runtime_plan:
        selected_runtime_phases = [args.phase]
    else:
        selected_runtime_phases = []

    for runtime_phase in selected_runtime_phases:
        profiles, pareto_scenarios = (
            (["idle", "light", "moderate", "heavy", "burst"], ["moderate", "heavy"])
            if runtime_phase == "runtime" else runtime_plan[runtime_phase]
        )
        for profile in profiles:
            benchmark = [py, "-m", "benchmark.run", infer_mode, "--frames", str(frames),
                         "--policy", "all", "--profile", profile, "--n", str(n), "--blocks", str(blocks),
                         "--seed", str(args.seed)]
            if not args.simulate:
                benchmark.append("--paper-mode")
            if args.energy_profile:
                benchmark += ["--energy-profile", args.energy_profile]
            if profile == "burst":
                benchmark += [
                    "--burst-peak-intensity", str(args.burst_peak_intensity),
                    "--burst-on-frames", str(args.burst_on_frames),
                    "--burst-off-frames", str(args.burst_off_frames),
                    "--burst-settle-s", str(args.burst_settle_s),
                    "--load-reserve-logical-cpus", str(args.load_reserve_logical_cpus),
                ]
            all_ok = run(f"paired_runtime_{runtime_phase}_{profile}", benchmark, manifest) and all_ok

        if pareto_scenarios:
            pareto = [py, "experiments/exp5_pareto.py", infer_mode, "--frames", str(frames),
                      "--n", str(n), "--blocks", str(blocks), "--seed", str(args.seed),
                      "--scenarios", ",".join(pareto_scenarios)]
            all_ok = run(f"pareto_{runtime_phase}", pareto, manifest) and all_ok

    if args.phase in {"all", "accuracy"}:
        tier_accuracy = [py, "experiments/exp8_accuracy_per_tier.py", "--dataset", "kitti",
                         "--images", str(images), "--labels", str(labels),
                         "--also-coco", "--coco-images", str(coco_images), "--coco-labels", str(coco_labels)]
        if max_images is not None:
            tier_accuracy += ["--max-images", str(max_images)]
        if args.simulate:
            tier_accuracy.append("--simulate")
        elif not args.smoke:
            tier_accuracy.append("--require-coco-ultralytics-map")
        all_ok = run("tier_accuracy_context", tier_accuracy, manifest) and all_ok

        policy_accuracy = [py, "experiments/exp11_policy_accuracy.py", "--images", str(images),
                           "--labels", str(labels), "--profiles", args.accuracy_profiles, "--seed", str(args.seed)]
        if max_images is not None:
            policy_accuracy += ["--max-images", str(max_images)]
        if args.simulate:
            policy_accuracy.append("--simulate")
        all_ok = run("policy_level_kitti_accuracy", policy_accuracy, manifest) and all_ok

    if args.phase in {"all", "retention"}:
        retention = [py, "experiments/exp12_retention_sensitivity.py", "--images", str(images),
                     "--labels", str(labels), "--profile", "moderate"]
        if max_images is not None:
            retention += ["--max-images", str(max_images)]
        all_ok = run("vru_retention_sensitivity", retention, manifest) and all_ok

    manifest_path = RESULTS / f"paper_{args.phase}_{run_stamp}_{args.device}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSuite manifest: {manifest_path}")
    print("Paper-facing outputs are valid only when smoke=false and simulated=false.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
