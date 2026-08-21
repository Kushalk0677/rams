"""Run the corrected burst replay and full accuracy phase back to back.

Phase 2d is runtime4, the transient burst replay. Phase 3 is the full KITTI
policy evaluation plus measured COCO tier-level validation. The script stops
before Phase 3 if Phase 2d fails and leaves the suite's per-phase manifests in
place. A small combined manifest records the two commands and outcomes.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--platform", choices=["auto", "windows", "macos", "jetson"], default="auto")
    parser.add_argument("--device", default=platform.node(), help="stable label recorded for this hardware")
    parser.add_argument("--backend", choices=["onnx", "coreml", "tensorrt", "pytorch"], default=None)
    parser.add_argument("--frames", default=None, help="KITTI replay image directory")
    parser.add_argument("--kitti-labels", default=None, help="KITTI label directory")
    parser.add_argument("--coco-images", default=None, help="COCO val2017 image directory")
    parser.add_argument("--coco-labels", default=None, help="YOLO-format COCO val2017 label directory")
    parser.add_argument("--energy-profile", default=None,
                        help="device-specific telemetry/TDP model required by a full Phase 2d run")
    parser.add_argument("--n", type=int, default=200, help="frames per policy/block")
    parser.add_argument("--blocks", type=int, default=10, help="independent replay blocks")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--accuracy-profiles", default="moderate",
                        help="comma-separated load profiles for Phase 3 policy accuracy")
    parser.add_argument("--burst-peak-intensity", type=float, default=0.85)
    parser.add_argument("--burst-on-frames", type=int, default=20)
    parser.add_argument("--burst-off-frames", type=int, default=80)
    parser.add_argument("--burst-settle-s", type=float, default=0.20)
    parser.add_argument("--load-reserve-logical-cpus", type=int, default=1)
    parser.add_argument("--smoke", action="store_true",
                        help="diagnostic only: use the suite's small replay settings")
    parser.add_argument("--simulate", action="store_true",
                        help="diagnostic only: never produces paper evidence")


def base_command(args: argparse.Namespace, phase: str) -> list[str]:
    command = [
        sys.executable, "scripts/run_paper_suite.py", "--phase", phase,
        "--platform", args.platform, "--device", args.device,
        "--n", str(args.n), "--blocks", str(args.blocks), "--seed", str(args.seed),
    ]
    optional_values = {
        "--backend": args.backend,
        "--frames": args.frames,
        "--kitti-labels": args.kitti_labels,
        "--coco-images": args.coco_images,
        "--coco-labels": args.coco_labels,
    }
    for flag, value in optional_values.items():
        if value:
            command += [flag, str(value)]
    if args.smoke:
        command.append("--smoke")
    if args.simulate:
        command.append("--simulate")
    return command


def run_stage(label: str, command: list[str]) -> dict:
    print("\n" + "=" * 88)
    print(f"[{label}] {' '.join(command)}")
    print("=" * 88)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT)
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_s": round(time.monotonic() - started, 2),
        "ok": completed.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RAMS Phase 2d transient burst replay, then Phase 3 accuracy."
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    if args.n < 1 or args.blocks < 1:
        parser.error("--n and --blocks must be positive")
    if not 0.0 < args.burst_peak_intensity <= 1.0:
        parser.error("--burst-peak-intensity must be in (0, 1]")
    if args.burst_on_frames < 1 or args.burst_off_frames < 1:
        parser.error("--burst-on-frames and --burst-off-frames must be positive")
    if args.burst_settle_s < 0:
        parser.error("--burst-settle-s must be non-negative")
    if not args.smoke and not args.simulate and not args.energy_profile:
        parser.error("a full Phase 2d run requires --energy-profile")

    RESULTS.mkdir(exist_ok=True)
    runtime4 = base_command(args, "runtime4")
    runtime4 += [
        "--burst-peak-intensity", str(args.burst_peak_intensity),
        "--burst-on-frames", str(args.burst_on_frames),
        "--burst-off-frames", str(args.burst_off_frames),
        "--burst-settle-s", str(args.burst_settle_s),
        "--load-reserve-logical-cpus", str(args.load_reserve_logical_cpus),
    ]
    if args.energy_profile:
        runtime4 += ["--energy-profile", args.energy_profile]
    accuracy = base_command(args, "accuracy")
    accuracy += ["--accuracy-profiles", args.accuracy_profiles]

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "phases": [],
    }
    runtime4_result = run_stage("phase2d_runtime4", runtime4)
    manifest["phases"].append(runtime4_result)
    if runtime4_result["ok"]:
        manifest["phases"].append(run_stage("phase3_accuracy", accuracy))
    else:
        print("\nPhase 2d failed. Phase 3 was not started.")

    manifest_path = RESULTS / (
        f"paper_runtime4_then_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nCombined manifest: {manifest_path}")
    return 0 if all(stage["ok"] for stage in manifest["phases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
