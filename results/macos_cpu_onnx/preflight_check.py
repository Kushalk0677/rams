#!/usr/bin/env python3
"""RAMS preflight validator.

Run this from the package directory BEFORE every paper-facing phase.  It fails
loudly on the conditions that would otherwise let a run finish "successfully"
while producing unusable numbers -- above all the silent simulation fallback in
rams/models.py, which never raises and never warns at suite level.

Usage:
    python preflight_check.py --backend onnx \
        --kitti-images ~/rams/data/kitti/images/val \
        --kitti-labels ~/rams/data/kitti/labels/val \
        --coco-images  ~/rams/data/coco/images/val2017 \
        --coco-labels  ~/rams/data/coco/labels/val2017 \
        --energy-profile configs/energy_profile_macos.json \
        --estimate

Exit code 0 means every hard check passed and the phase is safe to start.
Exit code 1 means at least one hard check failed -- do not run the phase.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_MODELS = [
    ("yolov8n.pt", 1_000_000), ("yolov8s.pt", 5_000_000), ("yolov8m.pt", 15_000_000),
    ("yolov8n.onnx", 1_000_000), ("yolov8s.onnx", 5_000_000), ("yolov8m.onnx", 15_000_000),
]
PLACEHOLDER_MARKERS = ("replace-with", "TODO", "<", "xxx")

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------
# 1. Interpreter and dependencies
# --------------------------------------------------------------------------

def check_environment(backend: str) -> None:
    section("1. Interpreter and dependencies")

    major, minor = sys.version_info[:2]
    if (major, minor) in {(3, 10), (3, 11), (3, 12)}:
        ok(f"Python {major}.{minor} is a supported version")
    else:
        fail(f"Python {major}.{minor} is unsupported; use 3.10, 3.11, or 3.12")

    if (ROOT / ".venv").is_dir() and sys.prefix == sys.base_prefix:
        warn("A .venv exists but this interpreter is not it -- did you activate?")

    for module, label in [
        ("cv2", "opencv-python"), ("numpy", "numpy"), ("psutil", "psutil"),
        ("yaml", "pyyaml"), ("onnxruntime", "onnxruntime"), ("ultralytics", "ultralytics"),
    ]:
        try:
            __import__(module)
            ok(f"{label} imports")
        except ImportError:
            fail(f"{label} is missing -- pip install {label}")

    try:
        from rams import RAMSController  # noqa: F401
        ok("rams package imports")
    except Exception as exc:  # pragma: no cover - environment dependent
        fail(f"rams package does not import: {exc}")

    # Backend / execution-provider agreement
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        ok(f"onnxruntime {ort.__version__}; providers: {', '.join(providers)}")
        if "CPUExecutionProvider" not in providers:
            fail("CPUExecutionProvider is unavailable -- the reference path cannot run")
        if backend == "coreml" and "CoreMLExecutionProvider" not in providers:
            fail(
                "--backend coreml requested but CoreMLExecutionProvider is absent. "
                "Build onnxruntime with --use_coreml and install that wheel, or use --backend onnx"
            )
        elif backend == "coreml":
            ok("CoreMLExecutionProvider is available")
    except ImportError:
        pass  # already reported above


# --------------------------------------------------------------------------
# 2. Model files
# --------------------------------------------------------------------------

def check_models() -> None:
    section("2. Model files (all six must exist)")
    for name, min_bytes in REQUIRED_MODELS:
        path = ROOT / name
        if not path.is_file():
            fail(f"{name} is missing -- rerun the export step in Runbook section 2")
        elif path.stat().st_size < min_bytes:
            fail(f"{name} is only {path.stat().st_size:,} B -- export looks truncated")
        else:
            ok(f"{name} ({path.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------
# 3. Datasets
# --------------------------------------------------------------------------

def check_kitti(images: Path, labels: Path) -> None:
    section("3. KITTI validation split")
    if not images.is_dir():
        fail(f"KITTI image directory not found: {images}")
        return
    if not labels.is_dir():
        fail(f"KITTI label directory not found: {labels}")
        return

    image_files = sorted(images.glob("*.png"))
    label_files = sorted(labels.glob("*.txt"))

    if len(image_files) == 1500:
        ok("1500 KITTI images")
    else:
        fail(f"{len(image_files)} KITTI images, expected exactly 1500 (frames 5981-7480)")

    if len(label_files) == 1500:
        ok("1500 KITTI labels")
    else:
        fail(f"{len(label_files)} KITTI labels, expected exactly 1500")

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}
    orphans = image_stems ^ label_stems
    if orphans:
        sample = ", ".join(sorted(orphans)[:5])
        fail(f"{len(orphans)} KITTI image/label stems do not pair (e.g. {sample})")
    elif image_files:
        ok("every KITTI image has a matching label")


def check_coco(images: Path, labels: Path) -> None:
    section("4. COCO val2017")
    if not images.is_dir():
        fail(f"COCO image directory not found: {images}")
        return
    if not labels.is_dir():
        fail(f"COCO label directory not found: {labels}")
        return

    n_images = sum(1 for _ in images.glob("*.jpg"))
    n_labels = sum(1 for _ in labels.glob("*.txt"))

    if n_images == 5000:
        ok("5000 COCO images")
    else:
        fail(f"{n_images} COCO images, expected 5000")

    # Fewer labels than images is expected: some COCO images have no objects.
    if 0 < n_labels <= n_images:
        ok(f"{n_labels} COCO label files (fewer than images is normal)")
    else:
        fail(f"{n_labels} COCO label files looks wrong against {n_images} images")


# --------------------------------------------------------------------------
# 5. Energy profile
# --------------------------------------------------------------------------

def check_energy_profile(path: Path | None) -> None:
    section("5. Energy profile")
    if path is None:
        warn("no --energy-profile given; required for runtime phases (not for accuracy/retention)")
        return
    if not path.is_file():
        fail(f"energy profile not found: {path}")
        return

    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"energy profile is not valid JSON: {exc}")
        return

    for field in ("name", "source"):
        value = str(profile.get(field, ""))
        if not value or any(marker in value for marker in PLACEHOLDER_MARKERS):
            fail(f"energy profile '{field}' is still a placeholder: {value!r}")
        else:
            ok(f"{field}: {value}")

    gpu = profile.get("gpu_dynamic_w")
    if gpu == 0:
        ok("gpu_dynamic_w is 0 (required: no portable Apple GPU telemetry source)")
    else:
        fail(f"gpu_dynamic_w is {gpu}; must be 0 on this package")

    for field in ("idle_w", "cpu_dynamic_w", "memory_dynamic_w", "max_power_w"):
        value = profile.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            fail(f"energy profile '{field}' is missing or non-positive: {value!r}")


# --------------------------------------------------------------------------
# 6. Live backend probe -- the check that matters most
# --------------------------------------------------------------------------

def check_live_backend(backend: str, kitti_images: Path, estimate: bool) -> None:
    section("6. Live backend probe (catches the silent simulation fallback)")

    import os
    os.environ["RAMS_BACKEND"] = backend

    try:
        import cv2
        from rams import RAMSController, FixedTierPolicy, Tier
    except Exception as exc:
        fail(f"cannot import for probe: {exc}")
        return

    frame = None
    if kitti_images.is_dir():
        candidates = sorted(kitti_images.glob("*.png"))
        if candidates:
            frame = cv2.imread(str(candidates[0]))
    if frame is None:
        warn("no KITTI frame available; probing with a synthetic frame instead")

    per_tier_ms: dict[str, float] = {}

    for tier in (Tier.NANO, Tier.SMALL, Tier.MEDIUM):
        try:
            # NOTE: README shows policy='fixed_nano', which raises ValueError.
            # make_policy has no fixed-tier string; pass the object instead.
            with RAMSController(simulate=False, policy=FixedTierPolicy(tier)) as controller:
                controller.infer(frame)          # warm-up, never timed
                samples = []
                for _ in range(5 if estimate else 1):
                    started = time.perf_counter()
                    result = controller.infer(frame)
                    samples.append((time.perf_counter() - started) * 1000.0)
        except Exception as exc:
            fail(f"{tier.name}: inference raised {type(exc).__name__}: {exc}")
            continue

        actual = result.get("backend")
        simulated = result.get("simulated", False)

        if simulated or actual == "simulation":
            fail(
                f"{tier.name}: backend fell back to SIMULATION. These numbers are "
                f"synthetic Gaussians from models.py, not measurements. Do not run the phase."
            )
        elif actual != backend:
            fail(f"{tier.name}: requested backend {backend!r} but got {actual!r}")
        else:
            per_tier_ms[tier.name] = sum(samples) / len(samples)
            detail = f"{tier.name}: real {actual} inference, {per_tier_ms[tier.name]:.1f} ms/frame"
            providers = result.get("execution_providers")
            if providers:
                detail += f" [{providers}]"
            ok(detail)

            if backend == "coreml" and providers and "CoreMLExecutionProvider" not in str(providers):
                fail(f"{tier.name}: coreml requested but provider chain is {providers}")

    if estimate and per_tier_ms:
        project_runtime(per_tier_ms)


def project_runtime(per_tier_ms: dict[str, float]) -> None:
    section("7. Projected wall-clock (rough, from measured per-tier latency)")
    mean_ms = sum(per_tier_ms.values()) / len(per_tier_ms)

    # 5 canonical policies x n(200) x blocks(10) inferences per load profile.
    per_profile_s = 5 * 200 * 10 * mean_ms / 1000.0
    plan = [
        ("runtime1 (idle + light)", 2 * per_profile_s),
        ("runtime2 (moderate + Pareto)", 2 * per_profile_s),
        ("runtime3 (heavy + Pareto)", 2 * per_profile_s),
        ("runtime4 (burst)", per_profile_s),
        ("accuracy: KITTI 1500 x 3 tiers", 1500 * 3 * mean_ms / 1000.0),
        ("accuracy: COCO 5000 x 3 tiers", 5000 * 3 * mean_ms / 1000.0),
        ("retention (2 confidence levels)", 2 * 1500 * mean_ms / 1000.0),
    ]
    total = 0.0
    for label, seconds in plan:
        total += seconds
        print(f"  ~{seconds / 60:6.0f} min   {label}")
    print(f"  {'-' * 46}")
    print(f"  ~{total / 60:6.0f} min   TOTAL (~{total / 3600:.1f} h), excluding load-injection overhead")
    print("  Treat as a lower bound: background load injection and thermal")
    print("  throttling both push this up on a laptop.")


# --------------------------------------------------------------------------

def check_disk() -> None:
    section("8. Disk")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb >= 25:
        ok(f"{free_gb:.1f} GB free")
    else:
        warn(f"only {free_gb:.1f} GB free; the runbook asks for ~25 GB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a RAMS environment before a paper phase")
    parser.add_argument("--backend", choices=["onnx", "coreml"], default="onnx")
    parser.add_argument("--kitti-images", type=Path, default=Path.home() / "rams/data/kitti/images/val")
    parser.add_argument("--kitti-labels", type=Path, default=Path.home() / "rams/data/kitti/labels/val")
    parser.add_argument("--coco-images", type=Path, default=Path.home() / "rams/data/coco/images/val2017")
    parser.add_argument("--coco-labels", type=Path, default=Path.home() / "rams/data/coco/labels/val2017")
    parser.add_argument("--energy-profile", type=Path, default=None)
    parser.add_argument("--estimate", action="store_true",
                        help="Time 5 frames per tier and project total suite wall-clock")
    parser.add_argument("--skip-datasets", action="store_true",
                        help="Environment and model checks only")
    args = parser.parse_args()

    print("=" * 70)
    print(f"RAMS PREFLIGHT  |  backend={args.backend}")
    print("=" * 70)

    check_environment(args.backend)
    check_models()
    if not args.skip_datasets:
        check_kitti(args.kitti_images.expanduser(), args.kitti_labels.expanduser())
        check_coco(args.coco_images.expanduser(), args.coco_labels.expanduser())
    check_energy_profile(args.energy_profile.expanduser() if args.energy_profile else None)
    check_live_backend(args.backend, args.kitti_images.expanduser(), args.estimate)
    check_disk()

    print("\n" + "=" * 70)
    if failures:
        print(f"PREFLIGHT FAILED -- {len(failures)} blocking problem(s). Do NOT start the phase.")
        for item in failures:
            print(f"  - {item}")
        if warnings:
            print(f"\n{len(warnings)} warning(s):")
            for item in warnings:
                print(f"  - {item}")
        print("=" * 70)
        return 1

    print("PREFLIGHT PASSED -- real backend confirmed, safe to start the phase.")
    if warnings:
        print(f"\n{len(warnings)} warning(s) worth a look:")
        for item in warnings:
            print(f"  - {item}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
