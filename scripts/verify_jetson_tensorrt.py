"""Verify RAMS TensorRT inference on the target Jetson before a paper run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path


def first_frame(path: Path) -> Path:
    if path.is_file():
        return path
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        candidates = sorted(path.glob(pattern))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No image frame found under {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require all RAMS tiers to execute one real TensorRT inference on this Jetson"
    )
    parser.add_argument(
        "--frame",
        required=True,
        type=Path,
        help="One readable image file or a directory containing KITTI replay frames",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("This preflight must run on an ARM64 Jetson Linux target, not on WSL2 or a hosted runner.")
    if not shutil.which("tegrastats"):
        raise RuntimeError("tegrastats is unavailable. Repair the JetPack environment before collecting results.")

    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python bindings are unavailable in this environment.") from error
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is unavailable in this environment.") from error

    from rams.models import PROFILES, ModelLibrary, Tier

    engine_paths: dict[str, str] = {}
    for tier, profile in PROFILES.items():
        candidates = (
            Path(profile.model_id).with_suffix(".engine"),
            Path(f"{Path(profile.model_id).stem}_imgsz{profile.imgsz}.engine"),
        )
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            names = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"Missing {tier.name} TensorRT engine. Expected one of: {names}")
        engine_paths[tier.name] = str(selected.resolve())

    image_path = first_frame(args.frame.expanduser())
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"OpenCV could not read {image_path}")

    os.environ["RAMS_BACKEND"] = "tensorrt"
    library = ModelLibrary(simulate=False)
    library.load_all()
    inferences: dict[str, dict[str, object]] = {}
    for tier in (Tier.NANO, Tier.SMALL, Tier.MEDIUM):
        result = library.infer(tier, frame)
        if result.get("backend") != "tensorrt" or result.get("simulated"):
            raise RuntimeError(f"{tier.name} did not execute TensorRT: {result.get('backend')!r}")
        inferences[tier.name] = {
            "backend": result["backend"],
            "latency_ms": result.get("latency_ms"),
            "detections": len(result.get("detections", [])),
        }
    library.unload_all()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "passed",
        "backend": "tensorrt",
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "machine": platform.machine(),
        "tensor_rt_version": str(trt.__version__),
        "frame": str(image_path.resolve()),
        "engines": engine_paths,
        "inferences": inferences,
    }
    output = args.output_dir / f"jetson_tensorrt_preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"TensorRT preflight report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
