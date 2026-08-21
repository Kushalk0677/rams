"""Build the distributable Windows, macOS, and Jetson validation packages.

The archives deliberately exclude datasets, model files, TensorRT engines,
virtual environments, and result records.  Each archive contains the same
current runtime source and a platform-specific operator guide.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"

PLATFORMS = {
    "windows": {
        "archive": "RAMS_Windows_validation.zip",
        "guide": "windows/README.md",
        "title": "RAMS Windows validation package",
        "scope": "Windows CPU ONNX Runtime validation. NVIDIA hardware is telemetry-only.",
    },
    "macos": {
        "archive": "RAMS_macOS_validation.zip",
        "guide": "macos/README.md",
        "title": "RAMS macOS Apple Silicon validation package",
        "scope": "CPU ONNX Runtime validation on Apple Silicon macOS.",
    },
    "jetson": {
        "archive": "RAMS_Jetson_validation.zip",
        "guide": "jetson/README.md",
        "title": "RAMS Jetson validation package",
        "scope": "Jetson TensorRT validation with device-built engines.",
    },
}

INCLUDED = (
    "benchmark",
    "configs",
    "experiments",
    "rams",
    "scripts",
    "tests",
    "requirements.txt",
    "requirements-inference.txt",
    "setup.py",
)
EXCLUDED = {
    "scripts/build_platform_packages.py",
    "tests/test_letter_revision.py",
    "__pycache__",
    ".pytest_cache",
}


def include_file(relative: Path) -> bool:
    normalized = str(relative).replace("\\", "/")
    if any(part in EXCLUDED for part in relative.parts) or normalized in EXCLUDED:
        return False
    return not (
        relative.parent == Path("configs")
        and relative.name.startswith("energy_profile_")
        and relative.name != "energy_profile.example.json"
    )


def copy_source(destination: Path) -> None:
    for relative_name in INCLUDED:
        source = ROOT / relative_name
        target = destination / relative_name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if include_file(relative):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def write_metadata(destination: Path, spec: dict[str, str]) -> None:
    guide = PACKAGING / spec["guide"]
    shutil.copy2(guide, destination / "README.md")
    (destination / "AI_OPERATOR_INSTRUCTIONS.md").write_text(
        "# RAMS operator instructions\n\n"
        "You are operating a reproducible evaluation package. Read `README.md` "
        "completely and execute its numbered sections in order. Work only inside "
        "the extracted package directory and never alter its source code, default "
        "configuration, models, or datasets unless the README explicitly directs it.\n\n"
        "Before calibration, confirm Python, the requested inference backend, all "
        "three model tiers, the fixed KITTI replay, and COCO image and label paths. "
        "Run the documented smoke command and stop on any failure. A smoke run is "
        "not evidence.\n\n"
        "For a full evaluation, create the device energy-profile input, run "
        "calibration once, preserve the resulting configuration, then run the "
        "phases in order. Retain the complete `results/` directory, including raw "
        "records, manifests, calibration snapshots, device state, and the energy "
        "profile. Do not report simulated results, proxy mAP, or estimated energy "
        "as measured physical energy.\n",
        encoding="utf-8",
    )
    (destination / "PACKAGE_MANIFEST.md").write_text(
        f"# {spec['title']}\n\n"
        f"Scope: {spec['scope']}\n\n"
        "Included: current RAMS source, tests, configuration, and operator documentation.\n\n"
        "Excluded: datasets, model checkpoints, ONNX exports, TensorRT engines, "
        "virtual environments, and result records.\n",
        encoding="utf-8",
    )


def archive_directory(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent).as_posix())


def build(platform_name: str, output_dir: Path) -> Path:
    spec = PLATFORMS[platform_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / spec["archive"]
    with tempfile.TemporaryDirectory(prefix="rams_package_") as temporary:
        package_root = Path(temporary) / "rams_validation"
        package_root.mkdir()
        copy_source(package_root)
        write_metadata(package_root, spec)
        archive_directory(package_root, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RAMS Windows, macOS, and Jetson packages")
    parser.add_argument("--platform", choices=["windows", "macos", "jetson", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "packages")
    args = parser.parse_args()
    selected = PLATFORMS if args.platform == "all" else {args.platform: PLATFORMS[args.platform]}
    for platform_name in selected:
        print(build(platform_name, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
