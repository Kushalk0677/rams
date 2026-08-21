# RAMS Reproducibility Guide

This guide defines the current RAMS evaluation workflow. It produces runtime,
telemetry, tier-accuracy, policy-accuracy, and VRU-retention artifacts from
real replay frames. Simulation is only a setup check and must not be used as
paper evidence.

The [README](README.md) gives the project overview. This document is the
canonical shared protocol: what may be compared, what must be retained, and
what may be claimed. It is not the preferred machine-setup guide. The package
README is the authoritative download-to-results guide for an operator on a
specific computer.

## Start with the correct package

Choose exactly one package for the target device. Download the archive, extract
it, and follow its root `README.md` from top to bottom before returning here.

| Target computer | Archive | Setup and run guide |
|---|---|---|
| Windows desktop or laptop | [`RAMS_Windows_validation.zip`](packages/RAMS_Windows_validation.zip) | [Windows guide](docs/RAMS_Windows_Runbook.md) |
| Apple Silicon Mac | [`RAMS_macOS_validation.zip`](packages/RAMS_macOS_validation.zip) | [macOS guide](docs/RAMS_macOS_Runbook.md) |
| NVIDIA Jetson | [`RAMS_Jetson_validation.zip`](packages/RAMS_Jetson_validation.zip) | [Jetson guide](docs/RAMS_Jetson_Runbook.md) |

The package guides install software, export or build the correct models,
download datasets, create the fixed KITTI replay without overwriting data, run
all gates, and provide the exact smoke, calibration, phase, and full-run
commands. This document defines how to interpret and retain those outputs.

## 1. Shared asset contract

The repository does not include datasets, checkpoints, ONNX exports, or
TensorRT engines. A full run needs:

- YOLOv8 NANO, SMALL, and MEDIUM checkpoints and their ONNX exports.
- KITTI 2D object-detection training images and labels, arranged as a fixed
  1,500-frame validation replay.
- COCO `val2017` images and matching Ultralytics YOLO-format labels.
- A completed device energy-profile JSON that documents the TDP basis and
  operating mode used for the estimate.

Use this data layout:

```text
<data-root>/kitti/images/val/
<data-root>/kitti/labels/val/
<data-root>/coco/images/val2017/
<data-root>/coco/labels/val2017/
```

Each package invokes `scripts/prepare_kitti_validation.py` to create the KITTI
replay from sorted training-frame indices 5981 through 7480 inclusive. Use
exactly this split on every device. The helper refuses to overwrite an existing
validation split. Download KITTI after registering at the
[KITTI 2D object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
Download COCO `val2017` from [COCO](https://cocodataset.org/#download) and the
corresponding Ultralytics YOLO-format label archive.

## 2. Source-checkout environment and models

This section is for researchers working directly from a repository clone. If
you are operating a downloaded package, use its README instead of these generic
commands.

Use Python 3.12 where supported by the platform. From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-inference.txt
pip install -e .
python -c "from rams import RAMSController; print('RAMS import OK')"
```

Download and export the three model tiers:

```bash
python -c "from ultralytics import YOLO; [YOLO(name) for name in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12); YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12); YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)"
```

The expected tier inputs are NANO 320, SMALL 416, and MEDIUM 640. Use the
platform package for platform-specific setup. In particular, do not replace
JetPack's PyTorch or TensorRT installation with generic CUDA wheels.

## 3. Backend matrix

| Backend | Intended platform | Evidence conditions |
|---|---|---|
| `onnx` | Windows, Linux, macOS | CPU ONNX Runtime reference path. |
| `tensorrt` | Jetson | Engines built on that exact Jetson, JetPack, TensorRT version, and build settings. |
| `coreml` | Apple Silicon macOS | Optional only when the ONNX Runtime CoreML provider is installed and recorded. |
| `pytorch` | Development fallback | Use only when the installed Ultralytics and PyTorch combination is confirmed compatible. |

For the current Windows protocol, use `--backend onnx`. NVIDIA hardware may
provide telemetry on Windows, but it is not an approved GPU-inference path for
the Windows CPU-ONNX evidence. Before a Jetson run, execute:

```bash
python3 scripts/verify_jetson_tensorrt.py --frame <one-real-kitti-frame>
```

It must report `"status": "passed"` with `"backend": "tensorrt"` for all
three tiers. Do not accept a fallback backend. WSL2 and hosted CI cannot
replace this target-Jetson preflight.

## 4. Energy profile and calibration

Copy the profile template, document the actual device and operating state, and
keep it beside the retained results:

```bash
cp configs/energy_profile.example.json configs/energy_profile_<device>.json
```

The energy model combines documented power assumptions with recorded runtime
telemetry. It estimates energy; it is not a physical power-meter measurement.
For a CPU-only route with no GPU-utilization source, use zero GPU dynamic power.

Run calibration once before the runtime phases:

```bash
python scripts/run_paper_suite.py --phase calibration --platform <platform> --device <device-label>
```

Calibration stores the configuration both before and after application under
`results/calibration_snapshots/`. Preserve the model files, energy profile,
backend, device power mode, clock condition, cooling condition, and calibration
configuration through all following phases.

## 5. Smoke preflight

Run the exact smoke command in the selected package README. It uses one block
of five frames and a small labelled subset. It proves installation only and
must not be reported as a measurement. A successful smoke manifest reports
`smoke: true`, `simulated: false`, and no failed command.

The script rejects paper-mode phases that lack real KITTI or COCO directories.
It also rejects a full runtime phase without `--energy-profile`.

## 6. Full phased workflow

All full runs use 10 independent blocks of 200 frames by default. The methods
in each block receive the same ordered replay trace; the manifest records its
hashes and random seed. GPU work is synchronized before timing where the
backend supports synchronization.

Run the exact phase commands from the selected package guide. Do not mix
Windows CPU ONNX, macOS CPU ONNX, and Jetson TensorRT commands or substitute a
backend partway through the sequence.

| Phase | Work |
|---|---|
| `runtime1` | Paired controller replay at idle and light load. |
| `runtime2` | Moderate replay and matched moderate Pareto evaluation. |
| `runtime3` | Heavy replay and matched heavy Pareto evaluation. |
| `runtime4` | Process-isolated burst replay. |
| `accuracy` | KITTI tier and policy accuracy plus required measured COCO tier validation. |
| `retention` | KITTI VRU-retention sensitivity analysis. |

Each package also provides a one-command `--phase all` full evaluation after
its setup and smoke gates pass. It runs phases `runtime1` through `retention`
in order after calibration. The script uses the `process_steady_v3` load
protocol for steady profiles and `process_isolated_burst_v2` for burst. Do not
combine its results with older thread-based Windows protocol results.

## 7. Required retained artifacts

Keep the complete device result directory. A reportable full run includes:

- Raw CSV records with timing components and telemetry.
- JSON summaries with block-bootstrap latency intervals.
- Replay manifests with ordered frame hashes, seed, backend, versions, and
  device state.
- Calibration snapshots, calibration records, and the energy-profile input.
- Tier accuracy, policy accuracy, VRU retention, and Pareto outputs.

For COCO mAP, report values only where the result records state
`map_source: ultralytics_val`. Such mAP is tier-level supplementary context.
Policy-level KITTI recall, precision, F1, false-negative rate, and non-VRU
performance are the primary accuracy evidence.

## 8. Interpretation limits

- VRU retention is reactive. It does not establish vehicle safety or repair an
  initially missed detector target.
- TDP-profile energy is an estimate, not a physical power measurement.
- Do not pool results from different devices, backends, calibrations, or load
  protocols into one confidence interval.
- Do not reuse TensorRT engines across Jetson devices or software stacks.
- Tier-level COCO mAP does not by itself establish policy-level accuracy.
- State the dataset mapping and limitations for any zero-shot cross-dataset
  result.
