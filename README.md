# RAMS: Resource-Adaptive Model Switching for Edge Perception

RAMS is a runtime controller that selects among warm-loaded YOLOv8 NANO,
SMALL, and MEDIUM tiers according to resource pressure and recent detections.
It is intended for reproducible runtime-perception experiments on embedded and
desktop edge devices.

RAMS reports runtime behaviour and detection outcomes. Its VRU-retention
policies are reactive prioritization policies, not safety guarantees. They
cannot correct VRUs missed before the policy receives a detection.

## Runtime features

- CPU, memory, thermal, battery, and optional NVIDIA GPU telemetry.
- Configurable pressure calculation with CPU, memory, thermal, GPU-utilization,
  and GPU-memory terms when supported by the device.
- Common source-image coordinate contract across ONNX, TensorRT, and
  Ultralytics paths.
- Normalized bounding-box area for two-level VRU retention, avoiding
  resolution-dependent proximity decisions.
- Five policies: `threshold`, `predictive`, `adaptive`, `safety`, and
  `safety2`.
- Paper-facing aliases: `ewma_smoothed`, `variance_adaptive_ewma`,
  `vru_retention`, and `vru_retention2`.
- Fixed-tier controller policies for NANO, SMALL, and MEDIUM comparisons.
- Raw timing components for policy selection, preprocessing, inference,
  postprocessing, and controller end-to-end latency.
- Replay manifests, block-bootstrap latency intervals, Wilson recall intervals,
  and documented TDP-profile energy estimates.

## Backends

| Backend | Intended platform | Notes |
|---|---|---|
| `onnx` | Windows, Linux, macOS | CPU ONNX Runtime reference path. |
| `tensorrt` | Jetson or supported NVIDIA systems | Requires device-specific TensorRT engines. |
| `coreml` | Apple Silicon macOS | Requires an ONNX Runtime build with `CoreMLExecutionProvider`; CPU fallback remains possible for unsupported graph partitions. |
| `pytorch` | Development fallback | Uses Ultralytics when compatible PyTorch support is available. |

## Quick start

Create a virtual environment, install the base package, and run a simulation
smoke check:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m benchmark.run --n 5 --policy threshold --profile heavy --simulate
```

For real inference, install the additional inference dependencies:

```bash
pip install -r requirements-inference.txt
```

Download or export the three YOLOv8 models in the repository root:

```bash
python -c "from ultralytics import YOLO; [YOLO(name) for name in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12); YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12); YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)"
```

Use [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the required KITTI and COCO
layout, calibration, phased paper workflow, manifests, and interpretation
limits. The verified Windows package provides a coauthor-ready CPU-ONNX setup
process.

## Platform packages

The verified public package is:

- `packages/RAMS_Windows_validation_process_v3_20260820.zip`

It runs the Windows CPU-ONNX process-v3 workflow. Its presence of NVIDIA
hardware may enable telemetry only; the package does not run inference on a
Windows GPU. Jetson and macOS users should currently reproduce from the source
workflow in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Do not reuse a
TensorRT engine across different Jetson device models.

## Curated result artifacts

The repository contains result artifacts and platform packages, not datasets,
model weights, ONNX exports, or TensorRT engines.

- `results/m4_air/` contains the Apple M4 MacBook Air CPU-ONNX phased evaluation.
- `results/i7_11800H/` contains the Intel Core i7-11800H CPU-ONNX process-v3
  evaluation.
- `results/old/` contains the historical device results in their original
  per-device directory layout.
- `packages/` contains coauthor deployment packages.

The M4 MacBook Air evidence is CPU ONNX only. Its energy fields are
telemetry-derived TDP-profile estimates, not physical power measurements.

## Repository layout

```text
rams/          Runtime controller, monitor, models, policies, and energy model
benchmark/     Replay benchmark harness and confidence intervals
experiments/   Tier accuracy, Pareto, policy accuracy, and retention experiments
scripts/       Calibration and phased paper workflow runners
configs/       Controller defaults and device energy-profile templates
tests/         Runtime-contract and regression tests
packages/      Verified public distribution archives
results/       Curated result artifacts
```

## Citation

See [`CITATION.cff`](CITATION.cff) for the repository citation record.

## Authors

Kushal Khemani, Evan Leri, Amit Hod, and George Xu.
