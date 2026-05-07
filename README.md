# RAMS: Resource-Adaptive Model Switching for Edge AI

RAMS is a Python runtime controller for edge perception pipelines. It monitors system resource pressure in real time and dynamically switches among three warm-loaded YOLOv8 detector tiers — `NANO`, `SMALL`, and `MEDIUM` — to keep inference latency and object-detection accuracy in balance. A safety override locks the system to a higher-accuracy tier whenever vulnerable road users (VRUs) are detected nearby.

> **Authors:** Kushal Khemani, Evan Leri, George Xu, Amit Hod

---

## How It Works

RAMS combines three cooperating components:

- **ResourceMonitor** — samples CPU/memory pressure at a configurable frequency (default 10 Hz) and produces a scalar resource signal *R(t)*.
- **Switching Policy** — maps *R(t)* to a target tier. Three policies are provided:
  - `threshold` — fixed *R(t)* thresholds with hysteresis to prevent rapid tier flapping.
  - `predictive` — EWMA short-horizon load forecasting that anticipates pressure spikes.
  - `safety` *(default)* — threshold policy plus a VRU proximity override that locks to `SMALL` or higher when pedestrians/cyclists are detected within a configurable time window.
- **ModelLibrary** — holds warm-loaded models at mixed resolutions (NANO @ 320 px, SMALL @ 416 px, MEDIUM @ 640 px) and dispatches inference requests to the best available backend (TensorRT → ONNX Runtime → Ultralytics PyTorch → calibrated simulation).

---

## Repository Layout

```
rams/                   # Core runtime: controller, monitor, models, policies
benchmark/              # End-to-end benchmark harness (benchmark.run)
experiments/            # exp1 – exp10 and complete run-all scripts
configs/                # default.yaml — all tunable parameters
scripts/                # Device calibration, aggregation, live demo
docs/                   # Runbooks for Windows, Jetson ONNX, and Jetson TRT
results/                # Curated output artifacts for five device settings
packages/               # Pre-built packages for Windows and Jetson Orin
REPRODUCIBILITY.md      # Exact commands for paper-facing reproduction
requirements.txt        # Minimal base dependencies
requirements-inference.txt
setup.py
CITATION.cff
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\\.venv\\Scripts\\Activate.ps1  # Windows PowerShell
```

### 2. Install base dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 3. Verify with a simulation smoke test (no models required)

```bash
python -m benchmark.run --n 5 --policy threshold --profile heavy --simulate
```

Expected output: a timestamped JSON/CSV pair in `results/`.

---

## Full Inference Setup

Real-image experiments require additional packages:

```bash
pip install -r requirements-inference.txt
```

This installs `ultralytics`, `onnxruntime`, `opencv-python`, PyTorch, and TorchVision. Model weights and datasets are **not** included in the repository — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for external asset setup.

For Jetson TensorRT deployment, follow `docs/RAMS_Jetson_Runbook.md` instead; TensorRT and CUDA bindings require platform-specific handling.

---

## Using the Controller

```python
from rams import RAMSController

# Simulation mode — no model files needed
with RAMSController(simulate=True, policy="safety") as ctrl:
    result = ctrl.infer(frame=my_frame)
    print(result["tier"], result["latency_ms"])
```

Available policies: `"threshold"`, `"predictive"`, `"safety"`. All parameters are tunable via `configs/default.yaml` or passed directly as `policy_kwargs`.

---

## Calibration

Before any real-device run, calibrate the resource thresholds to your hardware:

```bash
python scripts/calibrate.py --seconds 30 --apply
```

This writes a calibration JSON to `results/` and updates the threshold values used by the policy layer.

---

## Experiments

| Script | Description |
|---|---|
| `exp1_policy_comparison.py` | Latency and tier distribution across all three policies |
| `exp2_load_sweep.py` | Latency and switching rate vs. synthetic load |
| `exp3_hysteresis.py` | Hysteresis band sensitivity |
| `exp4_safety_override.py` | Safety override trigger analysis |
| `exp5_pareto.py` | Latency/accuracy Pareto curves |
| `exp6_transient.py` | Transient load spike response |
| `exp7_swas.py` | Safety-aware policy comparison on KITTI frames |
| `exp8_accuracy_per_tier.py` | Live VRU recall per tier on KITTI/COCO |
| `exp9_multidevice.py` | Cross-device ONNX runtime profiling |
| `exp10_safety_pareto.py` | Combined safety + Pareto frontier |
| `complete_runall.py` | Full suite for Windows/Linux ONNX hosts |
| `complete_runall_jetson.py` | Full suite for Jetson Orin |

---

## Included Result Artifacts

The repository ships curated result snapshots for five evaluated deployment settings so that reproduction can be validated against reference outputs:

| Device | Path |
|---|---|
| Intel Core i7-1165G7 | `results/i7_1165G7/` |
| Intel Core i7-13700F | `results/i7_13700F/` |
| Raspberry Pi 5 | `results/raspberry_pi5/` |
| Jetson Orin (ONNX) | `results/jetson_orin/onnx/` |
| Jetson Orin (TensorRT) | `results/jetson_orin/trt/` |

---

## Pre-built Packages

For out-of-the-box deployment without cloning the full repository:

- `packages/windows_package.zip` — Windows ONNX runtime bundle
- `packages/jetson_package.zip` — Jetson Orin bundle

Setup instructions are in the respective `docs/` runbooks.

---

## What Is Not Included

To keep the repository lightweight and GitHub-friendly:

- Paper source / LaTeX
- Datasets (KITTI, COCO)
- Model weights and ONNX/TensorRT engine files
- Transient local outputs outside the curated `results/` snapshot
- Temporary packaging artifacts

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for instructions on restoring all excluded dependencies on any target machine.

---

## Citation

If you use RAMS in your research, please cite using the metadata in [`CITATION.cff`](CITATION.cff).
