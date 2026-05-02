# RAMS: Resource-Adaptive Model Switching for Edge AI

RAMS is a runtime controller for edge perception systems that continuously
monitors resource pressure and switches among warm YOLOv8 detector tiers
(`NANO`, `SMALL`, `MEDIUM`) to balance latency and safety-sensitive accuracy.

This repository is structured for **reproducibility first**:

- core runtime code in `rams/`
- benchmark harness in `benchmark/`
- full experiment suite in `experiments/`
- calibration and utility scripts in `scripts/`
- device runbooks and reproducibility notes in `docs/`
- curated experimental artifacts in `results/`

The project supports three practical execution modes:

1. `simulate`: smoke tests and logic validation without models
2. `onnx`: reproducible CPU/ONNX experiments on Windows/Linux hosts
3. `tensorrt`: Jetson-focused accelerated deployment

## Repository Layout

```text
repo/
|- benchmark/             # end-to-end benchmark harness
|- configs/               # default runtime configuration
|- docs/                  # runbooks and reproducibility notes
|- experiments/           # exp1 ... exp10 and run-all scripts
|- rams/                  # controller, monitor, models, policies
|- results/               # curated device result artifacts
|- scripts/               # calibration, aggregation, live demo
|- .github/workflows/     # reproducibility smoke CI
|- REPRODUCIBILITY.md     # exact reproduction guidance
|- requirements.txt       # minimal/base dependencies
|- requirements-inference.txt
|- setup.py
`- CITATION.cff
```

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install base dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 3. Run a simulation smoke test

```bash
python -m benchmark.run --n 5 --policy threshold --profile heavy --simulate
```

This should create a small JSON/CSV pair under `results/`.

## Included Result Artifacts

This repository snapshot includes curated result folders for the main evaluated
deployment settings:

- `results/i7_1165G7/`
- `results/i7_13700F/`
- `results/raspberry_pi5/`
- `results/jetson_orin/onnx/`
- `results/jetson_orin/trt/`

These are included on purpose as part of the research artifact, so the repo is
not just source-only: it also carries the final summarized outputs used for
device comparison and paper drafting.

## Full Inference Setup

For real-image experiments, install the inference extras:

```bash
pip install -r requirements-inference.txt
```

Notes:

- `ultralytics`, `opencv-python`, and `onnxruntime` are needed for ONNX/PyTorch
  experiment paths.
- Jetson TensorRT reproduction should use the Jetson runbook in `docs/` because
  TensorRT and CUDA bindings are platform-specific.
- model weights/exports and datasets are **not** tracked in this repository

## Main Reproducibility Entry Points

- benchmark harness: `python -m benchmark.run`
- device calibration: `python scripts/calibrate.py --seconds 30 --apply`
- policy comparison: `python experiments/exp7_swas.py`
- tier accuracy: `python experiments/exp8_accuracy_per_tier.py`
- cross-device ONNX profiling: `python experiments/exp9_multidevice.py`
- Windows full suite: `python experiments/complete_runall.py`
- Jetson full suite: `python experiments/complete_runall_jetson.py`

The exact commands used to regenerate paper-facing outputs are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Windows and Jetson Users

If you want to use RAMS directly on Windows or Jetson, please check the
`packages/` folder for the packaged platform-specific materials and setup
guidance.

## What Is Not Included

To keep the repository GitHub-friendly and reproducible:

- paper source is excluded from this public repository snapshot
- datasets are excluded
- model weights and exported ONNX/engine files are excluded
- transient local outputs outside the curated `results/` artifact set are excluded
- temporary packaging artifacts are excluded

Use the runbooks and reproducibility guide to restore those dependencies on the
target machine.
