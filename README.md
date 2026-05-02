# RAMS: Resource-Adaptive Model Switching for Edge AI

RAMS is a runtime controller for edge perception systems that continuously
monitors resource pressure and switches among warm YOLOv8 detector tiers
(`NANO`, `SMALL`, `MEDIUM`) to balance latency and safety-sensitive accuracy.

This repository is structured for **reproducibility first**:

- core runtime code in `rams/`
- benchmark harness in `benchmark/`
- full experiment suite in `experiments/`
- calibration and utility scripts in `scripts/`
- paper source in `paper_ieee/`
- device runbooks in `docs/`

The project supports three practical execution modes:

1. `simulate`: smoke tests and logic validation without models
2. `onnx`: reproducible CPU/ONNX experiments on Windows/Linux hosts
3. `tensorrt`: Jetson-focused accelerated deployment

## Repository Layout

```text
repo/
|- benchmark/             # end-to-end benchmark harness
|- configs/               # default runtime configuration
|- docs/                  # runbooks and paper-facing notes
|- experiments/           # exp1 ... exp10 and run-all scripts
|- paper_ieee/            # IEEEtran paper source
|- rams/                  # controller, monitor, models, policies
|- results/               # local output directory (not versioned)
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

## Paper Source

The current IEEEtran paper draft is in:

- `paper_ieee/main.tex`
- `paper_ieee/references.bib`
- `paper_ieee/IEEEtran.cls`

Compile with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

from inside `paper_ieee/`.

## What Is Not Included

To keep the repository GitHub-friendly and reproducible:

- datasets are excluded
- model weights and exported ONNX/engine files are excluded
- local result folders are excluded
- temporary packaging artifacts are excluded

Use the runbooks and reproducibility guide to restore those dependencies on the
target machine.
