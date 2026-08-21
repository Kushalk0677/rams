# RAMS: Resource-Adaptive Model Switching for Edge Perception

[![arXiv](https://img.shields.io/badge/arXiv-2606.14716-b31b1b.svg)](https://arxiv.org/abs/2606.14716)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Kushal Khemani, Evan Leri, Amit Hod, George Xu**

## Contributors

- Amit Hod

RAMS is a runtime controller that selects among warm-loaded YOLOv8 NANO,
SMALL, and MEDIUM detectors according to measured resource pressure and recent
detections. It is intended for reproducible runtime-perception experiments on
embedded and desktop edge devices.

RAMS reports runtime behavior and detector outcomes. Its VRU-retention policies
are reactive prioritization policies, not safety guarantees. They cannot repair
a VRU that the active detector missed before the policy received a detection.

The accompanying paper is available on [arXiv](https://arxiv.org/abs/2606.14716).

## How it works

The controller keeps three detector tiers warm and makes one selection per
frame. The resource monitor samples CPU, memory, thermal, and, where supported,
NVIDIA GPU telemetry. A configurable pressure calculation informs the policy;
the selected model then returns detections in source-image coordinates.

```text
resource telemetry + previous detections
              |
              v
       pressure calculation
              |
              v
       switching policy -----> warm detector tier
              |                         |
              +----- next-frame state <-+
```

The repository retains the code names `threshold`, `predictive`, `adaptive`,
`safety`, and `safety2`. In the paper, these are respectively threshold,
EWMA-smoothed, variance-adaptive EWMA, VRU-retention, and two-level
VRU-retention. Fixed NANO, SMALL, and MEDIUM policies are also available for
fair controller-path comparisons.

## Runtime features

- CPU, memory, thermal, battery, and optional NVIDIA GPU telemetry.
- Configurable pressure with CPU, memory, thermal, GPU-utilization, and
  GPU-memory terms when those signals are supported.
- One source-image coordinate contract across ONNX, TensorRT, and Ultralytics
  inference paths.
- Normalized bounding-box area for two-level VRU retention, avoiding a
  resolution-dependent proximity threshold.
- Raw timing components for policy selection, preprocessing, inference,
  postprocessing, and controller end-to-end latency.
- Ordered replay manifests, block-bootstrap latency intervals, Wilson recall
  intervals, and documented TDP-profile energy estimates.

## Repository layout

```text
rams/              Runtime controller, monitor, models, policies, and energy model
benchmark/         Replay harness, load protocol, summaries, and confidence intervals
experiments/       Tier accuracy, Pareto, policy accuracy, and retention experiments
scripts/           Calibration, paper-suite, package, and platform-preflight commands
configs/           Controller defaults and per-device energy-profile templates
packaging/          Sources for the Windows, macOS, and Jetson operator packages
packages/           Ready-to-send Windows, macOS, and Jetson archives
results/            Curated current and historical evidence
tests/              Runtime contracts and regression tests
```

## Quick start

If you received one of the platform archives, do not use this generic source
checkout quick start. Go directly to [Platform packages](#platform-packages)
and follow the README inside the archive. The package README is the
authoritative setup and execution guide for that computer.

The base installation supports a simulated smoke check. It verifies the
controller and harness only. It is not paper evidence.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m benchmark.run --n 5 --policy threshold --profile heavy --simulate
```

For real inference, install the inference dependencies and download the three
model tiers:

```bash
pip install -r requirements-inference.txt
python -c "from ultralytics import YOLO; [YOLO(name) for name in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12); YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12); YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)"
```

Expected input sizes are NANO 320, SMALL 416, and MEDIUM 640. The repository
does not include checkpoints, ONNX exports, datasets, or TensorRT engines.

## Full evaluation setup

This section describes the shared asset contract for repository users. A person
running an archive should use its package README instead: it contains the exact
operating-system commands, checks, and stop conditions.

A paper-facing run requires real KITTI replay frames, COCO validation images
and labels, the three models, and a completed device energy-profile JSON. The
expected data root is:

```text
<data-root>/kitti/images/val/
<data-root>/kitti/labels/val/
<data-root>/coco/images/val2017/
<data-root>/coco/labels/val2017/
```

Use the fixed 1,500-frame KITTI validation replay formed from sorted training
frame indices 5981 through 7480 inclusive. Download KITTI 2D object-detection
data from the [KITTI benchmark site](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d),
and COCO `val2017` images from [COCO](https://cocodataset.org/#download), with
the matching Ultralytics YOLO-format labels.

Set the root explicitly before running:

```bash
# macOS/Linux
export RAMS_DATA_ROOT="$HOME/rams/data"
# Windows PowerShell: $env:RAMS_DATA_ROOT = 'D:\data'
```

The complete data preparation, calibration, phase sequence, retained outputs,
and interpretation constraints are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Run the paper suite

These are generic source-checkout examples. Use the exact Windows, macOS, or
Jetson commands in the relevant package guide for a paper-facing run.

Run a smoke check first. It uses a small real replay when the datasets are
present, validates the requested backend and paths, and must not be reported.

```bash
python scripts/run_paper_suite.py --platform windows --backend onnx --smoke --device <device-label>
```

For a full run, first document the device configuration and TDP assumptions:

```bash
cp configs/energy_profile.example.json configs/energy_profile_<device>.json
```

Then run calibration and the full phased suite. The default `--phase all`
performs runtime phases 1 through 4, tier accuracy, policy accuracy, and
retention sensitivity. It uses 10 independent blocks of 200 frames unless
overridden explicitly.

```bash
python scripts/run_paper_suite.py --phase calibration --platform <platform> --device <device-label>
python scripts/run_paper_suite.py --phase all --platform <platform> --backend <backend> --device <device-label> --energy-profile configs/energy_profile_<device>.json
```

Use `onnx` for the Windows CPU reference path, `onnx` for the macOS CPU
reference path, and `tensorrt` only with engines built on the exact target
Jetson. The full commands and platform requirements are in the operator
packages below.

## Platform packages

The archives are self-contained code and instructions for coauthors. They
exclude datasets, model weights, ONNX exports, TensorRT engines, environments,
and result records.

| Archive | Target route | Operator guide |
|---|---|---|
| [`RAMS_Windows_validation.zip`](packages/RAMS_Windows_validation.zip) | Windows CPU ONNX | README inside the archive and [Windows runbook](docs/RAMS_Windows_Runbook.md) |
| [`RAMS_macOS_validation.zip`](packages/RAMS_macOS_validation.zip) | Apple Silicon CPU ONNX | [macOS package guide](packaging/macos/README.md) |
| [`RAMS_Jetson_validation.zip`](packages/RAMS_Jetson_validation.zip) | Jetson TensorRT | [Jetson package guide](packaging/jetson/README.md) |

Download exactly one archive for the target computer, save it in that
computer's Downloads folder, extract it, then start with the `README.md` at
the root of the extracted `rams_validation` folder. Each package explains the
required software, model creation, dataset download, safe KITTI split creation,
smoke check, calibration, phased run, full run, and result handoff.

The documentation is deliberately split by responsibility:

| Need | Read this |
|---|---|
| Download, installation, data preparation, and commands on a target computer | The `README.md` inside that platform archive. |
| Shared experimental protocol, evidence required for a result, and interpretation limits | [REPRODUCIBILITY.md](REPRODUCIBILITY.md). |
| Project concepts, API orientation, result locations, and citation | This README. |

The Jetson package has a mandatory target-device TensorRT preflight. It must
load every device-built engine and run one real inference per tier before a
paper phase begins. GitHub-hosted Linux and WSL2 check package compatibility;
they cannot validate JetPack or a Jetson TensorRT engine.

Rebuild all current archives from tracked sources with:

```bash
python scripts/build_platform_packages.py --platform all
```

## Curated results

Current revised-protocol evidence is organized by processor family:

| Directory | Device and backend | Scope |
|---|---|---|
| [`results/m4_air`](results/m4_air) | Apple M4 MacBook Air, CPU ONNX | Real KITTI replay, calibration, raw records, figures, and provenance. |
| [`results/i7_11800H`](results/i7_11800H) | Intel Core i7-11800H, Windows CPU ONNX | Completed phased runtime evaluation, COCO validation, KITTI policy metrics, retention analysis, and manifests. The RTX 3050 Ti was not used for inference. |
| [`results/i7_11800H_rtx3050ti_cuda_onnx`](results/i7_11800H_rtx3050ti_cuda_onnx) | Intel Core i7-11800H with RTX 3050 Ti, Windows CUDA ONNX | Separate supplied CUDA-ONNX evidence. It includes a fresh 1,500-frame mapped KITTI COCOeval run: NANO 0.0470/0.0256, SMALL 0.0700/0.0370, MEDIUM 0.0736/0.0400 for mAP@0.50/mAP@0.50:0.95. |
| [`results/old`](results/old) | Historical devices and protocols | Retained for traceability only. Do not pool with revised-protocol results. |

TDP-profile energy fields are telemetry-conditioned estimates, not external
power measurements. mAP is reportable only where the records identify a fresh
evaluation source, such as `ultralytics_val`, `onnx_cuda_cocoeval`, or
`kitti_native_mapped_cocoeval`.

## Scope

RAMS evaluates runtime behavior, resource telemetry, and detector outcomes.
It does not establish vehicle safety, correct initially missed objects, model
closed-loop vehicle dynamics, or measure physical energy consumption. TensorRT
engines must never be reused across different Jetson models or software stacks.

## Citation

Use [`CITATION.cff`](CITATION.cff) for software citation, or cite the current
arXiv record:

```bibtex
@misc{khemani2026rams,
  title={RAMS: Resource-Adaptive Model Switching for Edge Perception},
  author={Khemani, Kushal and Leri, Evan and Hod, Amit and Xu, George},
  year={2026},
  eprint={2606.14716},
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
