# RAMS Reproducibility Guide

This guide covers the current RAMS evaluation workflow. It produces runtime,
telemetry, tier-accuracy, policy-accuracy, and VRU-retention artifacts from
real replay frames. Simulation is useful for setup checks only and must not be
used as paper evidence.

The verified package archive in `packages/` is for the Windows CPU-ONNX
workflow. Use this guide directly for source-based Jetson and macOS runs.

## 1. Scope and required assets

The repository does not include datasets, model checkpoints, ONNX exports, or
TensorRT engines. A full run requires:

- YOLOv8 NANO, SMALL, and MEDIUM checkpoints and ONNX exports.
- KITTI 2D object-detection training images and labels, arranged as a fixed
  1,500-frame validation replay.
- COCO `val2017` images and matching Ultralytics YOLO-format labels.
- A completed device energy-profile JSON for runtime phases.

The minimum layouts are:

```text
<data-root>/kitti/images/val/
<data-root>/kitti/labels/val/
<data-root>/coco/images/val2017/
<data-root>/coco/labels/val2017/
```

Create the KITTI replay from sorted training-frame indices 5981 through 7480,
inclusive. Use exactly the same split across devices.

## 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-inference.txt
pip install -e .
python -c "from rams import RAMSController; print('RAMS import OK')"
```

On NVIDIA systems, also install `nvidia-ml-py` when NVML telemetry is
available. On Apple Silicon, CPU ONNX is the reference path. The optional
Core ML backend requires an ONNX Runtime build containing
`CoreMLExecutionProvider`; see the macOS package README.

## 3. Download and export models

Run from the repository root:

```bash
python -c "from ultralytics import YOLO; [YOLO(name) for name in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12); YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12); YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)"
```

The expected input sizes are NANO 320, SMALL 416, and MEDIUM 640.

## 4. Create the energy profile

Copy the template and document the actual device, operating mode, and power
basis:

```bash
cp configs/energy_profile.example.json configs/energy_profile_<device>.json
```

Energy fields are estimates based on the documented profile and recorded
telemetry. They are not physical power measurements. For CPU-only paths where
there is no GPU-utilization telemetry source, keep `gpu_dynamic_w` at zero.

## 5. Smoke check

Use real KITTI and COCO paths. The smoke mode confirms installation only.

```bash
python scripts/run_paper_suite.py --platform windows --backend onnx --smoke \
  --device <device-label> \
  --frames <data-root>/kitti/images/val \
  --kitti-labels <data-root>/kitti/labels/val \
  --coco-images <data-root>/coco/images/val2017 \
  --coco-labels <data-root>/coco/labels/val2017
```

Do not include smoke outputs in paper tables.

## 6. Full phased workflow

Set a platform and an explicit backend. Use `onnx` for the CPU reference path,
`tensorrt` only with engines built on the exact target NVIDIA device, and
`coreml` only after its provider check succeeds.

```bash
python scripts/run_paper_suite.py --phase calibration --platform <platform> \
  --device <device-label>

python scripts/run_paper_suite.py --phase runtime1 --platform <platform> --backend <backend> \
  --device <device-label> --frames <kitti-images> --kitti-labels <kitti-labels> \
  --coco-images <coco-images> --coco-labels <coco-labels> \
  --energy-profile configs/energy_profile_<device>.json

python scripts/run_paper_suite.py --phase runtime2 --platform <platform> --backend <backend> \
  --device <device-label> --frames <kitti-images> --kitti-labels <kitti-labels> \
  --coco-images <coco-images> --coco-labels <coco-labels> \
  --energy-profile configs/energy_profile_<device>.json

python scripts/run_paper_suite.py --phase runtime3 --platform <platform> --backend <backend> \
  --device <device-label> --frames <kitti-images> --kitti-labels <kitti-labels> \
  --coco-images <coco-images> --coco-labels <coco-labels> \
  --energy-profile configs/energy_profile_<device>.json

python scripts/run_paper_suite.py --phase runtime4 --platform <platform> --backend <backend> \
  --device <device-label> --frames <kitti-images> --kitti-labels <kitti-labels> \
  --coco-images <coco-images> --coco-labels <coco-labels> \
  --energy-profile configs/energy_profile_<device>.json

python scripts/run_paper_suite.py --phase accuracy --platform <platform> --backend <backend> \
  --device <device-label> --frames <kitti-images> --kitti-labels <kitti-labels> \
  --coco-images <coco-images> --coco-labels <coco-labels>

python scripts/run_paper_suite.py --phase retention --platform <platform> --backend <backend> \
  --device <device-label> --frames <kitti-images> --kitti-labels <kitti-labels> \
  --coco-images <coco-images> --coco-labels <coco-labels>
```

The suite runs 10 independent blocks of 200 frames for each reported runtime
policy and load setting. `runtime4` is the process-isolated burst protocol.
`scripts/run_phase2d_then_phase3.py` runs `runtime4` and `accuracy` back to
back when the same configuration should be preserved.

## 7. Required retained artifacts

Keep the entire `results/` directory. A complete runtime run includes:

- Raw CSV records with timing components and telemetry.
- JSON summaries with block-bootstrap latency intervals.
- Replay manifests containing ordered frame hashes and the random seed.
- Calibration snapshots and the completed energy profile.
- `exp5_pareto_*`, `exp8_accuracy_*`, `exp11_policy_accuracy_*`, and
  `exp12_retention_sensitivity_*` outputs.

For COCO mAP, each tier must state `map_source: ultralytics_val`. This is
supplementary tier context. Policy-level KITTI recall is the runtime-policy
accuracy evidence.

## 8. Interpretation limits

- VRU retention is reactive. It does not establish vehicle safety or repair an
  initial detector miss.
- TDP-profile energy is an estimate, not a power measurement.
- Never pool results from different devices or backends into one confidence
  interval.
- Do not reuse TensorRT engines across different Jetson device models.
- Label any zero-shot cross-dataset KITTI result and state its mapping and
  evaluation limitations.
