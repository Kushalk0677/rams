# Reproducibility Guide

This document explains how to reproduce RAMS experiments in a way that is
consistent with the project artifacts used for evaluation.

The repository already includes a curated `results/` snapshot for the main
devices, so reproduction can be checked against those stored artifacts rather
than starting from an empty output tree.

## 1. Scope

This repository supports four practical levels of reproduction:

1. **Simulation smoke reproduction**
   - validates controller logic, switching, and benchmark paths
   - no datasets or model files required

2. **ONNX runtime reproduction**
   - Windows/Linux CPU baseline path
   - suitable for `exp7`, `exp8`, `exp9`, and full run-all

3. **Jetson ONNX reproduction**
   - same controller on Jetson without TensorRT acceleration

4. **Jetson TensorRT reproduction**
   - accelerated embedded deployment path
   - platform-specific and documented separately in the Jetson runbook

## 2. Dependencies

### Base

```bash
pip install -r requirements.txt
pip install -e .
```

### Inference

```bash
pip install -r requirements-inference.txt
```

Jetson TensorRT users should not assume generic pip packages are sufficient.
Follow `docs/RAMS_Jetson_Runbook.md`.

## 3. External Assets Required

The following assets are expected locally but are not versioned:

- YOLO model weights or exports:
  - `yolov8n.pt` or `yolov8n.onnx`
  - `yolov8s.pt` or `yolov8s.onnx`
  - `yolov8m.pt` or `yolov8m.onnx`
- KITTI validation images/labels
- optional COCO validation images/labels
- Jetson TensorRT `.engine` files for the TRT backend

## 3.1 Included Curated Results

The repository includes final curated outputs for:

- `results/i7_1165G7/`
- `results/i7_13700F/`
- `results/raspberry_pi5/`
- `results/jetson_orin/onnx/`
- `results/jetson_orin/trt/`

These folders are intended as reference artifacts for comparison and paper
traceability. Fresh runs may produce additional local files, but the included
snapshot should remain stable.

## 4. Minimum Smoke Check

Run this first on any machine:

```bash
python -m benchmark.run --n 5 --policy threshold --profile heavy --simulate
```

Expected outcome:

- the command exits successfully
- `results/` contains a timestamped JSON/CSV output pair

## 5. Calibration

Before any real-device run, calibrate locally:

```bash
python scripts/calibrate.py --seconds 30 --apply
```

This writes a calibration JSON to `results/` and updates the runtime thresholds
used by the policy layer.

## 6. Paper-Facing Commands

### 6.1 Jetson TensorRT headline result

This is the main Jetson latency/accuracy tradeoff artifact.

Run:

```bash
python experiments/exp5_pareto.py --no-simulate --n 200
```

Primary artifact:

- `results/exp5_pareto.json`

The paper headline uses the Jetson TensorRT heavy-load row comparing:

- `safety2`
- `FIXED_MEDIUM`

### 6.2 Safety-aware policy comparison

Run:

```bash
python experiments/exp7_swas.py --no-simulate --n 100 --frames <KITTI_IMAGE_DIR>
```

Primary artifacts:

- `results/exp7_swas.json`
- `results/exp7_swas.csv`

### 6.3 Live KITTI VRU recall

Run:

```bash
python experiments/exp8_accuracy_per_tier.py \
  --dataset kitti \
  --images <KITTI_IMAGE_DIR> \
  --labels <KITTI_LABEL_DIR> \
  --max-images 300
```

Primary artifacts:

- `results/exp8_accuracy_kitti.json`
- `results/exp8_accuracy_kitti.csv`

### 6.4 Cross-device ONNX runtime comparison

Run on each ONNX host:

```bash
python experiments/exp9_multidevice.py \
  --device <DEVICE_NAME> \
  --backend onnx \
  --frames <KITTI_IMAGE_DIR> \
  --n 100
```

Then aggregate:

```bash
python experiments/exp9_aggregate.py
```

Primary artifacts:

- `results/multidevice/exp9_<device>_onnx_<timestamp>.json`
- `results/exp9_summary.json`

## 7. Full Suite Reproduction

### Windows / general ONNX path

```powershell
python scripts\calibrate.py --seconds 30 --apply

python experiments\complete_runall.py `
  --no-simulate `
  --n 100 `
  --transient-total-n 180 `
  --frames <KITTI_IMAGE_DIR> `
  --kitti-images <KITTI_IMAGE_DIR> `
  --kitti-labels <KITTI_LABEL_DIR> `
  --device <DEVICE_NAME> `
  --backend onnx
```

### Jetson path

See:

- `docs/RAMS_Jetson_Runbook.md`

## 8. Interpretation Notes

Not every experiment makes the same kind of claim:

- `exp5` and `exp7` are best interpreted as **within-setting policy comparisons**
  because they use tier-level accuracy proxies.
- `exp8` is the **live safety envelope** because it measures recall directly.
- `exp9` is the **cross-device runtime comparison**.

This distinction is important for honest reproduction and reporting.
