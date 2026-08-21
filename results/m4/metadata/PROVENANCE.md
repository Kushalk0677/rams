# RAMS Experimental Results — Provenance

Generated 2026-08-03. Read this before using any file in this archive.

---

## Hardware and software

| | |
|---|---|
| Device | MacBook Air, Apple Silicon (arm64) |
| OS | macOS 26.5.1 |
| Python | 3.12.13 (Clang 21.0.0) |
| Inference backend | ONNX Runtime 1.28.0, `CPUExecutionProvider` |
| Models | YOLOv8n/s/m, COCO-pretrained, exported to ONNX opset 12 |
| Tier input sizes | NANO 320 · SMALL 416 · MEDIUM 640 |
| Power condition | Plugged in, Low Power Mode off |

Measured single-frame inference latency at preflight:
NANO 5.1 ms · SMALL 51.8 ms · MEDIUM 97.0 ms.

`CoreMLExecutionProvider` was available but **not** used. Every number here is CPU ONNX.
Do not pool these with any Core ML run.

---

## Datasets

**KITTI** — 2D object detection, left colour images. Validation split is the fixed
1,500-frame range (indices 5981–7480 of the sorted 7,481-image training set).
1,500 images / 1,500 labels, every image paired.

**COCO** — val2017, 5,000 images with 4,952 label files (images without objects have no
label file, which is expected).

---

## Run structure

* Calibration ran first and set `lo_thresh 0.43 / hi_thresh 0.68`, up from the shipped
  0.513/0.763. Config snapshots before and after are in `calibration_snapshots/`.
* `configs/default.yaml` was **not** modified after calibration. All later phases share
  the same thresholds.
* Runtime phases: 5 policies × 5 load profiles = 25 groups, n = 2,000 per group
  (200 frames × 10 blocks). Confidence intervals are block bootstrap, unit = frame.
* Phases ran sequentially 12:05–18:06 on 2026-08-03.

Verified across every retained file: no record has `simulated: true`, no record has
`smoke: true`, and every runtime group reports `backends: ["onnx"]`.

---

## Excluded directories — do not use

**`smoke_excluded/`** — a setup-validation run (`smoke: true`, n = 5, blocks = 1).
Real inference, but the sample size is meaningless. The suite itself states that
paper-facing outputs are valid only when `smoke=false` and `simulated=false`.

**`invalid_kitti_map/`** — the original KITTI mAP figure, produced before the fix below.
Superseded.

---

## Known defect, found and fixed

The first accuracy run reported KITTI mAP50 of 0.0173 / 0.0188 / 0.0346 — near zero.
Cause: `experiments/exp8_accuracy_per_tier.py` passed the hardcoded string `"kitti.yaml"`
to `model.val()`, which resolves to Ultralytics' bundled KITTI config. That pointed at a
different dataset and scored a COCO-pretrained 80-class model against KITTI's 9-class
taxonomy. The result measured a class-index mismatch, not detector accuracy.

Fix: KITTI labels were converted to YOLO format carrying COCO class indices
(`build_kitti_yolo.py`), a local dataset YAML was generated, and exp8 was patched to read
`$RAMS_KITTI_YAML` (`patch_exp8_kitti.py`). Only the accuracy phase was re-run; runtime
results were left untouched to avoid mixing thermal conditions.

Corrected KITTI mAP50: 0.1878 / 0.2893 / 0.4316. The VRU fields are byte-identical
before and after, confirming the patch affected only the mAP path.

KITTI class mapping used: Car and Van → car · Truck → truck · Pedestrian and
Person_sitting → person · **Cyclist → person** · Tram → train · Misc and DontCare dropped.

---

## Interpretation limits — carry these into the paper

1. **KITTI accuracy is zero-shot cross-dataset.** A COCO-pretrained detector evaluated on
   KITTI. Not comparable to the official KITTI benchmark, which uses KITTI-trained models
   and its own protocol.

2. **DontCare regions are not ignore regions.** KITTI's protocol ignores detections falling
   in DontCare zones; this package has no such support. Detections there count as false
   positives, which depresses precision and mAP. Visible at MEDIUM tier, where VRU precision
   falls to 0.288 under 1,718 false positives, making VRU F1 non-monotonic across tiers
   (0.342 / 0.461 / 0.389) even though recall rises monotonically (0.246 / 0.419 / 0.596).
   **Use recall for tier and policy comparisons. Do not report KITTI precision or F1 as
   detector performance.**

3. **Energy is a telemetry-derived estimate, not a physical measurement.** Computed from
   datasheet values and CPU utilisation, with `gpu_dynamic_w = 0` because the package has
   no portable Apple GPU-utilisation source. Treat as a relative comparison between
   policies, never as absolute joules.

4. **Thermal drift.** Phases ran back to back over roughly six hours on a fanless MacBook
   Air. Later phases ran on a warmer machine than earlier ones.

5. **Cyclist → person is a judgement call.** KITTI's Cyclist box covers rider plus bicycle;
   COCO separates them. This affects VRU figures and should be stated explicitly.

---

## Headline result

Under moderate load, moving from the `predictive` policy to `safety2` raises VRU rate from
0.252 to 0.293 (+16% relative) at 2.4× the estimated energy, rising to 3.4× under burst load.
The ordering holds across all five load profiles.

---

## Reproduction

```bash
python preflight_check.py --backend onnx \
  --energy-profile configs/energy_profile_macos.json --estimate
```

Preflight must pass before any phase. It fails loudly if the controller falls back to
simulation — the failure mode this package does not otherwise surface, since an unset
`RAMS_BACKEND` silently yields synthetic Gaussian latencies instead of raising.

Phase commands are in `RAMS_macOS_Runbook.md` §7. For the KITTI accuracy phase, export
`RAMS_KITTI_YAML` to the generated local YAML first.
