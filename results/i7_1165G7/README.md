# Intel Core i7-1165G7 Windows CPU-ONNX Results

This directory contains the complete RAMS paper suite collected on 13-14
September 2026 on `Kushals-Spectre-x360-AW2002TU`. The route used ONNX Runtime
CPU inference with Python 3.12.10 and the revised `process_steady_v3` load
injector. It is current evidence and is separate from the historical
`results/old/i7_1165G7/` material.

## Coverage

The suite completed calibration, idle, light, moderate, heavy, and burst
runtime replay; moderate and heavy Pareto runs; KITTI and COCO tier evaluation;
policy-level KITTI evaluation; and VRU-retention sensitivity. Every stage has a
successful entry in `manifests/paper_all_20260913_234154_i7_1165G7_cpu_onnx_process_v3.json`.

Calibration applied `lo_thresh = 0.458` and `hi_thresh = 0.761`. Its report
and before/after configuration snapshots are in `calibration/`.

## Directory layout

- `calibration/`: replay calibration report and configuration snapshots.
- `manifests/`: full-suite, runtime-block, and policy-accuracy manifests.
- `records/`: raw runtime records and summary JSON/CSV files for Pareto, tier,
  policy, and retention evaluations.
- `figures/`: the generated Pareto and tier-accuracy figures.
- `provenance/`: model checksums and the device-specific TDP profile.

## Accuracy evidence

COCO val2017 was evaluated over 5,000 images using `ultralytics_val`:
NANO 0.3181/0.2300, SMALL 0.4700/0.3489, and MEDIUM 0.5798/0.4426 for
mAP@0.50/mAP@0.50:0.95.

The KITTI tier records contain fresh detector-derived VRU metrics, but their
displayed mAP fields have `map_source: cached_profile`. Do not present those
KITTI mAP fields as newly measured mAP. The separate RTX 3050 Ti CUDA-ONNX
directory contains the mapped-KITTI COCOeval mAP evidence.

## Energy scope

`provenance/energy_profile_windows_i7_1165g7_tdp_up_28w.json` is a constant
28 W TDP ceiling model. Its energy values are estimates based on end-to-end
latency, not physical battery or wall-power measurements.
