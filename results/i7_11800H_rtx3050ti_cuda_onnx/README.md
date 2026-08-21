# Intel Core i7-11800H with RTX 3050 Ti, CUDA ONNX evidence

This directory contains supplied evidence for a Windows 11 laptop with an
Intel Core i7-11800H and an NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM).
Inference used ONNX Runtime 1.26.0 with the requested `onnx_cuda` backend and
the recorded provider chain `CUDAExecutionProvider;CPUExecutionProvider`.

This is a separate backend route from [`../i7_11800H`](../i7_11800H), which is
CPU ONNX evidence from the same processor family. Do not pool their latency,
confidence intervals, or energy estimates.

## Evidence layout

- `calibration/`: recorded load calibration.
- `manifests/`: replay, warm-up, software, and phase manifests.
- `records/`: raw runtime records, policy results, COCO evaluation records, and
  the supplied KITTI mapped-COCOeval summary and protocol records.
- `figures/`: supplied plots and tables.
- `metadata/`: device inventory, provenance, and the TDP energy model.

## KITTI accuracy

The supplied follow-up records evaluated 1,500 KITTI replay images using a
mapped COCOeval bounding-box protocol. KITTI categories are mapped to `car`,
`person`, and `bicycle`; the full mappings and IoU range are in
`records/exp8_kitti_map_protocol_*.json`.

| Tier | Input size | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|
| NANO | 320 | 0.0470 | 0.0256 |
| SMALL | 416 | 0.0700 | 0.0370 |
| MEDIUM | 640 | 0.0736 | 0.0400 |

These fresh fields replace the earlier cached KITTI tier references for this
device. The compact follow-up archive supplies the metric summary and protocol
but not the raw detection and ground-truth JSON named in those protocol files.
Retain the source archive with the repository evidence if independent metric
recomputation is needed.

## Energy interpretation

`metadata/energy_profile_windows_rtx3050.json` specifies a TDP model using a
67 W GPU cap, a 20 W dynamic CPU assumption, a 2 W memory assumption, and a
6 W idle baseline. Any energy values derived from it are
telemetry-conditioned estimates, not physical power or battery measurements.

See [`metadata/PROVENANCE.md`](metadata/PROVENANCE.md) for the complete scope,
source archives, and limitations.
