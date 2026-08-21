# RTX 3050 Ti CUDA ONNX result provenance

This directory is a normalized copy of the complete supplied archive
`RAMS_RTX3050_CUDA_ONNX_results_20260821.zip`. Every record in that archive is
retained here. Files are organized by purpose only: `calibration/`,
`manifests/`, `records/`, `figures/`, and `metadata/`.

## Device and backend

- Device: Intel Core i7-11800H laptop with NVIDIA GeForce RTX 3050 Ti Laptop
  GPU, 4 GB VRAM.
- Operating system: Windows 11, build 26100.
- Python: 3.12.9.
- ONNX Runtime: 1.26.0.
- Requested backend: `onnx_cuda`.
- Recorded execution providers: `CUDAExecutionProvider;CPUExecutionProvider`.
- GPU telemetry source: NVML, including GPU memory fraction, temperature, and
  clock fields in the raw records.

This is distinct from `results/i7_11800H/`, which is Windows CPU ONNX evidence.
Do not pool latency, confidence intervals, or energy estimates across the two
backend routes.

## Included evidence

The supplied archive contains non-simulated, non-smoke manifests for
calibration, `runtime1`, `runtime2`, `runtime3`, `runtime4`, accuracy, and
retention. Each runtime phase records 10 paired blocks of 200 frames per method
and load setting using `process_steady_v3` or `process_isolated_burst_v2`.

It also contains a separate smoke `paper_all` record. That file is retained for
setup traceability but is not paper evidence. Use the individual non-smoke
phase manifests instead.

COCO was evaluated on all 5,000 validation images. Its result files report
`map_source: onnx_cuda_cocoeval`. A subsequent supplied KITTI accuracy run
evaluated 1,500 real replay images through the same CUDA ONNX route. It reports
`map_source: kitti_native_mapped_cocoeval` using mapped `car`, `person`, and
`bicycle` categories with COCOeval bounding-box IoU thresholds from 0.50 to
0.95 in 0.05 increments. Its results are retained in `records/`:

| Tier | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|
| NANO | 0.0470 | 0.0256 |
| SMALL | 0.0700 | 0.0370 |
| MEDIUM | 0.0736 | 0.0400 |

The protocol JSON records the class mappings and metric configuration. The
supplied follow-up archive does not include the raw detection and ground-truth
JSON files referenced by those protocol records, so preserve the source archive
with this repository evidence. Policy-level KITTI recall, precision, F1, and
false-negative rate are also recorded from the real replay.

## Evidence limits

`metadata/energy_profile_windows_rtx3050.json` records the supplied TDP energy
model: a 67 W NVIDIA GPU cap, a 20 W dynamic CPU assumption, a 2 W dynamic
memory assumption, and a 6 W idle baseline. The stored energy values are
telemetry-conditioned estimates from this model, not physical energy
measurements from an external meter.

The current public Windows validation package documents CPU ONNX Runtime only.
This CUDA ONNX result was produced by a separate supplied Windows CUDA path.
It is retained as device evidence, but it does not by itself make the current
Windows package a reproducible CUDA package. Code and package support must be
integrated and validated separately before presenting this route as current
package output.
