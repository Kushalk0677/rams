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
`map_source: onnx_cuda_cocoeval`. The KITTI `map50` and `map5095` fields report
`map_source: cached_profile`; they are cached tier references, not newly
measured KITTI mAP. Policy-level KITTI recall, precision, F1, and false-negative
rate are recorded from the real replay.

## Evidence limits

The archive does not contain the referenced
`configs/energy_profile_windows_rtx3050.json`. Its stored energy values remain
telemetry-conditioned estimates, not physical energy measurements. Do not make
an energy claim from this directory until that input profile is recovered and
retained with the evidence.

The current public Windows validation package documents CPU ONNX Runtime only.
This CUDA ONNX result was produced by a separate supplied Windows CUDA path.
It is retained as device evidence, but it does not by itself make the current
Windows package a reproducible CUDA package. Code and package support must be
integrated and validated separately before presenting this route as current
package output.
