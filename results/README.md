# Curated Results

`old/` contains the historical results retained from the earlier RAMS
evaluation: i7-1165G7, i7-13700F, Raspberry Pi 5, and Jetson Orin. They are
preserved for traceability and must not be combined with revised-protocol
results.

`m4_air/` contains the browsable Apple M4 MacBook Air with 16 GB unified-memory CPU-ONNX evidence
received on 3 August 2026: manifests, calibration, raw records, figures,
configuration and provenance. It is a real KITTI replay evaluation. Its energy
fields are TDP-based estimates, not physical power
measurements. Its provenance records the historical execution context for that
evidence; it is not a setup guide. Use the current macOS package for a new run.

`i7_1165G7/` contains the complete revised-protocol Intel Core i7-1165G7
Windows CPU-ONNX suite collected on 13-14 September 2026. It uses
`process_steady_v3` for steady profiles and `process_isolated_burst_v2` for
the burst profile. It includes calibration, all runtime and Pareto stages,
COCO tier evaluation, policy-level KITTI metrics, retention sensitivity, raw
records, manifests, figures, model checksums, and the TDP-profile provenance.
The COCO mAP fields are fresh `ultralytics_val` results; the KITTI tier mAP
fields are cached-profile context and are not new mAP evidence.

`i7_11800H/` contains the completed Intel Core i7-11800H Windows CPU-ONNX
evidence received on 20 August 2026. It uses the revised
`process_steady_v3` load protocol on a 16-logical-core laptop with an RTX 3050
Ti present but not used for inference. It includes calibration, all phased
runtime results, measured COCO validation, KITTI policy metrics, and retention
analysis, raw records, manifests, and figures. Its energy fields are
telemetry-conditioned estimates, not physical power measurements. Do not
combine it with archived `thread_steady_v1` Windows results.

`i7_11800H_rtx3050ti_cuda_onnx/` contains the separate RTX 3050 Ti Laptop GPU
CUDA-ONNX evidence received on 21 August 2026. It preserves the complete
supplied archives in normalized form: calibration, manifests, records, figures,
device inventory, mapped KITTI COCOeval protocol records, and a TDP energy
profile. The 1,500-frame mapped KITTI evaluation reports fresh mAP@0.50 / mAP@
0.50:0.95 of 0.0470 / 0.0256 for NANO, 0.0700 / 0.0370 for SMALL, and 0.0736 /
0.0400 for MEDIUM. It is a distinct backend route from `i7_11800H/`; do not
pool their latency, confidence intervals, or energy estimates. The supplied
energy profile supports TDP-model estimates only, not physical energy claims.

The top level of each current device directory contains only calibration,
manifests, raw records, tables, and figures. Device configuration and
provenance are under `metadata/`. The `m4_air/excluded/` directory holds the
explicitly excluded smoke and invalid-map artifacts and is not paper evidence.

New runs created by the current workflow should be stored in a separate
device- and protocol-labelled folder. Keep the manifest, raw records,
calibration, and summary together for every reported result.


`jetson_agx_orin_64gb_cpu_onnx/`, `jetson_agx_orin_64gb_pytorch/`, and
`jetson_agx_orin_64gb_tensorrt/` contain corrected Jetson AGX Orin 64GB evidence. Each
route includes calibration, all five runtime profiles, fixed-tier baselines,
policy accuracy, retention, and measured per-rail replay-window energy
records. The rail records are not total board energy measurements. CPU ONNX
and TensorRT now also contain a fresh 13 September 2026 per-tier detector
matrix over 1,500 KITTI and 5,000 COCO validation images. TensorRT maps engine
class IDs to canonical COCO labels and its preflight passed before the
evaluation. The mapped KITTI values are COCO-style AP under the documented
mapping, not official KITTI difficulty-stratified AP.

Hardware identification: all three September Jetson routes record 62,841 MB
of RAM, twelve CPU entries, and the AGX Orin power rails in
`metadata/telemetry_preflight.json`. These identify an AGX Orin 64GB; the
previous `jetson_orin_nano_*` directory names were incorrect. Raw evidence
has been preserved unchanged. NVIDIA documents the rail mapping in its
[Jetson power-monitor guide](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling).

The April results under `old/jetson_orin/` have an unresolved hardware
identity. Their supplied report says Orin Nano but also describes twelve
CPU cores. See the [historical hardware note](old/jetson_orin/README.md).
