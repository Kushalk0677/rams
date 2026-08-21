# Curated Results

`old/` contains the historical results retained from the earlier RAMS
evaluation: i7-1165G7, i7-13700F, Raspberry Pi 5, and Jetson Orin. They are
preserved for traceability and must not be combined with revised-protocol
results.

`macos_cpu_onnx/` contains the browsable MacBook Air CPU-ONNX evidence
received on 3 August 2026: manifests, calibration, raw records, figures,
configuration, provenance, and supporting scripts. It is a real KITTI replay
evaluation. Its energy fields are TDP-based estimates, not physical power
measurements.

`windows_rtx3050ti_cpu_onnx_process_v3/` contains the completed Windows
CPU-ONNX evidence received on 20 August 2026. It uses the revised
`process_steady_v3` load protocol on a 16-logical-core laptop with an RTX 3050
Ti present but not used for inference. It includes calibration, all phased
runtime results, measured COCO validation, KITTI policy metrics, and retention
analysis, raw records, manifests, and figures. Its energy fields are
telemetry-conditioned estimates, not physical power measurements. Do not
combine it with archived `thread_steady_v1` Windows results.

New runs created by the current workflow should be stored in a separate
device- and protocol-labelled folder. Keep the manifest, raw records,
calibration, and summary together for every reported result.
