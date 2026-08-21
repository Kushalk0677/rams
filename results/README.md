# Curated Results

`old/` contains the historical results retained from the earlier RAMS
evaluation: i7-1165G7, i7-13700F, Raspberry Pi 5, and Jetson Orin. They are
preserved for traceability and must not be combined with revised-protocol
results.

`m4_air/` contains the browsable Apple M4 MacBook Air CPU-ONNX evidence
received on 3 August 2026: manifests, calibration, raw records, figures,
configuration and provenance. It is a real KITTI replay evaluation. Its energy
fields are TDP-based estimates, not physical power
measurements.

`i7_11800H/` contains the completed Intel Core i7-11800H Windows CPU-ONNX
evidence received on 20 August 2026. It uses the revised
`process_steady_v3` load protocol on a 16-logical-core laptop with an RTX 3050
Ti present but not used for inference. It includes calibration, all phased
runtime results, measured COCO validation, KITTI policy metrics, and retention
analysis, raw records, manifests, and figures. Its energy fields are
telemetry-conditioned estimates, not physical power measurements. Do not
combine it with archived `thread_steady_v1` Windows results.

The top level of each current device directory contains only calibration,
manifests, raw records, tables, and figures. Device configuration and
provenance are under `metadata/`. The `m4_air/excluded/` directory holds the
explicitly excluded smoke and invalid-map artifacts and is not paper evidence.

New runs created by the current workflow should be stored in a separate
device- and protocol-labelled folder. Keep the manifest, raw records,
calibration, and summary together for every reported result.
