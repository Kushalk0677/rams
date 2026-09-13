# Jetson Orin Nano CPU ONNX Results

Corrected RAMS evidence collected on 8 September 2026 on an NVIDIA Jetson AGX
Orin 64 GB. This route uses ONNX Runtime with `CPUExecutionProvider`; it is
not CUDA ONNX.

The run completed calibration, all five runtime profiles, moderate and heavy
Pareto runs, policy-level KITTI accuracy, and VRU-retention sensitivity. Every
runtime profile has 8 controller policies, including fixed NANO, SMALL, and
MEDIUM baselines, over 10 blocks of 200 real KITTI replay frames.

`calibration/` preserves the applied calibration and snapshots. Each
`runtime*` directory contains raw frame records, manifests, summaries, and
measured rail-energy windows. `pareto_*`, `policy_accuracy`, and `retention`
contain their corresponding raw records and summaries. `metadata/` contains
the suite manifest, source-patch verification receipts, backend preflight, and
telemetry preflight.

Energy is measured by integrating the recorded Jetson rails over each replay
window. `VDD_GPU_SOC` is a shared GPU and SoC rail, and none of the retained
rails is a whole-board measurement. Do not describe these values as total
device energy consumption.

This corrected bundle does not include a new COCO mAP run. CPU ONNX accuracy
must not be presented as directly equivalent to the PyTorch route without a
matched detector-evaluation protocol.

Source archive SHA-256: `9dbbb4c2bcda488e1d36d5c51760aee34b5ad0f4161c54c2153ea8268cd31d02`.
