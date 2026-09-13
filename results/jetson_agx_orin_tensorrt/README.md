# Jetson AGX Orin TensorRT Results

Corrected RAMS evidence collected on 9 September 2026 on an NVIDIA Jetson AGX
Orin 64 GB using TensorRT FP16 engines.

The run completed calibration, all five runtime profiles, moderate and heavy
Pareto runs, policy-level KITTI accuracy, and VRU-retention sensitivity. Every
runtime profile has 8 controller policies, including fixed NANO, SMALL, and
MEDIUM baselines, over 10 blocks of 200 real KITTI replay frames.

The TensorRT wrapper maps placeholder engine labels such as `class2` and
`class5` to the canonical COCO labels used by the other backends. The metadata
directory includes the tested source patch, patch hashes, source-application
receipt, backend preflight, telemetry preflight, and suite manifest. The
preflight rejects noncanonical TensorRT labels before a run begins.

`calibration/` preserves the applied calibration and snapshots. Each
`runtime*` directory contains raw frame records, manifests, summaries, and
measured rail-energy windows. `pareto_*`, `policy_accuracy`, and `retention`
contain their corresponding raw records and summaries.

Energy is measured by integrating the recorded Jetson rails over each replay
window. `VDD_GPU_SOC` is a shared GPU and SoC rail, and none of the retained
rails is a whole-board measurement. Do not describe these values as total
device energy consumption.

The COCO mAP figures in the PyTorch route characterize the shared checkpoints,
not the TensorRT wrapper.

Source patch commit: `b3ada2b0d7342cf8341f5ae36c07754a5819db9a`.

Acknowledgment: Evan Leri prepared the TensorRT label-contract patch and the
corrected on-device validation evidence preserved here.
