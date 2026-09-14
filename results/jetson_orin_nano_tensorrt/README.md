# Jetson Orin Nano TensorRT Results

Corrected RAMS evidence collected on an NVIDIA Jetson Orin Nano using TensorRT
FP16 engines.

The retained evidence covers calibration, all five runtime profiles, moderate
and heavy Pareto runs, policy-level KITTI accuracy, VRU-retention sensitivity,
and fresh per-tier accuracy evaluation. Every runtime profile has 8 controller
policies, including fixed NANO, SMALL, and MEDIUM baselines, over 10 blocks of
200 real KITTI replay frames.

The TensorRT wrapper maps placeholder engine labels such as `class2` and
`class5` to the canonical COCO labels used by the other backends. The metadata
directory includes the tested source patch, patch hashes, source-application
receipt, backend preflight, telemetry preflight, and suite manifest. The
preflight rejects noncanonical TensorRT labels before a run begins.

`tier_accuracy/` contains the missing fresh detector evaluation collected on
13 September 2026: all three tiers over 1,500 KITTI and 5,000 COCO validation
images. The canonical TensorRT-label preflight passed before the evaluations.
It preserves predictions, per-image scores, metric files, figures, preflight
logs, and source and dataset provenance. The mapped KITTI scores are COCO-style
AP under the documented class mapping, not official KITTI difficulty-stratified
AP. See [`tier_accuracy/README.md`](tier_accuracy/README.md).

`calibration/` preserves the applied calibration and snapshots. Each
`runtime*` directory contains raw frame records, manifests, summaries, and
measured rail-energy windows. `pareto_*`, `policy_accuracy`, and `retention`
contain their corresponding raw records and summaries.

Energy is measured by integrating the recorded Jetson rails over each replay
window. `VDD_GPU_SOC` is a shared GPU and SoC rail, and none of the retained
rails is a whole-board measurement. Do not describe these values as total
device energy consumption.

Acknowledgment: Evan Leri prepared the TensorRT label-contract patch and the
corrected on-device validation evidence preserved here, including the fresh
per-tier TensorRT evaluation.
