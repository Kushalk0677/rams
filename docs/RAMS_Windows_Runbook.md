# RAMS Windows CPU ONNX runbook

The current end-to-end operator guide is at the root of
[`RAMS_Windows_validation.zip`](../packages/RAMS_Windows_validation.zip). It
covers download, Python 3.12 setup, models, KITTI and COCO, calibration,
device inventory, smoke validation, and the phased full run.

Windows paper-facing evidence uses the CPU ONNX Runtime path. An installed
NVIDIA GPU may contribute telemetry, but it is not used as a Windows GPU
inference backend in this protocol. The shared evaluation rationale and
retained outputs are in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
