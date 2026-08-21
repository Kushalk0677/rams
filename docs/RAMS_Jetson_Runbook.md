# RAMS Jetson TensorRT runbook

The current end-to-end operator guide is
[packaging/jetson/README.md](../packaging/jetson/README.md). The same guide is
at the root of [`RAMS_Jetson_validation.zip`](../packages/RAMS_Jetson_validation.zip).

It covers JetPack setup, device-specific TensorRT engine creation, the
mandatory real-inference preflight, KITTI and COCO preparation, calibration,
and the full phased run. Use that guide instead of historical Jetson runbooks.

Do not reuse TensorRT engines across Jetson devices, JetPack releases, or
TensorRT versions.
