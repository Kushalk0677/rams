# Historical Jetson results

These preserved April 25-26, 2026 results include CPU ONNX and TensorRT
evidence from the host recorded as `soren-edge`.

## Hardware identity unresolved

The supplied [deployment report](trt/rams-jetson-deployment-report.md)
names a Jetson Orin Nano, but also twice describes twelve CPU cores.
Those descriptions conflict: an Orin Nano has six CPU cores. The preserved
files inspected do not establish physical RAM capacity or a definitive
board identifier, so neither an Orin Nano 8GB nor an AGX Orin 64GB should
be asserted for these historical runs without confirmation from the operator.

The original report and result records are retained unchanged as supplied.
The verified AGX Orin 64GB identity of the September results does not, by
itself, establish which device produced these April results.
