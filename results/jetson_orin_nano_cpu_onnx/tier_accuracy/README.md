# Fresh Per-tier Accuracy: Jetson Orin Nano CPU ONNX

This directory contains the completed CPU ONNX tier-accuracy stage collected
on 13 September 2026. It evaluates NANO, SMALL, and MEDIUM over the complete
1,500-image KITTI replay set and the 5,000-image COCO val2017 set. All 6
evaluations used real inference with ONNX Runtime `CPUExecutionProvider`, a
confidence floor of 0.25, and class-aware NMS at IoU 0.70.

| Dataset | Tier | Images | mAP@0.50 | mAP@0.50:0.95 | VRU recall |
|---|---:|---:|---:|---:|---:|
| KITTI | NANO | 1,500 | 0.1592 | 0.0783 | 0.2294 |
| KITTI | SMALL | 1,500 | 0.2400 | 0.1236 | 0.3627 |
| KITTI | MEDIUM | 1,500 | 0.3545 | 0.1911 | 0.4984 |
| COCO | NANO | 5,000 | 0.3145 | 0.2292 | 0.5022 |
| COCO | SMALL | 5,000 | 0.4652 | 0.3482 | 0.6514 |
| COCO | MEDIUM | 5,000 | 0.5806 | 0.4457 | 0.7721 |

`records/` contains one directory per dataset and tier. Each retains the raw
`predictions.jsonl`, `per-image.csv`, `metrics.json`, and `metrics.csv` files.
`figures/` contains the dataset-tier plots. `preflight/` contains dataset,
backend, telemetry, and scoring checks. `logs/` contains the six evaluator
logs. `provenance/` retains the complete-matrix receipt, source and model
hashes, software versions, scoring protocol, and the evaluator's AP helpers.

KITTI ground truth is mapped to the COCO label space: Car/Van to car,
Pedestrian/Person_sitting to person, Cyclist to bicycle, Truck to truck, and
Tram to train. DontCare and Misc regions suppress overlapping predictions under
the recorded rule. These are mapped COCO-style AP values, not official KITTI
difficulty-stratified AP. The archive's full verification passed all 12
backend-dataset-tier combinations and all 39,000 evaluated frames.

The evidence is tied to the preserved source snapshot and checksums in
`provenance/`. The tier evaluator and suite driver match their accepted source
hashes; use the records and provenance here when reporting these values.
