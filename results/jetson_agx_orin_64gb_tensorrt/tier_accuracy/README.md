# Fresh Per-tier Accuracy: Jetson AGX Orin 64GB TensorRT

This directory contains the completed TensorRT tier-accuracy stage collected
on 13 September 2026. It evaluates NANO, SMALL, and MEDIUM over the complete
1,500-image KITTI replay set and the 5,000-image COCO val2017 set. All 6
evaluations used real TensorRT FP16 inference, a confidence floor of 0.25, and
class-aware NMS at IoU 0.70.

| Dataset | Tier | Images | mAP@0.50 | mAP@0.50:0.95 | VRU recall |
|---|---:|---:|---:|---:|
| KITTI | NANO | 1,500 | 0.1590 | 0.0785 | 0.2303 |
| KITTI | SMALL | 1,500 | 0.2401 | 0.1240 | 0.3627 |
| KITTI | MEDIUM | 1,500 | 0.3550 | 0.1912 | 0.4992 |
| COCO | NANO | 5,000 | 0.3141 | 0.2289 | 0.5022 |
| COCO | SMALL | 5,000 | 0.4655 | 0.3483 | 0.6510 |
| COCO | MEDIUM | 5,000 | 0.5806 | 0.4456 | 0.7722 |

`records/` contains one directory per dataset and tier. Each retains the raw
`predictions.jsonl`, `per-image.csv`, `metrics.json`, and `metrics.csv` files.
`figures/` contains the dataset-tier plots. `preflight/` retains the canonical
TensorRT label preflight that passed before evaluation, plus dataset, telemetry,
and scoring checks. `logs/` contains the six evaluator logs. `provenance/`
retains the complete-matrix receipt, source and model hashes, software versions,
scoring protocol, and the evaluator's AP helpers.

KITTI ground truth is mapped to the COCO label space: Car/Van to car,
Pedestrian/Person_sitting to person, Cyclist to bicycle, Truck to truck, and
Tram to train. DontCare and Misc regions suppress overlapping predictions under
the recorded rule. These are mapped COCO-style AP values, not official KITTI
difficulty-stratified AP. The archive's full verification passed all 12
backend-dataset-tier combinations and all 39,000 evaluated frames.

The evidence is tied to the preserved source snapshot and checksums in
`provenance/`. The tier evaluator and suite driver match their accepted source
hashes; use the records and provenance here when reporting these values.
