# COCO mAP provenance — prior PyTorch evaluation (Kushal ask 5, provenance route)

Claimed: 2026-09-09. Author: Vesper (Hermes subagent run, RAMS TRT-label mission).
Route chosen: **provenance of the prior PyTorch COCO eval** (no fresh run) —
justification and full evidence chain below.

## The artifact being sourced

`soren@soren-edge:/home/soren/rams_sep2026/rams_validation/results/exp8_accuracy_coco.json`
(sha256 `493ab30044ed5da557a809eb619b5f9e99a2c9f54e69c284d0c8e39a0bfaceb9`,
mtime 2026-09-06 06:32:04 −0400)

| tier | imgsz | n_images | map50 | map50_95 | map_source |
|---|---|---|---|---|---|
| NANO | 320 | 5000 | 0.3182 | 0.2301 | ultralytics_val |
| SMALL | 416 | 5000 | 0.4702 | 0.3488 | ultralytics_val |
| MEDIUM | 640 | 5000 | 0.5797 | 0.4426 | ultralytics_val |

## How it was produced (recorded command, executed 2026-09-06 ~06:18–06:32 EDT)

Driving manifest:
`/home/soren/rams_sep2026/rams_validation/results/paper_all_20260906_093913_soren-edge.json`,
entry `tier_accuracy_context`, returncode 0, elapsed 856.75 s:

```
/usr/local/bin/python3 experiments/exp8_accuracy_per_tier.py \
  --dataset kitti --images /root/rams/data/kitti/images/val --labels /root/rams/data/kitti/labels/val \
  --also-coco --coco-images /root/rams/data/coco/images/val2017 --coco-labels /root/rams/data/coco/labels/val2017 \
  --require-coco-ultralytics-map
```

Manifest preflight block records device=soren-edge, python 3.10.12,
Linux-5.15.148-tegra-aarch64, run under `RAMS_BACKEND=tensorrt` env — the
COCO mAP numbers themselves come from the **ultralytics PyTorch path**:
`exp8_accuracy_per_tier.py` lines 240–253 load `YOLO(cfg["model"])` (`.pt`)
and line 351 calls `model.val(data=<generated exp8_coco_local.yaml>,
imgsz=<per-tier 320/416/640>, conf=0.25, iou=0.5)`; `map_source=ultralytics_val`
is written only on that branch (line 354), and `--require-coco-ultralytics-map`
makes any fallback to cached-profile mAP a hard error (lines 243–251, 356–360).
The TensorRT env var does not affect this branch: ultralytics `val()` on a
`.pt` weights file runs PyTorch.

Launcher + dataset mounts:
`/home/soren/rams_sep2026/run_in_container.sh` (mtime 2026-09-05 22:04) ran the
`rams:sep2026` image mounting
- images ← `/home/soren/rams_full/data/coco/val` (5000 .jpg; on-disk dir also
  carries AppleDouble `._*` files — 10001 entries total; ultralytics matched
  the 5000 jpgs),
- labels ← `/home/soren/rams_kushal_completion_20260811/datasets/coco/labels/val2017`
  (4952 .txt),
at container paths `/root/rams/data/coco/{images,labels}/val2017`.

## Identical-weights check (Kushal's condition)

Weights at eval time = the files in `rams_validation/` (sha256 recorded in
`/home/soren/rams_sep2026/corrected-20260908/model-sha256.txt` and reverified
2026-09-09):

- yolov8n.pt `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- yolov8s.pt `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`
- yolov8m.pt `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`

The 2026-09-08 corrected run and the TRT engines consume the same files
(`source/yolov8*.pt|onnx|engine` are symlinks to
`/original` = `rams_validation/`). Engine build:
`~/rams_sep2026/build_engines_in_container.sh` (trtexec --fp16 from the
matching .onnx). Eval script byte-identical between the eval-time tree and
the current patched tree (sha256
`3fbeac5055745f9b2f351ca7cc0040ab588b6297a7f5fcc5c601dbeceab99494` both).

## Qualifications (honest limits)

- mAP came from ultralytics `val()` (PyTorch, COCO80 names via the generated
  `rams_validation/results/exp8_coco_local.yaml`, standard COCO names list),
  not from the RAMS TRT wrapper — it characterizes the weights+tiers, not the
  TRT runtime. TRT class correctness is established separately by the
  2026-09-09 patch preflight (this mission).
- `n_images=5000` is COCO val2017 in that container view; 4952 label files —
  48 images legitimately unlabeled in COCO val (normal), matching
  ultralytics' unannotated-image handling (scored as negatives).
- Dataset dirs still on disk with the exact mounts recorded above; image-dir
  name/size manifest sha256 `0038ec3e…` (10001 entries incl. AppleDouble),
  labels manifest `9236fa31…` (4952). Per-file content digests were not
  recorded at eval time — this is manifest-level binding (names+sizes), not
  per-image hashes. Stated as-is: no per-image hash claim.
- If Kushal requires a *fresh* COCO eval re-executed today, it can be
  reproduced with the recorded command + launcher; not run here because the
  ask explicitly allowed the provenance route and the artifact predates the
  TRT-label patch scope (PyTorch results accepted as-is per his 09-08 review).
