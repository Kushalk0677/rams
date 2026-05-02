# RAMS Complete Runall — Windows Runbook

This runbook adds a new orchestrator without replacing the old one.

Files added:
- `experiments/complete_runall.py`
- `COMPLETE_RUNALL_WINDOWS_RUNBOOK.md`

## What this runner does

It runs experiments 1 through 10 in separate Python processes, so one broken experiment does not poison the rest of the suite. It also writes a manifest to `results/complete_runall_manifest.json` showing exactly what ran, what was skipped, and what failed.

## Recommended Windows usage

### 1) Quick smoke test

```powershell
python experiments\complete_runall.py --simulate --n 5 --transient-total-n 40
```

### 2) Smoke test with a small KITTI subset

```powershell
python experiments\complete_runall.py `
  --simulate `
  --n 5 `
  --transient-total-n 40 `
  --frames C:\rams\data\kitti\images\val `
  --kitti-images C:\rams\data\kitti\images\val `
  --kitti-labels C:\rams\data\kitti\labels\val `
  --max-images 50 `
  --device spectre-i7-1165g7 `
  --backend onnx
```

### 3) Full real run on Windows CPU/ONNX

Run calibration first if you want thresholds in `configs/default.yaml` refreshed:

```powershell
python scripts\calibrate.py --seconds 30 --apply
```

Then run:

```powershell
python experiments\complete_runall.py `
  --no-simulate `
  --n 100 `
  --transient-total-n 180 `
  --frames C:\rams\data\kitti\images\val `
  --kitti-images C:\rams\data\kitti\images\val `
  --kitti-labels C:\rams\data\kitti\labels\val `
  --coco-images C:\rams\data\coco\images\val2017 `
  --coco-labels C:\rams\data\coco\labels\val2017 `
  --max-images 300 `
  --device spectre-i7-1165g7 `
  --backend onnx
```

## Notes

- `exp8` runs only when KITTI paths are provided.
- `exp9` runs only when `--frames` is provided.
- `exp10_kitti` runs when both an `exp8_accuracy_kitti.json` and a benchmark JSON exist.
- `exp10_coco` also runs when COCO paths are supplied and `exp8_accuracy_coco.json` exists.
- The old runner `experiments/run_all.py` is untouched.

## Outputs to check

- `results/complete_runall_manifest.json`
- `results/exp1_*` through `results/exp10_*`
- `results/rams_*.json` from `benchmark.run`
- `results/exp9_summary.json`

## Important fixes already wired in

- Config values from `configs/default.yaml` are now applied by the controller.
- `safety2` now receives post-inference detections properly.
- The duplicate `run_trial()` implementation in `experiments/utils.py` was removed.
- `exp8` now has a RAMS-wrapper fallback path for smoke testing when Ultralytics is unavailable.
- `exp6` now supports `--total-n` for shorter verification runs.
