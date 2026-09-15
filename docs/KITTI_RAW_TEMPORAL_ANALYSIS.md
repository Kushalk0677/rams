# KITTI Raw Temporal Preflight

`experiments/exp13_temporal_lead_time.py` prepares the fixed ground-truth
transition manifest for the i7-1165G7 ONNX temporal follow-up. It does not run
detector inference and does not create paper results.

The local raw image archive is expected at `D:\data\kitti_raw`. Each selected
drive needs RGB frames, `image_02/timestamps.txt`, and its official
`tracklet_labels.xml`. KITTI released tracklets only for a subset of raw
drives. Three later-date drives in the local image archive have no matching
official label archive and must be excluded, not treated as missing data.

Keep downloaded labels separate if preferred. Their directory must mirror the
raw drive names, for example:

```text
D:\data\kitti_raw_tracklets\2011_09_26\2011_09_26_drive_0001_sync\tracklet_labels.xml
```

Run an audit before any replay:

```powershell
cd C:\rams\rams_v2
.\.venv\Scripts\python.exe experiments\exp13_temporal_lead_time.py `
  --raw-root D:\data\kitti_raw `
  --tracklet-root D:\data\kitti_raw_tracklets `
  --labelled-only `
  --output runs\temporal_raw\kitti_raw_audit.json
```

Use `--smoke` to inspect only the first drive. `--labelled-only` selects only
local drives for which official tracklets exist. Add
`--prepare` to emit the transition/control manifest:

```powershell
.\.venv\Scripts\python.exe experiments\exp13_temporal_lead_time.py `
  --raw-root D:\data\kitti_raw `
  --tracklet-root D:\data\kitti_raw_tracklets `
  --labelled-only `
  --output runs\temporal_raw\kitti_raw_transition_manifest.json `
  --prepare --window-s 0.5 --min-events 30
```

The manifest accepts only pedestrian/cyclist entries preceded by a complete
VRU-free window. It records the camera-derived frame count for 0.5 seconds and
deterministic same-sequence control candidates. Treat fewer than 30 accepted
events as underpowered. This evaluates pre-entry retention carry-over, not VRU
prediction or safety.
