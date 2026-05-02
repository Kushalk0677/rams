# RAMS — Full Experiment Runbook 


**Device**: HP Spectre x360, Intel Core i7-1165G7 (4C/8T, Tiger Lake, Intel Iris Xe)  
**OS**: Windows 10/11  
**Datasets**: KITTI object detection + COCO val2017  
**Experiments**: 1–10 (all)

---

## Pre-flight checklist

| Requirement | Notes |
|---|---|
| Python 3.10–3.12 | 3.13/3.14 works but has no pre-built torch wheels yet |
| ~25 GB free disk | KITTI ~12 GB, COCO val2017 ~6 GB, models ~300 MB |
| 16 GB RAM recommended | Three YOLO tiers warm simultaneously; MEDIUM needs ~1.5 GB |
| Laptop plugged in | Tiger Lake throttles aggressively on battery; plugged = full 28 W TDP |
| Windows long-path support | Enable once (see below) to avoid path-length errors |

**Enable long paths once (run PowerShell as Administrator):**
```powershell
Set-ItemProperty `
  -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

---

## Phase 1 — Environment setup

```powershell
# Unzip the repo somewhere convenient
Expand-Archive -Path rams_with_experiments.zip -DestinationPath C:\rams
cd C:\rams\rams_full

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Core deps (from requirements.txt)
pip install psutil pyyaml numpy

# Real YOLOv8 inference (downloads yolov8n/s/m.pt automatically on first run)
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ONNX Runtime for faster CPU inference (uses all 8 threads)
pip install onnxruntime

# Plotting / experiments
pip install matplotlib opencv-python

# Install the rams package itself in editable mode
pip install -e .
```

> **Why `--index-url .../cpu`?** The i7-1165G7 has no CUDA. The CPU-only
> torch wheel is ~200 MB instead of ~2.5 GB, and installs in seconds.

### Export ONNX models (do this once — much faster inference than .pt)

```powershell
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)
YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12)
YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)
"
```

This produces `yolov8n.onnx`, `yolov8s.onnx`, `yolov8m.onnx` in the working
directory. RAMS automatically prefers ONNX over PyTorch if the `.onnx` file
exists alongside the `.pt`.

---

## Phase 2 — Dataset download

### 2a. KITTI object detection dataset

KITTI requires **free registration** at:
> https://www.cvlibs.net/datasets/kitti/user_login.php

After registering, download from the **2D Object Detection** page:
> https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

| File | Size | Purpose |
|---|---|---|
| Left color images of object data set | ~12 GB | Images for all 7481 frames |
| Training labels of object data set | ~5 MB | KITTI-format .txt labels |

Extract into:
```
C:\rams\data\kitti\
  images\
    training\image_2\     ← 000000.png … 007480.png
  labels\
    training\label_2\     ← 000000.txt … 007480.txt
```

Create a val split (last 1500 frames — a common community split):
```powershell
# Create val directories
mkdir C:\rams\data\kitti\images\val
mkdir C:\rams\data\kitti\labels\val

# Copy frames 5981–7480 as val (1500 images)
$imgs = Get-ChildItem C:\rams\data\kitti\images\training\image_2\*.png | Sort-Object Name
$imgs[5981..7480] | ForEach-Object {
    Copy-Item $_.FullName C:\rams\data\kitti\images\val\
}
$lbls = Get-ChildItem C:\rams\data\kitti\labels\training\label_2\*.txt | Sort-Object Name
$lbls[5981..7480] | ForEach-Object {
    Copy-Item $_.FullName C:\rams\data\kitti\labels\val\
}
```

Final structure expected by RAMS:
```
C:\rams\data\kitti\images\val\   ← 1500 × .png
C:\rams\data\kitti\labels\val\   ← 1500 × .txt  (KITTI format)
```

### 2b. COCO val2017

```powershell
# Images (~6 GB)
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" `
    -OutFile C:\rams\data\coco_val2017_images.zip
Expand-Archive C:\rams\data\coco_val2017_images.zip -DestinationPath C:\rams\data\coco

# YOLO-format labels (from Ultralytics — free, no registration)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip" `
    -OutFile C:\rams\data\coco2017labels.zip
Expand-Archive C:\rams\data\coco2017labels.zip -DestinationPath C:\rams\data\coco
```

Expected result:
```
C:\rams\data\coco\
  images\val2017\    ← 5000 × .jpg
  labels\val2017\    ← 5000 × .txt  (YOLO format)
```

---

## Phase 3 — Experiments 1–7 (simulation mode, fast)

These experiments don't require datasets. Run simulation first to validate the
full pipeline (takes ~5–10 min), then optionally re-run with real inference.

```powershell
cd C:\rams\rams_full

# ── Simulation run (no models needed, validate pipeline) ──────────────────
python experiments\run_all.py --simulate --n 60 --skip 9

# ── Real inference run (uses ONNX models downloaded in Phase 1) ───────────
python experiments\run_all.py --no-simulate --n 60 --skip 8 9 10 `
    --frames C:\rams\data\kitti\images\val
```

> **Expected time (real inference, n=60)**:
> Each of exps 1–7 runs 3 policies × 5 load profiles × 60 inferences.
> On the i7-1165G7 with ONNX, expect ~25–35 min total for exps 1–7.

Results are written to `results\` as CSV and JSON per experiment, plus PNG plots.

---

## Phase 4 — Benchmark run (required input for Experiment 10)

Experiment 10 needs a benchmark summary JSON. Run the benchmark harness
across all policies and all load profiles:

```powershell
# Simulation (fast, produces the JSON structure exp10 needs)
python -m benchmark.run --n 100 --policy all --simulate

# Real inference (recommended for the paper — takes ~20 min)
PS C:\rams\rams_full> "light","moderate","heavy","burst" | ForEach-Object {
    python -m benchmark.run --n 100 --policy all --no-simulate --profile $_
}

```

This writes:
```
results\rams_<timestamp>_<hostname>.csv
results\rams_<timestamp>_<hostname>.json   ← this is --benchmark-json for exp10
```

Note the exact `.json` filename — you'll need it in Phase 7.

---

## Phase 5 — Experiment 8 (per-tier mAP + VRU recall, both datasets)

This is the ground-truth accuracy measurement. Exp 8 runs all three YOLO tiers
against both KITTI and COCO and reports mAP50, mAP50-95, and VRU recall/FN
rates. It requires real models (no simulation mode).

```powershell
# ── KITTI ─────────────────────────────────────────────────────────────────
python experiments\exp8_accuracy_per_tier.py `
    --dataset kitti `
    --images  C:\rams\data\kitti\images\val `
    --labels  C:\rams\data\kitti\labels\val

# ── COCO val2017 ───────────────────────────────────────────────────────────
python experiments\exp8_accuracy_per_tier.py `
    --dataset coco `
    --images  C:\rams\data\coco\images\val2017 `
    --labels  C:\rams\data\coco\labels\val2017

# ── Both in one call ───────────────────────────────────────────────────────
python experiments\exp8_accuracy_per_tier.py `
    --dataset kitti `
    --images  C:\rams\data\kitti\images\val `
    --labels  C:\rams\data\kitti\labels\val `
    --also-coco `
    --coco-images C:\rams\data\coco\images\val2017 `
    --coco-labels C:\rams\data\coco\labels\val2017
```

> **Tip — smoke test first** (fast, ~200 images each):
> ```powershell
> python experiments\exp8_accuracy_per_tier.py `
>     --dataset kitti `
>     --images C:\rams\data\kitti\images\val `
>     --labels C:\rams\data\kitti\labels\val `
>     --max-images 200
> ```

> **Expected time**: MEDIUM tier at 640px on a CPU is the bottleneck.
> 1500 KITTI images × 3 tiers ≈ 60–90 min. 5000 COCO images × 3 tiers
> ≈ 3–5 hours. Consider using `--max-images 500` for COCO if time is tight.

Outputs:
```
results\exp8_accuracy_kitti.json   ← needed for exp10
results\exp8_accuracy_coco.json
results\exp8_vru_recall_kitti.png
results\exp8_map_per_tier_kitti.png
```

---

## Phase 6 — Experiment 9 (this device as a node)

Exp 9 is designed to run independently on each device then be aggregated.
Register this laptop as the CPU node:

```powershell
python experiments\exp9_multidevice.py `
    --device   spectre-i7-1165g7 `
    --backend  onnx `
    --frames   C:\rams\data\kitti\images\val `
    --n        200
```

This writes results to `results\multidevice\` tagged with the device name.

Then aggregate (useful even for a single device — produces normalized tables):
```powershell
python experiments\exp9_aggregate.py --results-dir results\multidevice\
```

---

## Phase 7 — Experiment 10 (safety–latency Pareto, real VRU recall)

Exp 10 combines the benchmark latencies (Phase 4) with the ground-truth VRU
recall from exp 8 (Phase 5) to produce the key paper figure.

```powershell
# ── KITTI Pareto ───────────────────────────────────────────────────────────
python experiments\exp10_safety_pareto.py `
    --exp8-json      results\exp8_accuracy_kitti.json `
    --benchmark-json results\rams_<TIMESTAMP>_<HOSTNAME>.json `
    --dataset        kitti

# ── With proximity window sweep (produces the Pareto arc) ─────────────────
python experiments\exp10_safety_pareto.py `
    --exp8-json       results\exp8_accuracy_kitti.json `
    --benchmark-json  results\rams_<TIMESTAMP>_<HOSTNAME>.json `
    --dataset         kitti `
    --sweep-proximity `
    --frames          C:\rams\data\kitti\images\val `
    --n               200

# ── COCO variant ───────────────────────────────────────────────────────────
python experiments\exp10_safety_pareto.py `
    --exp8-json      results\exp8_accuracy_coco.json `
    --benchmark-json results\rams_<TIMESTAMP>_<HOSTNAME>.json `
    --dataset        coco
```

Replace `<TIMESTAMP>` and `<HOSTNAME>` with the actual filename from Phase 4.

---

## Complete run order (summary)

```
Phase 1  →  Setup + ONNX export
Phase 2  →  Download KITTI + COCO
Phase 3  →  Exps 1–7  (simulate first, then real)
Phase 4  →  Benchmark run  (produces JSON for exp 10)
Phase 5  →  Exp 8  (KITTI + COCO, ~2–6 hours)
Phase 6  →  Exp 9  (this device, ~30 min)
Phase 7  →  Exp 10 (KITTI + COCO Pareto)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
```powershell
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'matplotlib'`
```powershell
pip install matplotlib
```

### PowerShell execution policy blocks `.ps1`
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Ultralytics model download times out
The first `--no-simulate` run downloads `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`
from GitHub. If behind a proxy or firewall, download them manually:
```
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
```
Place the three `.pt` files in `C:\rams\rams_full\` (same directory as `setup.py`).

### Exp 8 hangs or OOM with MEDIUM model
The MEDIUM tier loads a 640-px model which uses ~1.5 GB RAM for inference.
If your machine has 8 GB RAM and other processes are running, you may hit limits.
Reduce the batch size by using `--max-images 300` and close other applications.

### Intel Iris Xe acceleration (optional)
To use the OpenVINO execution provider for faster CPU inference on Tiger Lake:
```powershell
pip install onnxruntime-openvino
```
Then set the environment variable before running:
```powershell
$env:ORT_EXECUTION_PROVIDER = "OpenVINOExecutionProvider"
python experiments\run_all.py ...
```
Note: RAMS defaults to `CPUExecutionProvider` — the OpenVINO EP is optional
but can give 30–50% speedup on Intel integrated graphics.

### KITTI labels parse errors
If exp 8 prints warnings about malformed KITTI labels, some frames in the
training split have `DontCare` objects with zero-area bboxes. These are
filtered automatically by `parse_kitti_label` (it checks `len(parts) >= 15`).
This is expected and not an error.

---

## Expected output files

After all phases complete, `results\` will contain:

```
results\
├── exp1_policy_comparison.csv / .json / .png (×4) / .tex
├── exp2_load_sweep.csv / .json / .png
├── exp3_hysteresis.csv / .json / .png
├── exp4_safety_override.csv / .json / .png
├── exp5_pareto_moderate.png
├── exp5_pareto_heavy.png
├── exp5_pareto.csv / .json / .tex
├── exp6_transient.csv / .json / .png
├── exp7_swas.csv / .json / .png / .tex
├── exp8_accuracy_kitti.csv / .json
├── exp8_accuracy_coco.csv / .json
├── exp8_vru_recall_kitti.png
├── exp8_vru_recall_coco.png
├── exp8_map_per_tier_kitti.png
├── exp8_map_per_tier_coco.png
├── exp8_latex.tex
├── exp10_pareto_kitti.png
├── exp10_pareto_coco.png
├── exp10_proximity_sweep_kitti.png   (if --sweep-proximity)
├── exp10_latex.tex
├── rams_<timestamp>_<hostname>.csv   (benchmark)
├── rams_<timestamp>_<hostname>.json  (benchmark summary)
└── multidevice\
    └── spectre-i7-1165g7_*.csv / .json
```
