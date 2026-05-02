# RAMS â€” Jetson Orin Runbook (Calibrated Â· TensorRT)

**Device**: NVIDIA Jetson Orin  
**Backend**: TensorRT (`.engine` files)  
**Goal**: Calibrated full experiment run, experiments 1 through 10  
**Dataset**: KITTI val (1500 frames)

> This runbook produces **calibrated numbers only** â€” the uncalibrated
> ablation is already covered by the Spectre results.

---

## Pre-flight checklist

| Requirement | Notes |
|---|---|
| JetPack 5.x or 6.x | Provides CUDA, cuDNN, TensorRT â€” do not install these separately |
| Python 3.8â€“3.11 | Comes with JetPack; avoid 3.12+ on Orin until wheels stabilise |
| ~20 GB free on `/home` | KITTI ~12 GB, TRT engines ~500 MB, results ~1 GB |
| Jetson on mains power | `sudo nvpmodel -m 0 && sudo jetson_clocks` â€” locks to max TDP before any run |
| Swap enabled | 8 GB swap recommended; MEDIUM engine build needs headroom |

**Lock to max performance before every session:**
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**Add swap if not already present:**
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Phase 1 â€” Environment setup

JetPack ships PyTorch for Jetson â€” **do not** `pip install torch` from PyPI;
it will install a CPU-only wheel that ignores the GPU.

```bash
# Clone / copy the repo
cd ~
unzip rams_with_experiments.zip
cd rams_full

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Core deps
pip install psutil pyyaml numpy matplotlib opencv-python

# Ultralytics (pulls in no heavy torch dependency â€” uses the system one)
pip install ultralytics

# TensorRT Python bindings (already on system via JetPack, just expose them)
pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-tensorrt 2>/dev/null || true
# If the above fails, the bindings are already accessible system-wide;
# the venv just needs to see them:
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

# Install the rams package
pip install -e .
```

> **Verify GPU is visible:**
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```
> Should print `True  Orin` (or similar). If False, PyTorch is the wrong build.

---

## Phase 2 â€” Export TensorRT engines (do once)

TensorRT engines are device-specific â€” engines built on the Jetson will not
run on the Spectre, and vice versa. Build them here before any experiment.

```bash
cd ~/rams_full

python - <<'EOF'
from ultralytics import YOLO

# fp16 gives the best latency/accuracy trade-off on Orin
YOLO('yolov8n.pt').export(
    format='engine', imgsz=320, half=True, device=0,
    workspace=4   # GB; reduce to 2 if you hit OOM during build
)
YOLO('yolov8s.pt').export(
    format='engine', imgsz=416, half=True, device=0, workspace=4
)
YOLO('yolov8m.pt').export(
    format='engine', imgsz=640, half=True, device=0, workspace=4
)
EOF
```

This produces `yolov8n.engine`, `yolov8s.engine`, `yolov8m.engine` in the
working directory. Each build takes 3â€“8 minutes; the MEDIUM build is longest.

> **If a build OOMs**, drop `workspace` to 2 and retry. If it still fails,
> build one at a time with other processes killed.

RAMS auto-detects `.engine` files and uses the TensorRT execution path â€” no
code changes needed.

---

## Phase 3 â€” Dataset

Copy the KITTI val split from the Spectre (or re-download):

```bash
mkdir -p ~/rams/data/kitti/images/val
mkdir -p ~/rams/data/kitti/labels/val

# Option A: scp from the Spectre (fastest)
scp -r <spectre-ip>:C:/rams/data/kitti/images/val/* ~/rams/data/kitti/images/val/
scp -r <spectre-ip>:C:/rams/data/kitti/labels/val/* ~/rams/data/kitti/labels/val/

# Option B: re-download and split (same 5981â€“7480 frames as the Spectre)
# Download from https://www.cvlibs.net/datasets/kitti/eval_object.php
# then:
ls ~/kitti/data_object_image_2/training/image_2/*.png | sort | \
    tail -n +5982 | head -1500 | \
    xargs -I{} cp {} ~/rams/data/kitti/images/val/
ls ~/kitti/data_object_label_2/training/label_2/*.txt | sort | \
    tail -n +5982 | head -1500 | \
    xargs -I{} cp {} ~/rams/data/kitti/labels/val/
```

Confirm counts:
```bash
ls ~/rams/data/kitti/images/val | wc -l   # expect 1500
ls ~/rams/data/kitti/labels/val | wc -l   # expect 1500
```

---

## Phase 4 â€” Calibration

Calibration measures the Jetson's idle resource baseline and writes
device-specific thresholds. The Jetson idles hotter than a laptop, so this
should be done before the full experiment suite.

```bash
cd ~/rams_full
source .venv/bin/activate

python scripts/calibrate.py --seconds 30 --apply
```

This samples R(t) for about 30 seconds at idle and writes:

```text
results/calibration_<timestamp>.json
```

Check the output: `lo_thresh` and `hi_thresh` should look reasonable
(typically lo around 0.30-0.45, hi around 0.55-0.70 on Orin depending on
ambient temperature and clock state). If the thresholds look extreme
(lo > 0.60), ensure `sudo jetson_clocks` was run first and the board has had
2-3 minutes to reach steady-state temperature before calibrating.

---

## Phase 5 â€” One-command full run (recommended)

After Phase 1-4 are complete, this runs the full suite:

```bash
cd ~/rams_full
source .venv/bin/activate

python experiments/complete_runall_jetson.py \
    --frames       ~/rams/data/kitti/images/val \
    --kitti-images ~/rams/data/kitti/images/val \
    --kitti-labels ~/rams/data/kitti/labels/val \
    --n            200 \
    --backend      tensorrt \
    --device       jetson-orin \
    --require-full-kitti \
    --skip-calibration
```

What it does, in order:

1. Checks KITTI paths and TensorRT engine availability.
2. Exports TensorRT engines if they are missing.
3. Runs experiments 1, 2, 3, 4, 5, 6, and 7.
4. Runs the benchmark harness needed by exp10.
5. Runs exp8 on KITTI.
6. Runs exp9 for `jetson-orin/tensorrt` and aggregates multidevice results.
7. Runs exp10 from the new exp8 and benchmark outputs.

The manifest is written to:

```text
results/jetson_complete_runall_manifest.json
```

For a quick smoke test, reduce counts:

```bash
python experiments/complete_runall_jetson.py \
    --n 5 \
    --transient-total-n 40 \
    --max-images 50
```

If you want the full runner to calibrate for you instead, omit `--skip-calibration`.
If you already exported engines in the same boot session:

```bash
python experiments/complete_runall_jetson.py --skip-export --skip-calibration --n 200
```


## Phase 6 â€” Experiment 9 only (manual fallback)

```bash
cd ~/rams_full
source .venv/bin/activate

python experiments/exp9_multidevice.py \
    --device   jetson-orin \
    --backend  tensorrt \
    --frames   ~/rams/data/kitti/images/val \
    --n        200
```

This runs all 5 policies Ã— 5 load profiles Ã— 200 inferences with real KITTI
frames, using the TensorRT engines built in Phase 2.

Expected time: **45â€“75 min** (MEDIUM tier at 640px is the bottleneck even on
Orin; load injection adds overhead at burst).

Results are written to:
```
results/multidevice/exp9_jetson-orin_tensorrt_<timestamp>.csv
results/multidevice/exp9_jetson-orin_tensorrt_<timestamp>.json
```

---

## Phase 7 â€” Aggregate with Spectre results

Copy the Jetson's multidevice output to the Spectre (or vice versa) so both
devices' CSVs are in the same directory:

```bash
# On the Jetson â€” push to Spectre
scp results/multidevice/exp9_jetson-orin_* \
    <spectre-user>@<spectre-ip>:"C:/rams/rams_full/results/calibrated/multidevice/"
```

Then on the Spectre:
```powershell
cd C:\rams\rams_full
python experiments\exp9_aggregate.py --results-dir results\calibrated\multidevice\
```

This produces the cross-device comparison table and plots.

---

## Complete run order (summary)

```
Phase 1  â†’  Environment setup
Phase 2  â†’  TensorRT engine export  (~15â€“25 min total)
Phase 3  â†’  KITTI val dataset
Phase 4  â†’  Calibration
Phase 5  â†’  complete_runall_jetson.py (experiments 1â€“10)
Phase 6  â†’  Manual exp9 fallback
Phase 7  â†’  Aggregate on Spectre
```

---

## Troubleshooting

### `torch.cuda.is_available()` returns False
The venv picked up the wrong PyTorch. Exit the venv and check:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If this returns True but the venv's Python does not, the venv's `pip install`
pulled a CPU wheel. Fix:
```bash
pip uninstall torch torchvision -y
# Point pip at the JetPack-provided wheel directly
pip install /path/to/torch-*.whl   # find with: find /usr -name "torch*.whl" 2>/dev/null
```

### TensorRT engine build fails with `out of memory`
Reduce workspace and kill background processes:
```bash
sudo systemctl stop nvargus-daemon  # frees ~200 MB GPU memory
# Then retry export with workspace=2
```

### Engine build hangs indefinitely
TensorRT profiling on first build can take 10+ minutes for MEDIUM at 640px.
This is normal â€” do not kill it. If it hangs past 20 minutes, check GPU
utilisation with `tegrastats`; if GPU is 0% the process has silently crashed.

### `tegrastats` shows thermal throttling during exp 9
The experiment adds artificial CPU load. If the Jetson hits 85Â°C+ it will
throttle, which invalidates the latency measurements. Ensure adequate airflow,
re-run `sudo jetson_clocks` after any reboot, and monitor with:
```bash
watch -n1 tegrastats
```
If temperature exceeds 80Â°C during a run, abort, cool down, and re-run.

### Calibration produces lo_thresh > 0.65
The board is not at steady-state idle. Wait 3â€“5 minutes after locking clocks,
then re-run calibration. If it still reads high, the board may be warm from
previous runs â€” let it cool fully first.

### exp9_multidevice.py cannot find `.engine` files
RAMS looks for engine files in the working directory (same as `setup.py`).
Confirm:
```bash
ls ~/rams_full/*.engine   # should show all three
```
If missing, re-run Phase 2 from `~/rams_full/`.
