# RAMS — Resource-Adaptive Model Switching for Edge AI

**Kushal Khemani · Ajinkyaa Lokhande**

RAMS is a runtime framework that continuously monitors system resource pressure
and dynamically selects among a tiered set of YOLOv8 perception models to
maximise safety-weighted accuracy within hard latency bounds.

---

## How It Works

```
ResourceMonitor ──► R(t) ∈ [0,1] ──► SwitchingPolicy ──► ModelLibrary
     10 Hz             pressure          (threshold /          (nano /
  (CPU, mem,                             predictive /          small /
   temp, bat)                             safety)              medium)
                                               │
                                         infer(frame)
```

Five policies are included:

| Policy | Description |
|---|---|
| `threshold` | Fixed R(t) thresholds with hysteresis to prevent flapping |
| `predictive` | EWMA forecast of pressure; anticipates load spikes |
| `safety` | Threshold + class-conditional override: locks to SMALL+ when a VRU is detected within 0.5 s |
| `adaptive` | Self-tuning EWMA alpha based on rolling pressure variance |
| `safety2` | Two-level safety override: near VRU → MEDIUM, distant VRU → SMALL |

All three model tiers are kept **warm in memory** — no model-loading latency
during switching events.

---

## Inference Backends

RAMS supports three inference backends, selected automatically at load time:

| Priority | Backend | When used |
|---|---|---|
| 1 | **ONNX Runtime** | `.onnx` file found alongside `.pt` — fastest on CPU |
| 2 | **Ultralytics** | `.onnx` not found, `.pt` available |
| 3 | **Simulation** | No model files found — calibrated Gaussian latency |

### Per-tier resolution (mixed-resolution strategy)

| Tier | Model | Resolution | Purpose |
|---|---|---|---|
| NANO | `yolov8n.pt` | 320 × 320 | Speed — high pressure fallback |
| SMALL | `yolov8s.pt` | 416 × 416 | Balance — default safe tier |
| MEDIUM | `yolov8m.pt` | 640 × 640 | Accuracy — VRU safety override |

Running each tier at its own resolution gives ~2–4× latency reduction for
NANO and SMALL versus a flat 640px baseline, while preserving full-resolution
accuracy for the safety-critical MEDIUM tier.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Kushalk0677/rams
cd rams
pip install -e .

# For real YOLOv8 inference:
pip install -e ".[inference]"

# For ONNX Runtime (recommended for CPU deployment):
pip install onnxruntime
```

### 2. Download YOLOv8 weights

```bash
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
YOLO('yolov8s.pt')
YOLO('yolov8m.pt')
"
```

Weights are downloaded automatically from Ultralytics on first use and cached
in your system's Ultralytics assets directory.

### 3. Export ONNX models (recommended)

Export each tier at its designated resolution. The `.onnx` files must be placed
in the same directory as the corresponding `.pt` files (or your working
directory) so RAMS can find them automatically.

```bash
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)
YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12)
YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)
"
```

This produces `yolov8n.onnx`, `yolov8s.onnx`, and `yolov8m.onnx`. On a
mid-range laptop CPU (e.g. i7-1165G7), ONNX Runtime reduces inference latency
by approximately 2–3× compared to the PyTorch backend.

> **Note for Windows users:** if the export command opens a browser window or
> hangs, run it from a plain `cmd` terminal rather than PowerShell.

### 4. Verify ONNX is being used

```bash
python -c "
from rams import RAMSController
with RAMSController(simulate=False, policy='safety') as ctrl:
    result = ctrl.infer()
    print('Backend:', result['backend'])   # should print 'onnx'
    print('Tier:',    result['tier'])
    print('Latency:', result['latency_ms'], 'ms')
"
```

### 5. Live demo

```bash
# Simulation mode — no GPU or weights needed
python scripts/live_demo.py --simulate --policy safety

# Real inference (PyTorch backend)
python scripts/live_demo.py --no-simulate --policy safety

# Real inference with ONNX (fastest — export ONNX files first)
python scripts/live_demo.py --no-simulate --policy safety
```

### 6. Run benchmark

```bash
# Simulation — fast smoke test
python -m benchmark.run --n 100 --policy all --simulate

# Real inference with real frames (recommended — enables VRU detection)
python -m benchmark.run --n 100 --policy all --no-simulate --frames /path/to/frames

# Heavy load only
python -m benchmark.run --n 50 --profile heavy --no-simulate
```

The `--frames` argument accepts a directory of `.jpg` or `.png` images.
RAMS cycles through them during the benchmark so that the VRU detection
and safety override fire on realistic scene content. Without `--frames`,
the benchmark uses blank dummy frames and VRU detection will not trigger.

Results are written to `results/rams_<timestamp>_<hostname>.csv` and a
JSON summary including per-cell SWAS scores.

### 7. Aggregate results from multiple machines

Copy all `results/*.csv` files from every machine into one directory, then:

```bash
python scripts/aggregate.py --dir results/
```

Each CSV is tagged with hostname and timestamp — no filename collisions.

---

## Project Structure

```
rams/
├── rams/
│   ├── __init__.py
│   ├── monitor.py        # ResourceMonitor — R(t) computation
│   ├── models.py         # ModelLibrary — ONNX/Ultralytics/sim tier wrappers
│   ├── policy.py         # ThresholdPolicy, PredictivePolicy, SafetyPolicy,
│   │                     # AdaptivePredictivePolicy, SafetyTwoLevelPolicy
│   └── controller.py    # RAMSController — top-level API
├── benchmark/
│   └── run.py            # Benchmark harness with load + frame injection
├── experiments/
│   ├── exp1_policy_comparison.py
│   ├── exp2_load_sweep.py
│   ├── exp3_hysteresis.py
│   ├── exp4_safety_override.py
│   ├── exp5_pareto.py
│   ├── exp6_transient.py
│   ├── exp7_swas.py      # Safety-Weighted Accuracy Score (SWAS)
│   └── run_all.py
├── scripts/
│   ├── live_demo.py      # Live terminal demo
│   └── aggregate.py      # Multi-machine results aggregator
├── configs/
│   └── default.yaml      # All tunable parameters
├── results/              # Output CSVs, JSONs, plots (gitignored)
├── requirements.txt
└── setup.py
```

---

## Using RAMS in Your Own Code

```python
from rams import RAMSController

with RAMSController(simulate=False, policy="safety") as ctrl:
    while capturing:
        frame = camera.read()
        result = ctrl.infer(frame)

        print(result["tier"])         # "NANO" | "SMALL" | "MEDIUM"
        print(result["latency_ms"])   # end-to-end inference time (ms)
        print(result["pressure"])     # R(t) at inference time
        print(result["detections"])   # list of {class, conf, xyxy}
        print(result["backend"])      # "onnx" | "ultralytics" | "simulation"
```

---

## Configuration

All parameters are in `configs/default.yaml`. Key knobs:

| Parameter | Default | Effect |
|---|---|---|
| `policy.default` | `safety` | Which policy to use |
| `policy.threshold.lo_thresh` | `0.45` | R(t) below → MEDIUM |
| `policy.threshold.hi_thresh` | `0.72` | R(t) above → NANO |
| `policy.threshold.hysteresis_window` | `3` | Samples before tier commits |
| `policy.safety.proximity_window_s` | `0.5` | VRU lock window (seconds) |
| `policy.safety.min_conf` | `0.25` | Min detection confidence for VRU trigger |
| `monitor.hz` | `10.0` | Resource sampling rate |
| `models.simulate` | `false` | Skip real model loading |

---

## ONNX Troubleshooting

**RAMS is using Ultralytics instead of ONNX:**
Check that the `.onnx` files are present in your working directory or the
same folder as the `.pt` files. RAMS looks for `yolov8n.onnx`, `yolov8s.onnx`,
and `yolov8m.onnx` by name.

**Export fails with shape mismatch:**
Make sure you export each model at the correct resolution for its tier —
320 for NANO, 416 for SMALL, 640 for MEDIUM. Mixing resolutions will cause
incorrect detections.

**`onnxruntime` not found:**
```bash
pip install onnxruntime        # CPU
pip install onnxruntime-gpu    # GPU (if available)
```

**Slow inference despite ONNX:**
Set `intra_op_num_threads` to match your physical core count. RAMS defaults
to 4 threads — on machines with fewer cores (e.g. Raspberry Pi), lower this
to 2 for better throughput.

---

## Running Across Multiple Machines

1. Clone and install on each machine.
2. Export ONNX models on each machine (resolutions are machine-independent).
3. Run the benchmark:
   ```bash
   python -m benchmark.run --n 100 --policy all --no-simulate --frames /path/to/frames
   ```
4. Copy all `results/*.csv` to one machine.
5. Aggregate:
   ```bash
   python scripts/aggregate.py --dir results/
   ```

---

## Companion Work

RAMS is the model-selection layer. For the complementary task-scheduling
layer (which task to run, in what order), see:

> **PAES: Priority-Aware Adaptive Scheduling for Multi-Model Edge AI Systems**
> Khemani, Maniar & Lokhande — IEEE Embedded Systems Letters
> https://github.com/Kushalk0677/Priority-Aware-Adaptive-Scheduling-for-Multi-Model-Edge-AI-Systems

Together, PAES handles *scheduling order* and RAMS handles *which model tier
to run* — forming a two-layer adaptive edge AI stack.

---

## Citation

```bibtex
@misc{khemani2025rams,
  title  = {Resource-Adaptive Model Switching for Safety-Aware Perception in Autonomous Edge Systems},
  author = {Khemani, Kushal and Lokhande, Ajinkyaa},
  year   = {2025},
}
```
