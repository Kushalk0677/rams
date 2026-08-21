# RAMS Jetson validation package

This package runs RAMS on an NVIDIA Jetson with device-built TensorRT engines.
It is for runtime and resource-telemetry evaluation. TensorRT engines are tied
to the exact Jetson model, JetPack release, TensorRT version, and engine build
settings. Do not copy an engine from another device.

Read `AI_OPERATOR_INSTRUCTIONS.md` before running commands. This package must
run on the physical target Jetson, not on WSL2, a desktop Linux machine, or a
remote emulator. The TensorRT engines are part of the evidence: build them on
the exact Jetson that will execute the phases.

The GitHub Linux compatibility workflow validates this package's source layout
and simulation CLI path. It does not validate JetPack, CUDA, TensorRT,
`tegrastats`, clocks, thermals, or real Jetson inference. Those checks must be
performed on the target Jetson.

## Before you begin

You need an NVIDIA Jetson with a supported JetPack installation, internet
access, a fan or otherwise stable cooling, adequate free storage for KITTI,
COCO, models, and results, and the ability to enter the Jetson administrator
password for `sudo` commands. Keep the selected power mode, clocks, cooling,
and power supply unchanged from calibration through the final run.

Obtain this `RAMS_Jetson_validation.zip` archive, KITTI training images and
labels, and the COCO `val2017` images and labels. KITTI requires registration.
The archive does not include datasets, model weights, TensorRT engines, or a
virtual environment.

## 1. Extract the archive and prepare Jetson

Install a supported JetPack release, then extract the archive on the Jetson.
Use JetPack's Python, CUDA, TensorRT, and PyTorch packages. Do not replace the
JetPack PyTorch build with a generic PyPI CUDA wheel.

Open a Terminal on the Jetson or connect using SSH. Save the archive under
`~/Downloads` before starting. Run each command in order:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv
unzip RAMS_Jetson_validation.zip -d ~/rams
cd ~/rams/rams_validation
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics
pip install -e .
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Both checks must succeed before collecting evidence. If `import tensorrt` or
`torch.cuda.is_available()` fails, stop. Resolve the JetPack installation
before continuing. Do not use a CPU or ONNX fallback while calling the result a
Jetson TensorRT run.

## 2. Build device-specific models and engines

Choose and record the Jetson power mode and cooling condition. The selected
mode and clocks must remain unchanged during a phase sequence.

The `nvpmodel` value below is an example for boards where mode `0` is the
highest available mode. Confirm the available modes on the target Jetson using
`sudo nvpmodel -q --verbose`; if mode `0` is not valid, select the documented
high-performance mode and record the exact output with the results.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
python3 -c "from ultralytics import YOLO; [YOLO(x) for x in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
python3 - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='engine', imgsz=320, half=True, device=0, workspace=4)
YOLO('yolov8s.pt').export(format='engine', imgsz=416, half=True, device=0, workspace=4)
YOLO('yolov8m.pt').export(format='engine', imgsz=640, half=True, device=0, workspace=4)
PY
ls -lh yolov8n.engine yolov8s.engine yolov8m.engine
```

If an engine cannot be built, stop and resolve that Jetson-specific issue. Do
not silently fall back to another backend for a TensorRT run.

## 3. Required TensorRT preflight

Before sending the package or starting any paper phase, execute one real
TensorRT inference through every tier on this Jetson. This fails if TensorRT,
`tegrastats`, an engine, an image, or an individual tier is unavailable. It
writes a retained JSON report under `results/`.

```bash
FRAME=$(find ~/rams/data/kitti/images/val -maxdepth 1 -name '*.png' | sort | head -n 1)
test -n "$FRAME"
python3 scripts/verify_jetson_tensorrt.py --frame "$FRAME"
```

Continue only if the report says `"status": "passed"` and every tier lists
`"backend": "tensorrt"`. This is an on-device gate. WSL2 and GitHub-hosted
Linux cannot replace it because they do not run the Jetson's TensorRT engines.

## 4. Prepare datasets

Download KITTI 2D Object Detection images and labels after registering at
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d>.
Create the fixed 1,500-frame replay from sorted indices 5981 through 7480.

```bash
python3 scripts/prepare_kitti_validation.py --data-root ~/rams/data
find ~/rams/data/kitti/images/val -name '*.png' | wc -l
find ~/rams/data/kitti/labels/val -name '*.txt' | wc -l
```

Both counts must be 1,500. If a complete replay already exists, use
`python3 scripts/prepare_kitti_validation.py --data-root ~/rams/data --reuse-existing`.
The helper refuses to overwrite a validation split and does not touch any
separate raw KITTI directory.

Download COCO `val2017.zip` and Ultralytics `coco2017labels.zip`, then place
them at `~/rams/data/coco/images/val2017/` and
`~/rams/data/coco/labels/val2017/`.

The following downloads the public COCO files and checks their expected
directories. KITTI must still be downloaded manually after registration.

```bash
mkdir -p ~/rams/data/downloads ~/rams/data/coco
curl -L --fail --output ~/rams/data/downloads/val2017.zip https://images.cocodataset.org/zips/val2017.zip
curl -L --fail --output ~/rams/data/downloads/coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
unzip ~/rams/data/downloads/val2017.zip -d ~/rams/data/coco
unzip ~/rams/data/downloads/coco2017labels.zip -d ~/rams/data/coco
test -d ~/rams/data/coco/images/val2017
test -d ~/rams/data/coco/labels/val2017
```

COCO has 5,000 images. Fewer label files is normal because some images contain
no annotated objects. If either final check fails, inspect the directories made
by the archive and move only the `images` or `labels` directory to the required
layout. Do not rename individual files.

## 5. Calibrate and run

Create and document a device-specific TDP profile. It produces an estimate,
not a physical energy measurement.

```bash
cp configs/energy_profile.example.json configs/energy_profile_jetson.json
nano configs/energy_profile_jetson.json
export RAMS_DATA_ROOT="$HOME/rams/data"
DEVICE="$(hostname)"
```

Smoke test the actual TensorRT path before a full run.

```bash
python3 scripts/run_paper_suite.py --platform jetson --backend tensorrt --smoke --device "$DEVICE"
```

After smoke completes, inspect its newest `results/paper_*.json` file. It must
report `"smoke": true`, `"simulated": false`, and no failed command. The
TensorRT preflight and smoke are installation gates, not paper evidence.

Run the phases in order. Preserve the calibration, engine files, power mode,
clock state, and cooling condition from calibration through retention.

```bash
python3 scripts/run_paper_suite.py --phase calibration --platform jetson --device "$DEVICE"
for phase in runtime1 runtime2 runtime3 runtime4; do
  python3 scripts/run_paper_suite.py --phase "$phase" --platform jetson --backend tensorrt --device "$DEVICE" --energy-profile configs/energy_profile_jetson.json || exit 1
done
python3 scripts/run_paper_suite.py --phase accuracy --platform jetson --backend tensorrt --device "$DEVICE"
python3 scripts/run_paper_suite.py --phase retention --platform jetson --backend tensorrt --device "$DEVICE"
```

Retain the full `results/` directory, including raw records, manifests,
calibration snapshots, device state, and the energy-profile input.

After each phase, inspect its newest `results/paper_<phase>_*.json` before
starting the next. Stop if any command records `"ok": false`. A one-command
full run, used only after passing all earlier setup gates, is:

```bash
python3 scripts/run_paper_suite.py --phase all --platform jetson --backend tensorrt --device "$DEVICE" --energy-profile configs/energy_profile_jetson.json
```

When handing results back, retain the complete `results/` directory and energy
profile. Include the Jetson model, JetPack version, TensorRT version, selected
power mode, clock state, cooling condition, and the successful TensorRT
preflight report. A result is complete only if the `paper_all` manifest has no
failed stage.
