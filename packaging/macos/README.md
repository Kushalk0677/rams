# RAMS macOS Apple Silicon validation package

This package runs the current RAMS workflow on Apple Silicon macOS with the
CPU ONNX Runtime provider. It collects runtime, telemetry, and detection-policy
evidence. It does not claim GPU, Neural Engine, MPS, vehicle-control, safety,
or physical energy measurement results.

Read `AI_OPERATOR_INSTRUCTIONS.md` before running commands. This guide assumes
a MacBook Air or MacBook Pro with Apple Silicon. The required inference route
is CPU ONNX Runtime. Do not replace it with a GPU, Neural Engine, MPS, or Core
ML claim unless a separately documented protocol authorizes that route.

## Before you begin

You need macOS on Apple Silicon, internet access, an administrator password for
the developer tools, about 25 GB free storage for the package, models, and
datasets, and mains power for a laptop. Close other sustained workloads and
keep power mode, cooling, and the power connection unchanged from calibration
through the final phase.

Obtain this `RAMS_macOS_validation.zip` archive, KITTI 2D Object Detection
training images and labels, COCO `val2017` images, and Ultralytics YOLO-format
COCO labels. KITTI requires free registration at the linked site below.

## 1. Extract the archive and install Python

Open **Terminal** from Applications, Utilities. Run the following one command
at a time. If `brew` is not found, first install Homebrew from
[brew.sh](https://brew.sh/) and reopen Terminal. Use Python 3.12 and work from
the extracted package directory.

```bash
xcode-select --install
brew install python@3.12
unzip RAMS_macOS_validation.zip -d ~/rams
cd ~/rams/rams_validation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-inference.txt
pip install -e .
python -c "import onnxruntime as ort; print(ort.get_available_providers()); assert 'CPUExecutionProvider' in ort.get_available_providers()"
python -c "from rams import RAMSController; print('RAMS import OK')"
```

If `xcode-select --install` opens a dialog, complete that installation before
continuing. If the final ONNX command does not list `CPUExecutionProvider`,
stop and resolve the environment error before downloading the datasets.

## 2. Download models

Download and export the three required YOLOv8 tiers in the package directory.

```bash
python -c "from ultralytics import YOLO; [YOLO(x) for x in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12); YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12); YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)"
ls -lh yolov8n.onnx yolov8s.onnx yolov8m.onnx
```

## 3. Download and prepare the datasets

Download KITTI 2D Object Detection training images and labels after registering
at <https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d>.
Extract them under `~/rams/data/kitti/images/training/image_2` and
`~/rams/data/kitti/labels/training/label_2`, then create the fixed validation
replay used across devices.

```bash
python scripts/prepare_kitti_validation.py --data-root ~/rams/data
find ~/rams/data/kitti/images/val -name '*.png' | wc -l
find ~/rams/data/kitti/labels/val -name '*.txt' | wc -l
```

If the replay already exists and both counts are 1,500, use
`python scripts/prepare_kitti_validation.py --data-root ~/rams/data --reuse-existing`.
The helper never overwrites an existing split.

Both counts must be 1,500. Download COCO `val2017.zip` from
<https://cocodataset.org/#download>, and the Ultralytics `coco2017labels.zip`
release. Extract them to:

```text
~/rams/data/coco/images/val2017/
~/rams/data/coco/labels/val2017/
```

COCO has 5,000 images; fewer label files are normal for images with no
annotations.

To download the two public COCO archives in Terminal, use:

```bash
mkdir -p ~/rams/data/downloads ~/rams/data/coco
curl -L --fail --output ~/rams/data/downloads/val2017.zip https://images.cocodataset.org/zips/val2017.zip
curl -L --fail --output ~/rams/data/downloads/coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
unzip ~/rams/data/downloads/val2017.zip -d ~/rams/data/coco
unzip ~/rams/data/downloads/coco2017labels.zip -d ~/rams/data/coco
test -d ~/rams/data/coco/images/val2017
test -d ~/rams/data/coco/labels/val2017
```

If either final check fails, inspect the directories created by the archive and
move only the `images` or `labels` directory to the layout shown above. Do not
rename individual files. Keep any separate raw KITTI dataset directory intact.

## 4. Calibrate and run

Create the TDP-profile estimate input. It is not a power-meter measurement.

```bash
cp configs/energy_profile.example.json configs/energy_profile_macos.json
nano configs/energy_profile_macos.json
export RAMS_DATA_ROOT="$HOME/rams/data"
DEVICE="$(scutil --get ComputerName)"
```

Run a smoke check first. It is a setup check only.

```bash
python scripts/run_paper_suite.py --platform macos --backend onnx --smoke --device "$DEVICE"
```

After smoke completes, inspect the newest `results/paper_*.json`. It must say
`"smoke": true`, `"simulated": false`, and contain no failed command. Do not
use this output in paper tables.

Then run the phases in order. Keep the resulting calibration and model files
unchanged for all later phases.

```bash
python scripts/run_paper_suite.py --phase calibration --platform macos --device "$DEVICE"
for phase in runtime1 runtime2 runtime3 runtime4; do
  python scripts/run_paper_suite.py --phase "$phase" --platform macos --backend onnx --device "$DEVICE" --energy-profile configs/energy_profile_macos.json || exit 1
done
python scripts/run_paper_suite.py --phase accuracy --platform macos --backend onnx --device "$DEVICE"
python scripts/run_paper_suite.py --phase retention --platform macos --backend onnx --device "$DEVICE"
```

Keep the complete `results/` directory. The required records include raw CSV
files, JSON summaries, replay manifests, calibration snapshots, and the energy
profile. COCO mAP is reportable only when records say
`map_source: ultralytics_val`.

After each phase, inspect its newest `results/paper_<phase>_*.json` before
starting the next one. Stop if a command has `"ok": false`. For a one-command
full run after the smoke check, use:

```bash
python scripts/run_paper_suite.py --phase all --platform macos --backend onnx --device "$DEVICE" --energy-profile configs/energy_profile_macos.json
```

When handing off the work, retain the complete `results/` directory and the
energy-profile JSON. State the Mac model, macOS version, device label, and
whether the full `paper_all` manifest completed without failures.
