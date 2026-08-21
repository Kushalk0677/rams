# RAMS macOS Apple Silicon validation package

This package runs the current RAMS workflow on Apple Silicon macOS with the
CPU ONNX Runtime provider. It collects runtime, telemetry, and detection-policy
evidence. It does not claim GPU, Neural Engine, MPS, vehicle-control, safety,
or physical energy measurement results.

## 1. Prepare the machine

Use Python 3.12 and work from the extracted package directory.

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
mkdir -p ~/rams/data/kitti/images/val ~/rams/data/kitti/labels/val
mapfile -t images < <(find ~/rams/data/kitti/images/training/image_2 -maxdepth 1 -name '*.png' | sort)
mapfile -t labels < <(find ~/rams/data/kitti/labels/training/label_2 -maxdepth 1 -name '*.txt' | sort)
for i in $(seq 5981 7480); do cp "${images[$i]}" ~/rams/data/kitti/images/val/; cp "${labels[$i]}" ~/rams/data/kitti/labels/val/; done
find ~/rams/data/kitti/images/val -name '*.png' | wc -l
find ~/rams/data/kitti/labels/val -name '*.txt' | wc -l
```

Both counts must be 1,500. Download COCO `val2017.zip` from
<https://cocodataset.org/#download>, and the Ultralytics `coco2017labels.zip`
release. Extract them to:

```text
~/rams/data/coco/images/val2017/
~/rams/data/coco/labels/val2017/
```

COCO has 5,000 images; fewer label files are normal for images with no
annotations.

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
