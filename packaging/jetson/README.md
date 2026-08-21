# RAMS Jetson validation package

This package runs RAMS on an NVIDIA Jetson with device-built TensorRT engines.
It is for runtime and resource-telemetry evaluation. TensorRT engines are tied
to the exact Jetson model, JetPack release, TensorRT version, and engine build
settings. Do not copy an engine from another device.

The GitHub Linux compatibility workflow validates this package's source layout
and simulation CLI path. It does not validate JetPack, CUDA, TensorRT,
`tegrastats`, clocks, thermals, or real Jetson inference. Those checks must be
performed on the target Jetson.

## 1. Prepare Jetson

Install a supported JetPack release, then extract the archive on the Jetson.
Use JetPack's Python, CUDA, TensorRT, and PyTorch packages. Do not replace the
JetPack PyTorch build with a generic PyPI CUDA wheel.

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

Both checks must succeed before collecting evidence.

## 2. Build device-specific models and engines

Choose and record the Jetson power mode and cooling condition. The selected
mode and clocks must remain unchanged during a phase sequence.

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

## 3. Prepare datasets

Download KITTI 2D Object Detection images and labels after registering at
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d>.
Create the fixed 1,500-frame replay from sorted indices 5981 through 7480.

```bash
mkdir -p ~/rams/data/kitti/images/val ~/rams/data/kitti/labels/val
mapfile -t images < <(find ~/rams/data/kitti/images/training/image_2 -maxdepth 1 -name '*.png' | sort)
mapfile -t labels < <(find ~/rams/data/kitti/labels/training/label_2 -maxdepth 1 -name '*.txt' | sort)
for i in $(seq 5981 7480); do cp "${images[$i]}" ~/rams/data/kitti/images/val/; cp "${labels[$i]}" ~/rams/data/kitti/labels/val/; done
```

Download COCO `val2017.zip` and Ultralytics `coco2017labels.zip`, then place
them at `~/rams/data/coco/images/val2017/` and
`~/rams/data/coco/labels/val2017/`.

## 4. Calibrate and run

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
