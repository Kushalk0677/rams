# RAMS Windows CPU ONNX validation package

This package runs RAMS on Windows with the CPU ONNX Runtime provider. It collects runtime, resource-telemetry, and detection-policy evidence. It does not claim Windows GPU inference, vehicle control, safety guarantees, or physical energy measurement.

Read this guide before running commands. An autonomous operator must read `AI_OPERATOR_INSTRUCTIONS.md` first and then follow this guide in order.

## Before you begin

You need a 64-bit Windows 10 or 11 computer, an internet connection, at least 16 GB RAM, about 25 GB free disk space for the package, models, and datasets, and a mains-power connection for a laptop. Close games, browsers, and other long-running workloads before calibration and runtime phases. Keep the computer plugged in and leave its power plan unchanged for the complete run.

You must obtain three things before a complete paper run:

1. This `RAMS_Windows_validation.zip` archive.
2. KITTI 2D Object Detection training images and labels, which require free KITTI registration.
3. COCO `val2017` images and Ultralytics YOLO-format COCO labels.

The package can be smoke-tested only after the models and both datasets are available. Do not begin a full run with missing data.

## 1. Extract the archive and install Python

Use Python 3.12.x and work from the extracted package directory. The archive does not include datasets, model weights, ONNX models, an environment, or results. An NVIDIA GPU may provide telemetry but is not used for inference in this protocol.

If `python --version` does not show Python 3.12.x, download the 64-bit Python 3.12 installer from [python.org](https://www.python.org/downloads/), run it, select **Add Python to PATH**, then close and reopen PowerShell. To open PowerShell, press the Windows key, type `PowerShell`, and open it normally. Do not run it as Administrator.

```powershell
Expand-Archive -Path "$HOME\Downloads\RAMS_Windows_validation.zip" -DestinationPath "$HOME\rams"
cd "$HOME\rams\rams_validation"
python --version
# Must report Python 3.12.x.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-inference.txt
pip install pynvml
pip install -e .
python -c "import onnxruntime as ort; print(ort.get_available_providers()); assert 'CPUExecutionProvider' in ort.get_available_providers()"
python -c "from rams import RAMSController; print('RAMS import OK')"
```

Keep `PYTHONIOENCODING` set in the PowerShell session for calibration and all run commands. `pynvml` is optional. It exposes NVIDIA telemetry when NVML is available but does not change the CPU ONNX inference route.

If activation is blocked by PowerShell policy, run this once for the current user, close PowerShell, reopen it, return to the extracted directory, and activate the environment again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Download weights and export ONNX models

RAMS uses NANO at 320, SMALL at 416, and MEDIUM at 640. Download the official Ultralytics checkpoints into the package directory, then export ONNX models from the same checkpoints.

```powershell
python -c "from ultralytics import YOLO; [YOLO(name) for name in ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')]"
pip install onnx onnxslim
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12); YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12); YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)"
Get-ChildItem yolov8n.pt,yolov8s.pt,yolov8m.pt,yolov8n.onnx,yolov8s.onnx,yolov8m.onnx
```

Do not continue unless all six files exist. A Windows paper run fails rather than silently changing backend.

## 3. Download and prepare KITTI and COCO

Register at the [KITTI 2D Object Detection page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d), download left-color training images and training labels, and extract them. Keep the original extraction. The commands below create the fixed 1,500-frame replay used across devices, from sorted indices 5981 through 7480 inclusive.

```powershell
python scripts\prepare_kitti_validation.py --data-root D:\data
(Get-ChildItem -LiteralPath D:\data\kitti\images\val -Filter *.png).Count
(Get-ChildItem -LiteralPath D:\data\kitti\labels\val -Filter *.txt).Count
```

If the fixed replay was already created and both counts are 1,500, run the
helper with `--reuse-existing`. It never overwrites a validation split.

Both counts must be 1,500. Download `val2017.zip` from [COCO](https://cocodataset.org/#download) and `coco2017labels.zip` from the Ultralytics asset release. Extract them to:

```text
D:\data\coco\images\val2017\000000000139.jpg
D:\data\coco\labels\val2017\000000000139.txt
```

COCO has 5,000 images. Fewer label files are normal because some images have no annotations. Set the common data root:

If you want PowerShell to download the two public COCO files, use the following. KITTI remains a manual download because its site requires registration. Do not move or overwrite an unrelated `kitti_raw` directory.

```powershell
New-Item -ItemType Directory -Force -Path D:\data\downloads,D:\data\coco | Out-Null
Invoke-WebRequest -Uri 'https://images.cocodataset.org/zips/val2017.zip' -OutFile D:\data\downloads\val2017.zip
Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip' -OutFile D:\data\downloads\coco2017labels.zip
Expand-Archive -LiteralPath D:\data\downloads\val2017.zip -DestinationPath D:\data\coco
Expand-Archive -LiteralPath D:\data\downloads\coco2017labels.zip -DestinationPath D:\data\coco
Test-Path D:\data\coco\images\val2017
Test-Path D:\data\coco\labels\val2017
```

If the two final checks return `False`, inspect the folders created by the archive and move only the `images` and `labels` directories so that they match the layout above. Do not rename individual image or label files.

```powershell
$env:RAMS_DATA_ROOT = 'D:\data'
```

## 4. Create the energy profile

The workflow estimates energy from runtime telemetry and documented TDP assumptions. It does not use a power meter. Copy the template and replace its example values and source note with the actual machine and operating state.

```powershell
Copy-Item configs\energy_profile.example.json configs\energy_profile_windows.json
notepad configs\energy_profile_windows.json
```

Only claim estimated energy under the stated telemetry-conditioned power model. Never call these values measured energy.

For a first run, preserve the template structure and fill in the device name, processor, operating mode, and the documented TDP basis. Do not invent a physical power measurement. Keep this exact JSON file with the results produced by this device.

## 5. Smoke check

Use one stable device label in every command. Smoke uses real models, one five-frame replay block, and a small labelled subset. It confirms installation, paths, models, and output creation. It is not reportable evidence.

```powershell
python scripts\verify_process_load.py --intensity 0.75 --settle-s 3 --samples 8
python scripts\run_paper_suite.py --platform windows --backend onnx --smoke `
  --device <machine-label> `
  --frames D:\data\kitti\images\val `
  --kitti-labels D:\data\kitti\labels\val `
  --coco-images D:\data\coco\images\val2017 `
  --coco-labels D:\data\coco\labels\val2017
```

The load diagnostic is a setup check. It should show several process workers and substantial measured CPU activity. The requested target need not equal the measured value exactly. Calibration and recorded telemetry determine acceptable separation.

After smoke completes, inspect the newest `results\paper_*.json` file. It must report `"smoke": true`, `"simulated": false`, and no failed command. A smoke output proves the setup works but must not be copied into paper tables.

## 6. Full evaluation in phases

Run calibration once, retain its configuration, then run runtime phases in order. Steady profiles use `process_steady_v3`, with independent workers across all but one logical CPU. Burst uses `process_isolated_burst_v2`. Do not combine these results with historical thread-based load results.

```powershell
# Calibration
python scripts\run_paper_suite.py --phase calibration --platform windows --device <machine-label> --frames D:\data\kitti\images\val

# Runtime 1: idle and light. Runtime 2: moderate plus Pareto. Runtime 3: heavy plus Pareto. Runtime 4: burst.
python scripts\run_paper_suite.py --phase runtime1 --platform windows --backend onnx --device <machine-label> --energy-profile configs\energy_profile_windows.json
python scripts\run_paper_suite.py --phase runtime2 --platform windows --backend onnx --device <machine-label> --energy-profile configs\energy_profile_windows.json
python scripts\run_paper_suite.py --phase runtime3 --platform windows --backend onnx --device <machine-label> --energy-profile configs\energy_profile_windows.json
python scripts\run_paper_suite.py --phase runtime4 --platform windows --backend onnx --device <machine-label> --energy-profile configs\energy_profile_windows.json

# Measured KITTI and COCO accuracy, then KITTI VRU-retention sensitivity.
python scripts\run_paper_suite.py --phase accuracy --platform windows --backend onnx --device <machine-label>
python scripts\run_paper_suite.py --phase retention --platform windows --backend onnx --device <machine-label>
```

`RAMS_DATA_ROOT` supplies the KITTI and COCO paths. Do not change backend, models, energy profile, cooling, power mode, or calibrated configuration during the sequence.

After Sections 1 through 5, the single-command full evaluation is:

```powershell
python scripts\run_paper_suite.py --phase all --platform windows --backend onnx --device <machine-label> --energy-profile configs\energy_profile_windows.json
```

It uses 10 independent paired replay blocks of 200 frames per method and load setting by default, and can take many hours. Use the phase sequence when the machine cannot remain available continuously.

After every phase, inspect its newest `results\paper_<phase>_*.json` file before continuing. Stop if any command has `"ok": false`. Do not rerun only selected methods to replace an unfavourable result; rerun the whole failed phase after fixing its stated error.

## 7. Required handoff

Keep the complete `results/` directory, not just charts or summaries. A complete result handoff has raw CSV records, JSON summaries, replay manifests, calibration snapshots, model and backend details, and the completed energy profile. A valid full manifest reports `smoke: false` and `simulated: false` and has no failed stage.

Report COCO mAP only where records state `map_source: ultralytics_val`. It is tier-level supplementary context. Policy-level KITTI metrics are the primary accuracy evidence.

When handing results to the project owner, zip the complete device `results` directory and include the completed energy-profile JSON. State the device label, Windows version, CPU model, available RAM, whether NVIDIA telemetry was available, and whether the full `paper_all` manifest completed without failures.
