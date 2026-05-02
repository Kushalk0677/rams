"""
Experiment 9 — Multi-Device Benchmark
======================================
Runs the full RAMS policy comparison across multiple hardware targets:
  - Jetson Orin (TensorRT backend)
  - RTX 6000 / RTX 3060 (CUDA + ONNX or TensorRT)
  - CPU baseline (existing Spectre results)

This experiment is designed to be run independently on each device
and then aggregated via exp9_aggregate.py. Results are tagged with
device name and backend so they can be compared cleanly.

Usage — run this on each device
---------------------------------
  # Jetson Orin (TensorRT)
  python experiments/exp9_multidevice.py \\
      --device jetson-orin \\
      --backend tensorrt \\
      --frames /data/kitti/images/val \\
      --n 200

  # RTX 6000 (ONNX on CUDA)
  python experiments/exp9_multidevice.py \\
      --device rtx6000 \\
      --backend onnx \\
      --frames /data/kitti/images/val \\
      --n 200

  # RTX 3060
  python experiments/exp9_multidevice.py \\
      --device rtx3060 \\
      --backend onnx \\
      --frames /data/kitti/images/val \\
      --n 200

After collecting results from all devices, aggregate:
  python experiments/exp9_aggregate.py --results-dir results/multidevice/

Outputs (per device run)
------------------------
  results/multidevice/exp9_<device>_<backend>_<timestamp>.csv
  results/multidevice/exp9_<device>_<backend>_<timestamp>.json
"""

import argparse
import csv
import json
import logging
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.controller import RAMSController
from rams.policy import POLICIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "multidevice"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VRU_CLASSES = {"person", "pedestrian", "cyclist", "bicycle", "motorbike", "motorcycle", "rider"}

LOAD_PROFILES = {
    "idle":     0.00,
    "light":    0.25,
    "moderate": 0.50,
    "heavy":    0.75,
    "burst":    1.00,
}


# ---------------------------------------------------------------------------
# TensorRT export helper
# ---------------------------------------------------------------------------

def export_tensorrt_models(imgsz_map: dict[str, int]):
    """Export YOLOv8 models to TensorRT engine if not already done."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        sys.exit(1)

    engines = {}
    for tier, (model_id, imgsz) in imgsz_map.items():
        stem = Path(model_id).stem
        engine_candidates = [Path(f"{stem}.engine"), Path(f"{stem}_imgsz{imgsz}.engine")]
        engine_path = next((p for p in engine_candidates if p.exists()), None)
        if engine_path is None:
            logger.info("Exporting %s → TensorRT (imgsz=%d) ...", model_id, imgsz)
            m = YOLO(model_id)
            exported = Path(m.export(format="engine", imgsz=imgsz, half=True, device=0, workspace=4))
            engine_path = exported if exported.exists() else engine_candidates[0]
            logger.info("Exported: %s", engine_path)
        else:
            logger.info("TensorRT engine already exists: %s", engine_path)
        engines[tier] = str(engine_path)
    return engines


# ---------------------------------------------------------------------------
# Load injector
# ---------------------------------------------------------------------------

class LoadInjector:
    def __init__(self, intensity: float = 0.0):
        self.intensity = intensity
        self._stop = threading.Event()
        self._threads = []

    def _burn(self):
        while not self._stop.is_set():
            _ = sum(i * i for i in range(1000))
            time.sleep(max(0.0, (1.0 - self.intensity) * 0.001))

    def start(self):
        n = max(0, int(self.intensity * 4))
        for _ in range(n):
            t = threading.Thread(target=self._burn, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Frame loader
# ---------------------------------------------------------------------------

def load_frame_paths(frames_dir: str) -> list[Path]:
    p = Path(frames_dir)
    paths = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")))
    if not paths:
        raise ValueError(f"No images found in {frames_dir}")
    logger.info("Loaded %d frames from %s", len(paths), frames_dir)
    return paths


def read_frame(path: Path):
    import cv2
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not read: {path}")
    return frame


# ---------------------------------------------------------------------------
# GPU utilization sampler
# ---------------------------------------------------------------------------

class GPUSampler:
    def __init__(self):
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._available = True
            logger.info("GPU utilization sampling enabled")
        except Exception:
            logger.info("pynvml not available — GPU util not sampled")

    def start(self):
        if not self._available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        import pynvml
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(util.gpu)
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._samples:
            return round(sum(self._samples) / len(self._samples), 1)
        return None


# ---------------------------------------------------------------------------
# Single policy × profile run
# ---------------------------------------------------------------------------

def run_policy(policy_name: str, n_inferences: int, load_intensity: float,
               profile_label: str, frame_paths: list, device_label: str,
               backend: str, simulate: bool) -> list[dict]:

    injector = LoadInjector(intensity=load_intensity)
    gpu_sampler = GPUSampler()
    injector.start()
    gpu_sampler.start()

    records = []
    frame_idx = 0

    try:
        with RAMSController(simulate=simulate, policy=policy_name) as ctrl:
            time.sleep(0.5)

            for i in range(n_inferences):
                frame = None
                frame_name = None
                if frame_paths:
                    path = frame_paths[frame_idx % len(frame_paths)]
                    frame_idx += 1
                    frame_name = path.name
                    try:
                        frame = read_frame(path)
                    except IOError:
                        frame = None

                t0 = time.perf_counter()
                result = ctrl.infer(frame=frame)
                t1 = time.perf_counter()

                detections = result.get("detections", [])
                vru_detected = any(
                    str(d.get("class", "")).lower() in VRU_CLASSES for d in detections
                )

                records.append({
                    "run_idx":        i,
                    "device":         device_label,
                    "backend":        backend,
                    "policy":         policy_name,
                    "load_profile":   profile_label,
                    "load_intensity": load_intensity,
                    "tier":           result["tier"],
                    "latency_ms":     round(result["latency_ms"], 3),
                    "wall_ms":        round((t1 - t0) * 1000, 3),
                    "pressure":       result.get("pressure"),
                    "cpu_pct":        result.get("cpu_pct"),
                    "mem_pct":        result.get("mem_pct"),
                    "cpu_temp":       result.get("cpu_temp"),
                    "n_detections":   len(detections),
                    "vru_detected":   vru_detected,
                    "frame":          frame_name or "null",
                    "simulated":      result.get("simulated", True),
                })

                if (i + 1) % 50 == 0:
                    logger.info("  [%s/%s/%s] %d/%d tier=%-6s latency=%.1f ms",
                                device_label, policy_name, profile_label,
                                i + 1, n_inferences, result["tier"], result["latency_ms"])
    finally:
        injector.stop()
        avg_gpu = gpu_sampler.stop()
        if avg_gpu is not None:
            for r in records:
                r["gpu_util_pct"] = avg_gpu

    return records


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(records: list[dict], device: str, backend: str) -> dict:
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for r in records:
        groups[(r["policy"], r["load_profile"])].append(r)

    summary = {
        "device":   device,
        "backend":  backend,
        "host":     socket.gethostname(),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "groups":   [],
    }

    for (policy, profile), recs in sorted(groups.items()):
        latencies   = [r["latency_ms"] for r in recs]
        tiers       = [r["tier"] for r in recs]
        tier_counts = {t: tiers.count(t) for t in set(tiers)}
        vru_rate    = sum(1 for r in recs if r.get("vru_detected")) / len(recs)
        gpu_utils   = [r["gpu_util_pct"] for r in recs if "gpu_util_pct" in r]
        gpu_mean    = round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else None

        # Switching count
        switches = sum(1 for i in range(1, len(tiers)) if tiers[i] != tiers[i-1])

        summary["groups"].append({
            "policy":          policy,
            "load_profile":    profile,
            "n":               len(recs),
            "latency_mean":    round(statistics.mean(latencies), 2),
            "latency_std":     round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
            "latency_p50":     round(statistics.median(latencies), 2),
            "latency_p95":     round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "tier_counts":     tier_counts,
            "tier_switches":   switches,
            "vru_rate":        round(vru_rate, 4),
            "gpu_util_mean":   gpu_mean,
        })

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 9 — Multi-Device Benchmark")
    parser.add_argument("--device",   required=True,
                        help="Device label e.g. jetson-orin, rtx6000, rtx3060, spectre-x360")
    parser.add_argument("--backend",  required=True, choices=["tensorrt", "onnx", "pytorch"],
                        help="Inference backend")
    parser.add_argument("--frames",   required=True, help="Path to image directory")
    parser.add_argument("--n",        type=int, default=200, help="Inferences per combo")
    parser.add_argument("--policy",   default="all",
                        help="all | threshold | predictive | safety | adaptive | safety2")
    parser.add_argument("--profile",  default="all",
                        help="all | idle | light | moderate | heavy | burst")
    parser.add_argument("--simulate", action="store_true", default=False,
                        help="Use simulated inference (no real models)")
    parser.add_argument("--export-trt", action="store_true",
                        help="Export TensorRT engines before benchmarking (Jetson only)")
    parser.add_argument("--export-only", action="store_true",
                        help="Export TensorRT engines and exit without benchmarking")
    args = parser.parse_args()

    # Optionally export TensorRT engines
    if args.export_trt and args.backend == "tensorrt":
        export_tensorrt_models({
            "NANO":   ("yolov8n.pt", 320),
            "SMALL":  ("yolov8s.pt", 416),
            "MEDIUM": ("yolov8m.pt", 640),
        })
        if args.export_only:
            return

    frame_paths = load_frame_paths(args.frames)

    policies = list(POLICIES.keys()) if args.policy == "all" else [args.policy]
    profiles = (
        list(LOAD_PROFILES.items()) if args.profile == "all"
        else [(args.profile, LOAD_PROFILES[args.profile])]
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"exp9_{args.device}_{args.backend}_{ts}"
    logger.info("=== Multi-Device Benchmark  run_id=%s ===", run_id)
    logger.info("Device=%s  Backend=%s  N=%d  Policies=%s  Profiles=%s",
                args.device, args.backend, args.n, policies, [p for p, _ in profiles])

    all_records = []

    for policy_name in policies:
        for profile_label, intensity in profiles:
            logger.info("--- Policy: %s  Load: %s ---", policy_name, profile_label)
            records = run_policy(
                policy_name=policy_name,
                n_inferences=args.n,
                load_intensity=intensity,
                profile_label=profile_label,
                frame_paths=frame_paths,
                device_label=args.device,
                backend=args.backend,
                simulate=args.simulate,
            )
            all_records.extend(records)

    # Save CSV
    csv_path = RESULTS_DIR / f"{run_id}.csv"
    if all_records:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            w.writeheader()
            w.writerows(all_records)
        logger.info("CSV -> %s", csv_path)

    # Save JSON
    summary = compute_summary(all_records, args.device, args.backend)
    json_path = RESULTS_DIR / f"{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("JSON -> %s", json_path)

    # Print table
    print("\n" + "=" * 90)
    print(f"Device: {args.device}  Backend: {args.backend}")
    print(f"{'Policy':<14} {'Profile':<10} {'N':>5}  {'Mean ms':>8}  "
          f"{'P95 ms':>8}  {'Switches':>9}  {'VRU%':>6}  Tiers")
    print("-" * 90)
    for g in summary["groups"]:
        tier_str = ", ".join(f"{k}:{v}" for k, v in sorted(g["tier_counts"].items()))
        print(
            f"{g['policy']:<14} {g['load_profile']:<10} {g['n']:>5}  "
            f"{g['latency_mean']:>8.1f}  {g['latency_p95']:>8.1f}  "
            f"{g['tier_switches']:>9}  {g['vru_rate']*100:>5.1f}%  {tier_str}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
