"""
Model Tier Library
Defines the three YOLOv8 tiers and their warm-loaded inference wrappers.

Inference backend priority (per tier):
  1. TensorRT      — used on Jetson/GPU if .engine file exists
  2. ONNX Runtime  — fastest on CPU, used if .onnx file exists
  3. Ultralytics   — fallback if only .pt file is available
  4. Simulation    — calibrated Gaussian, used if no real model found

Per-tier resolution (mixed-resolution strategy):
  NANO   → imgsz=320  (speed priority)
  SMALL  → imgsz=416  (balance)
  MEDIUM → imgsz=640  (accuracy priority — used under VRU safety override)

To export ONNX models:
    python -c "
    from ultralytics import YOLO
    YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)
    YOLO('yolov8s.pt').export(format='onnx', imgsz=416, opset=12)
    YOLO('yolov8m.pt').export(format='onnx', imgsz=640, opset=12)
    "
"""

from __future__ import annotations

import time
import random
import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    NANO   = 1
    SMALL  = 2
    MEDIUM = 3


@dataclass(frozen=True)
class TierProfile:
    tier: Tier
    model_id: str
    onnx_id:  str
    label: str
    imgsz: int             # per-tier inference resolution
    latency_mean_ms: float
    latency_std_ms:  float
    map50: float


PROFILES: dict[Tier, TierProfile] = {
    Tier.NANO: TierProfile(
        tier=Tier.NANO, model_id="yolov8n.pt", onnx_id="yolov8n.onnx",
        label="YOLOv8-nano", imgsz=320,
        latency_mean_ms=18.0, latency_std_ms=2.5, map50=0.372,
    ),
    Tier.SMALL: TierProfile(
        tier=Tier.SMALL, model_id="yolov8s.pt", onnx_id="yolov8s.onnx",
        label="YOLOv8-small", imgsz=416,
        latency_mean_ms=32.0, latency_std_ms=4.0, map50=0.448,
    ),
    Tier.MEDIUM: TierProfile(
        tier=Tier.MEDIUM, model_id="yolov8m.pt", onnx_id="yolov8m.onnx",
        label="YOLOv8-medium", imgsz=640,
        latency_mean_ms=58.0, latency_std_ms=6.5, map50=0.503,
    ),
}

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


def _parse_onnx_output(outputs, conf_thresh: float = 0.25) -> list[dict]:
    """Parse a YOLOv8 ONNX Runtime output array into detection dicts.

    Expects output shape (1, 84, 8400) from a YOLOv8 ONNX export.
    Each detection dict has keys: class, conf, xyxy (list of 4 floats).

    Args:
        outputs: Raw ONNX model output (list of numpy arrays).
        conf_thresh: Minimum confidence threshold.

    Returns:
        List of detection dicts, may be empty.
    """
    import numpy as np
    preds      = np.squeeze(outputs[0], axis=0).T   # (8400, 84)
    boxes      = preds[:, :4]
    scores     = preds[:, 4:]
    class_ids  = np.argmax(scores, axis=1)
    class_conf = scores[np.arange(len(scores)), class_ids]
    mask       = class_conf >= conf_thresh
    detections = []
    for (cx, cy, w, h), cls_id, conf in zip(boxes[mask], class_ids[mask], class_conf[mask]):
        detections.append({
            "class": COCO_NAMES[int(cls_id)] if int(cls_id) < len(COCO_NAMES) else str(cls_id),
            "conf":  float(conf),
            "xyxy":  [float(cx-w/2), float(cy-h/2), float(cx+w/2), float(cy+h/2)],
        })
    return detections


def _letterbox_image(frame, size: int):
    """Resize and pad a frame to a square canvas for YOLOv8 inference.

    Maintains aspect ratio, fills unused areas with gray (114).
    Returns the canvas, scale factor, padding offsets, and original dims.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=frame.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y, w, h


def _scale_letterbox_boxes(detections: list[dict], scale: float,
                           pad_x: int, pad_y: int,
                           orig_w: int, orig_h: int) -> list[dict]:
    """Map detection boxes from letterbox canvas coords back to original frame.

    Clips boxes to the original frame dimensions and discards any that
    become invalid (zero-area) after scaling.
    """
    scaled = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        box = [
            max(0.0, min(orig_w, (x1 - pad_x) / scale)),
            max(0.0, min(orig_h, (y1 - pad_y) / scale)),
            max(0.0, min(orig_w, (x2 - pad_x) / scale)),
            max(0.0, min(orig_h, (y2 - pad_y) / scale)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        det = dict(det)
        det["xyxy"] = box
        scaled.append(det)
    return scaled


class ModelWrapper:
    """
    Wraps a YOLOv8 model. Backend priority: TensorRT > ONNX > Ultralytics > Simulation.
    Each tier runs at its own resolution (320 / 416 / 640).
    """

    def __init__(self, tier: Tier, simulate: bool = False):
        self.tier     = tier
        self.profile  = PROFILES[tier]
        self.simulate = simulate
        self.imgsz    = self.profile.imgsz     # per-tier resolution
        self._model:   Optional[Any] = None
        self._backend: str           = "simulation"
        self._loaded:  bool          = False

    def load(self):
        if self._loaded:
            return
        if self.simulate:
            self._backend = "simulation"
            self._loaded  = True
            logger.info("[RAMS] Tier %s: simulation mode (imgsz=%d).",
                        self.tier.name, self.imgsz)
            return

        # 1. TensorRT engine via Ultralytics. Engines are device-specific, so
        # prefer them only when explicitly present in the working directory.
        engine_candidates = [
            Path(self.profile.model_id).with_suffix(".engine"),
            Path(f"{Path(self.profile.model_id).stem}_imgsz{self.imgsz}.engine"),
        ]
        engine_path = next((p for p in engine_candidates if p.exists()), None)
        if engine_path is not None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(engine_path))
                self._backend = "tensorrt"
                import numpy as np
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype="uint8")
                self._model(dummy, verbose=False, imgsz=self.imgsz)
                logger.info("[RAMS] %s ready — TensorRT %s (imgsz=%d).",
                            self.profile.label, engine_path, self.imgsz)
                self._loaded = True
                return
            except Exception as e:
                logger.warning("[RAMS] TensorRT load failed (%s), trying ONNX.", e)

        # 2. ONNX Runtime
        onnx_path = Path(self.profile.onnx_id)
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._model   = ort.InferenceSession(
                    str(onnx_path), opts, providers=["CPUExecutionProvider"])
                self._backend = "onnx"
                import numpy as np
                dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype="float32")
                self._model.run(None, {self._model.get_inputs()[0].name: dummy})
                logger.info("[RAMS] %s ready — ONNX (imgsz=%d).",
                            self.profile.label, self.imgsz)
                self._loaded = True
                return
            except ImportError:
                logger.warning("[RAMS] onnxruntime not installed, trying ultralytics.")
            except Exception as e:
                logger.warning("[RAMS] ONNX load failed (%s), trying ultralytics.", e)

        # 3. Ultralytics / PyTorch
        try:
            from ultralytics import YOLO
            pt = Path(self.profile.model_id)
            self._model   = YOLO(str(pt) if pt.exists() else self.profile.model_id)
            self._backend = "ultralytics"
            import numpy as np
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype="uint8")
            self._model(dummy, verbose=False, imgsz=self.imgsz)
            logger.info("[RAMS] %s ready — Ultralytics (imgsz=%d).",
                        self.profile.label, self.imgsz)
        except Exception as e:
            logger.warning("[RAMS] Load failed (%s) — simulation fallback.", e)
            self.simulate = True
            self._backend = "simulation"

        self._loaded = True

    def infer(self, frame=None) -> dict:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        # ── Simulation ────────────────────────────────────────────────────────
        if self._backend == "simulation" or self._model is None:
            sim_ms = max(1.0, random.gauss(
                self.profile.latency_mean_ms, self.profile.latency_std_ms))
            time.sleep(sim_ms / 1000.0)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            base   = max(0, int(random.gauss(6, 1.5)))
            recall = {Tier.NANO: 0.70, Tier.SMALL: 0.86, Tier.MEDIUM: 1.00}[self.tier]
            n_det  = max(0, int(round(base * recall + random.gauss(0, 0.5))))
            mu     = {Tier.NANO: 0.52, Tier.SMALL: 0.62, Tier.MEDIUM: 0.72}[self.tier]
            std    = {Tier.NANO: 0.10, Tier.SMALL: 0.08, Tier.MEDIUM: 0.07}[self.tier]
            GENERIC    = ["car","truck","bus","traffic light","stop sign"]
            VRU        = ["person","cyclist","bicycle"]
            vru_recall = {Tier.NANO: 0.55, Tier.SMALL: 0.78, Tier.MEDIUM: 0.94}[self.tier]
            dets = []
            for _ in range(n_det):
                x1 = random.uniform(0, 540); y1 = random.uniform(0, 380)
                dets.append({
                    "class": random.choice(GENERIC),
                    "conf":  float(min(0.99, max(0.30, random.gauss(mu, std)))),
                    "xyxy":  [x1, y1,
                              min(x1+random.uniform(20,200), 640),
                              min(y1+random.uniform(20,160), 480)],
                })
            if random.random() < 0.20 and random.random() < vru_recall:
                prox = random.choice(["near","mid","far"])
                sz   = {"near":(120,220,160,250),
                        "mid": (50, 120, 80, 150),
                        "far": (20,  50, 30,  70)}[prox]
                x1 = random.uniform(50, 500); y1 = random.uniform(50, 380)
                dets.append({
                    "class":     random.choice(VRU),
                    "conf":      float(min(0.99, max(0.35, random.gauss(mu, std)))),
                    "xyxy":      [x1, y1,
                                  min(x1+random.uniform(sz[0],sz[1]), 640),
                                  min(y1+random.uniform(sz[2],sz[3]), 480)],
                    "proximity": prox,
                })
            return {
                "tier": self.tier.name, "simulated": True, "backend": "simulation",
                "latency_ms": latency_ms, "detections": dets,
                "accuracy_proxy": float(min(1.0, max(0.0,
                    random.gauss(self.profile.map50, 0.02)))),
            }

        # ── ONNX Runtime ──────────────────────────────────────────────────────
        if self._backend == "onnx":
            import numpy as np
            if frame is None:
                img = np.zeros((self.imgsz, self.imgsz, 3), dtype="uint8")
                letterbox_meta = None
            else:
                img, scale, pad_x, pad_y, orig_w, orig_h = _letterbox_image(frame, self.imgsz)
                letterbox_meta = (scale, pad_x, pad_y, orig_w, orig_h)
            inp  = (img.astype("float32") / 255.0).transpose(2, 0, 1)[None]
            name = self._model.get_inputs()[0].name
            out  = self._model.run(None, {name: inp})
            latency_ms = (time.perf_counter() - t0) * 1000.0
            detections = _parse_onnx_output(out)
            coords = "model"
            if letterbox_meta is not None:
                detections = _scale_letterbox_boxes(detections, *letterbox_meta)
                coords = "original"
            return {
                "tier": self.tier.name, "simulated": False, "backend": "onnx",
                "latency_ms": latency_ms, "coords": coords,
                "detections": detections,
                "accuracy_proxy": self.profile.map50,
            }

        # ── Ultralytics / PyTorch ─────────────────────────────────────────────
        if frame is None:
            import numpy as np
            frame = np.zeros((self.imgsz, self.imgsz, 3), dtype="uint8")
        res        = self._model(frame, verbose=False, imgsz=self.imgsz)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        dets = [
            {"class": r.names[int(b.cls)], "conf": float(b.conf),
             "xyxy":  b.xyxy.tolist()}
            for r in res for b in r.boxes
        ]
        return {
            "tier": self.tier.name, "simulated": False, "backend": self._backend,
            "latency_ms": latency_ms, "detections": dets,
            "accuracy_proxy": self.profile.map50,
        }

    def unload(self):
        self._model  = None
        self._loaded = False


class ModelLibrary:
    """Keeps all three tiers warm in memory simultaneously."""

    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self._models: dict[Tier, ModelWrapper] = {
            t: ModelWrapper(t, simulate=simulate) for t in Tier
        }

    def load_all(self):
        for m in self._models.values():
            m.load()

    def get(self, tier: Tier) -> ModelWrapper:
        return self._models[tier]

    def infer(self, tier: Tier, frame=None) -> dict:
        return self._models[tier].infer(frame)

    def unload_all(self):
        for m in self._models.values():
            m.unload()
