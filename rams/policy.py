"""
Switching Policies
Three variants:
  1. ThresholdPolicy      — fixed R(t) thresholds with hysteresis
  2. PredictivePolicy     — EWMA short-horizon load forecasting
  3. SafetyPolicy         — threshold + class-conditional VRU override
                            locks to SMALL+ when vulnerable road users
                            detected within proximity window
                            min_conf lowered to 0.25 to compensate for
                            reduced recall at lower input resolutions
"""

from __future__ import annotations

import time
import logging
from collections import deque
from typing import Optional

from rams.models import Tier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BasePolicy:
    name: str = "base"

    def select_tier(
        self,
        pressure: float,
        last_tier: Tier,
        recent_detections: Optional[list[dict]] = None,
    ) -> Tier:
        raise NotImplementedError

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# 1. Threshold Policy
# ---------------------------------------------------------------------------

class ThresholdPolicy(BasePolicy):
    """
    Selects tier by R(t) thresholds with a hysteresis band to
    prevent rapid flapping between tiers.

    Default thresholds:
        R < lo_thresh             → MEDIUM  (resources plentiful)
        lo_thresh ≤ R < hi_thresh → SMALL   (moderate load)
        R ≥ hi_thresh             → NANO    (high load)

    Hysteresis: a tier change only commits if the new tier would
    have been selected for `hysteresis_window` consecutive samples.
    """

    name = "threshold"

    def __init__(
        self,
        lo_thresh: float = 0.45,
        hi_thresh: float = 0.72,
        hysteresis_window: int = 3,
    ):
        self.lo     = lo_thresh
        self.hi     = hi_thresh
        self.window = hysteresis_window
        self._candidate: Optional[Tier] = None
        self._candidate_count: int = 0

    def _raw_tier(self, pressure: float) -> Tier:
        if pressure >= self.hi:
            return Tier.NANO
        elif pressure >= self.lo:
            return Tier.SMALL
        else:
            return Tier.MEDIUM

    def select_tier(self, pressure, last_tier, recent_detections=None) -> Tier:
        raw = self._raw_tier(pressure)
        if raw == last_tier:
            self._candidate = None
            self._candidate_count = 0
            return last_tier
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1
        if self._candidate_count >= self.window:
            self._candidate = None
            self._candidate_count = 0
            return raw
        return last_tier

    def reset(self):
        self._candidate = None
        self._candidate_count = 0


# ---------------------------------------------------------------------------
# 2. Predictive Policy
# ---------------------------------------------------------------------------

class PredictivePolicy(BasePolicy):
    """
    EWMA-based pressure forecasting. Applies threshold logic on the
    forecast rather than the raw R(t) value, anticipating load spikes.
    """

    name = "predictive"

    def __init__(
        self,
        history_len: int   = 20,
        alpha: float       = 0.35,
        lo_thresh: float   = 0.40,
        hi_thresh: float   = 0.68,
        use_lstm: bool     = False,
    ):
        self.history_len = history_len
        self.alpha       = alpha
        self.lo          = lo_thresh
        self.hi          = hi_thresh
        self.use_lstm    = use_lstm
        self._history: deque[float] = deque(maxlen=history_len)
        self._ewma: Optional[float] = None
        self._lstm_model = None
        if use_lstm:
            self._try_load_lstm()

    def _try_load_lstm(self):
        try:
            import torch
            import torch.nn as nn

            class _PressureLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 16, batch_first=True)
                    self.fc   = nn.Linear(16, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return torch.sigmoid(self.fc(out[:, -1, :]))

            self._lstm_model = _PressureLSTM()
            logger.info("[RAMS] PredictivePolicy: LSTM ready.")
        except ImportError:
            logger.warning("[RAMS] torch unavailable — using EWMA.")
            self.use_lstm = False

    def _forecast(self, current: float) -> float:
        self._history.append(current)
        if self._ewma is None:
            self._ewma = current
        else:
            self._ewma = self.alpha * current + (1 - self.alpha) * self._ewma
        if self.use_lstm and self._lstm_model and len(self._history) >= 5:
            try:
                import torch
                seq = list(self._history)[-10:]
                x   = torch.tensor([[v] for v in seq],
                                   dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    return self._lstm_model(x).item()
            except Exception:
                pass
        return self._ewma

    def select_tier(self, pressure, last_tier, recent_detections=None) -> Tier:
        forecast = self._forecast(pressure)
        if forecast >= self.hi:
            return Tier.NANO
        elif forecast >= self.lo:
            return Tier.SMALL
        else:
            return Tier.MEDIUM

    def reset(self):
        self._history.clear()
        self._ewma = None


# ---------------------------------------------------------------------------
# 3. Safety-Constrained Policy  (primary contribution)
# ---------------------------------------------------------------------------

VULNERABLE_CLASSES = {
    "person", "pedestrian", "cyclist", "bicycle", "motorbike", "motorcycle"
}


class SafetyPolicy(BasePolicy):
    """
    Threshold policy + class-conditional VRU override.

    If a vulnerable road user was detected within `proximity_window_s`
    seconds, the tier is locked to SMALL or above regardless of pressure.

    min_conf defaults to 0.25 (lowered from 0.40) to compensate for
    reduced recall at NANO's 320px resolution — ensures the safety
    override fires reliably even when the small model misses weak detections.
    """

    name = "safety"

    def __init__(
        self,
        lo_thresh: float          = 0.45,
        hi_thresh: float          = 0.72,
        hysteresis_window: int    = 3,
        proximity_window_s: float = 0.5,
        min_conf: float           = 0.25,   # lowered for mixed-resolution robustness
    ):
        self._base              = ThresholdPolicy(lo_thresh, hi_thresh, hysteresis_window)
        self.proximity_window_s = proximity_window_s
        self.min_conf           = min_conf
        self._last_vru_time: Optional[float] = None

    def _update_vru(self, detections: Optional[list[dict]]):
        if not detections:
            return
        for det in detections:
            cls  = str(det.get("class", "")).lower()
            conf = float(det.get("conf", 0.0))
            if cls in VULNERABLE_CLASSES and conf >= self.min_conf:
                self._last_vru_time = time.monotonic()
                return

    def _vru_active(self) -> bool:
        if self._last_vru_time is None:
            return False
        return (time.monotonic() - self._last_vru_time) <= self.proximity_window_s

    def select_tier(self, pressure, last_tier, recent_detections=None) -> Tier:
        self._update_vru(recent_detections)
        tier = self._base.select_tier(pressure, last_tier, recent_detections)
        if self._vru_active() and tier == Tier.NANO:
            logger.debug("[RAMS] SafetyPolicy: VRU override → SMALL.")
            return Tier.SMALL
        return tier

    def reset(self):
        self._base.reset()
        self._last_vru_time = None


# ---------------------------------------------------------------------------
# 4. Adaptive Predictive Policy
# ---------------------------------------------------------------------------

class AdaptivePredictivePolicy(BasePolicy):
    """
    PredictivePolicy with self-tuning EWMA alpha based on rolling
    pressure variance. High volatility → higher alpha (faster response).
    Low volatility → lower alpha (smoother forecast).
    """

    name = "adaptive"

    def __init__(
        self,
        alpha_min: float = 0.10,
        alpha_max: float = 0.70,
        var_window: int  = 15,
        lo_thresh: float = 0.40,
        hi_thresh: float = 0.68,
    ):
        self.alpha_min  = alpha_min
        self.alpha_max  = alpha_max
        self.var_window = var_window
        self.lo         = lo_thresh
        self.hi         = hi_thresh
        self._history: deque[float] = deque(maxlen=var_window)
        self._ewma: Optional[float] = None

    def _adaptive_alpha(self) -> float:
        if len(self._history) < 3:
            return self.alpha_min
        import statistics as _st
        sigma      = _st.stdev(self._history)
        sigma_norm = min(1.0, sigma / 0.30)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * sigma_norm

    def _forecast(self, current: float) -> float:
        self._history.append(current)
        alpha = self._adaptive_alpha()
        if self._ewma is None:
            self._ewma = current
        else:
            self._ewma = alpha * current + (1.0 - alpha) * self._ewma
        return self._ewma

    def select_tier(self, pressure, last_tier, recent_detections=None) -> Tier:
        forecast = self._forecast(pressure)
        if forecast >= self.hi:
            return Tier.NANO
        elif forecast >= self.lo:
            return Tier.SMALL
        else:
            return Tier.MEDIUM

    def reset(self):
        self._history.clear()
        self._ewma = None


# ---------------------------------------------------------------------------
# 5. Two-Level Safety Policy
# ---------------------------------------------------------------------------

def _bbox_area(det: dict) -> float:
    xyxy = det.get("xyxy")
    if not xyxy or len(xyxy) < 4:
        return 0.0
    x1, y1, x2, y2 = xyxy[:4]
    return max(0.0, (x2 - x1) * (y2 - y1))


class SafetyTwoLevelPolicy(BasePolicy):
    """
    Two-level safety override using bounding-box area as proximity proxy.

    VRU detected + bbox >= near_area_thresh  →  lock to MEDIUM (near VRU)
    VRU detected + bbox <  near_area_thresh  →  lock to SMALL  (distant VRU)
    No VRU                                   →  base ThresholdPolicy

    min_conf defaults to 0.25 for same mixed-resolution robustness reasons
    as SafetyPolicy.
    """

    name = "safety2"

    def __init__(
        self,
        lo_thresh: float          = 0.45,
        hi_thresh: float          = 0.72,
        hysteresis_window: int    = 3,
        proximity_window_s: float = 0.5,
        min_conf: float           = 0.25,   # lowered for mixed-resolution robustness
        near_area_thresh: float   = 8_000.0,
    ):
        self._base              = ThresholdPolicy(lo_thresh, hi_thresh, hysteresis_window)
        self.proximity_window_s = proximity_window_s
        self.min_conf           = min_conf
        self.near_area_thresh   = near_area_thresh
        self._last_vru_time: Optional[float] = None
        self._last_vru_area: float           = 0.0

    def _update_vru(self, detections: Optional[list[dict]]):
        if not detections:
            return
        for det in detections:
            cls  = str(det.get("class", "")).lower()
            conf = float(det.get("conf", 0.0))
            if cls in VULNERABLE_CLASSES and conf >= self.min_conf:
                area = _bbox_area(det)
                if (self._last_vru_time is None
                        or area >= self._last_vru_area
                        or (time.monotonic() - self._last_vru_time) > self.proximity_window_s):
                    self._last_vru_time = time.monotonic()
                    self._last_vru_area = area

    def _vru_lock_tier(self) -> Optional[Tier]:
        if self._last_vru_time is None:
            return None
        if (time.monotonic() - self._last_vru_time) > self.proximity_window_s:
            return None
        return Tier.MEDIUM if self._last_vru_area >= self.near_area_thresh else Tier.SMALL

    def select_tier(self, pressure, last_tier, recent_detections=None) -> Tier:
        self._update_vru(recent_detections)
        base_tier = self._base.select_tier(pressure, last_tier, recent_detections)
        lock      = self._vru_lock_tier()
        if lock is not None and base_tier < lock:
            logger.debug("[RAMS] SafetyTwoLevel: VRU override (area=%.0f) → %s",
                         self._last_vru_area, lock.name)
            return lock
        return base_tier

    def reset(self):
        self._base.reset()
        self._last_vru_time = None
        self._last_vru_area = 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

POLICIES = {
    "threshold":  ThresholdPolicy,
    "predictive": PredictivePolicy,
    "safety":     SafetyPolicy,
    "adaptive":   AdaptivePredictivePolicy,
    "safety2":    SafetyTwoLevelPolicy,
}


def make_policy(name: str, **kwargs) -> BasePolicy:
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Choose from: {list(POLICIES)}")
    return POLICIES[name](**kwargs)
