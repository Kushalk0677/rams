"""
RAMS Controller
The top-level runtime that ties ResourceMonitor, ModelLibrary,
and a switching Policy into a single inference entry point.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from rams.monitor import ResourceMonitor
from rams.models  import ModelLibrary, Tier
from rams.policy  import BasePolicy, SafetyPolicy, make_policy

logger = logging.getLogger(__name__)


class RAMSController:
    """
    Usage
    -----
        ctrl = RAMSController(simulate=True, policy="safety")
        ctrl.start()

        result = ctrl.infer(frame=my_frame)
        print(result["tier"], result["latency_ms"])

        ctrl.stop()

    Or as a context manager:

        with RAMSController(simulate=False, policy="threshold") as ctrl:
            result = ctrl.infer()
    """

    def __init__(
        self,
        simulate: bool = False,
        policy: str | BasePolicy = "safety",
        monitor_hz: float = 10.0,
        policy_kwargs: Optional[dict] = None,
    ):
        self.simulate = simulate
        self.monitor  = ResourceMonitor(hz=monitor_hz)
        self.library  = ModelLibrary(simulate=simulate)

        if isinstance(policy, str):
            self.policy = make_policy(policy, **(policy_kwargs or {}))
        else:
            self.policy = policy

        self._current_tier: Tier = Tier.SMALL
        self._switch_log: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        logger.info("[RAMS] Starting resource monitor ...")
        self.monitor.start()
        logger.info("[RAMS] Loading model tiers ...")
        self.library.load_all()
        logger.info("[RAMS] Controller ready. Policy: %s", self.policy.name)

    def stop(self):
        self.monitor.stop()
        self.library.unload_all()
        logger.info("[RAMS] Controller stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, frame=None) -> dict:
        """
        1. Read current resource pressure.
        2. Run switching policy → possibly change tier.
        3. Run inference on selected tier.
        4. Return result dict.
        """
        snap = self.monitor.snapshot
        _override = getattr(self, '_pressure_override', None)
        pressure = float(_override) if _override is not None else (snap.pressure_index if snap else 0.0)

        # Most recent detections for safety override
        new_tier = self.policy.select_tier(
            pressure=pressure,
            last_tier=self._current_tier,
            recent_detections=None,  # populated after first result
        )

        if new_tier != self._current_tier:
            logger.info(
                "[RAMS] Tier switch: %s → %s  (R=%.3f)",
                self._current_tier.name, new_tier.name, pressure,
            )
            self._switch_log.append({
                "ts": time.time(),
                "from": self._current_tier.name,
                "to":   new_tier.name,
                "pressure": pressure,
            })
            self._current_tier = new_tier

        result = self.library.infer(self._current_tier, frame)

        # Feed detections back to safety policy
        if isinstance(self.policy, SafetyPolicy):
            self.policy.select_tier(
                pressure=pressure,
                last_tier=self._current_tier,
                recent_detections=result.get("detections", []),
            )

        result["pressure"] = round(pressure, 4)
        if snap:
            result["cpu_pct"]  = snap.cpu_percent
            result["mem_pct"]  = snap.memory_percent
            result["cpu_temp"] = snap.cpu_temp
            result["battery"]  = snap.battery_percent

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def current_tier(self) -> Tier:
        return self._current_tier

    @property
    def switch_log(self) -> list[dict]:
        return list(self._switch_log)

    def status(self) -> dict:
        snap = self.monitor.snapshot
        return {
            "policy":       self.policy.name,
            "current_tier": self._current_tier.name,
            "pressure":     snap.pressure_index if snap else None,
            "cpu_pct":      snap.cpu_percent    if snap else None,
            "mem_pct":      snap.memory_percent if snap else None,
            "cpu_temp":     snap.cpu_temp       if snap else None,
            "battery":      snap.battery_percent if snap else None,
            "n_switches":   len(self._switch_log),
        }


    # convenience for sandbox/simulation environments
    def set_pressure_override(self, value):
        """Inject a synthetic R(t) value, bypassing psutil-based monitor."""
        self._pressure_override = value if value is not None else None
