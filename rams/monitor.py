"""
Resource Monitor
Samples CPU, memory, thermal, and battery state at configurable Hz.
Produces a scalar resource pressure index R(t) in [0, 1].
"""

import time
import threading
import psutil
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float          # 0-100
    memory_percent: float       # 0-100
    cpu_temp: Optional[float]   # Celsius, None if unavailable
    battery_percent: Optional[float]  # 0-100, None if plugged in / unavailable
    pressure_index: float       # R(t) in [0, 1]


def _read_cpu_temp() -> Optional[float]:
    """Read CPU temperature across platforms."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
            if key in temps:
                readings = temps[key]
                if readings:
                    return max(r.current for r in readings)
        # fallback: first available sensor
        for readings in temps.values():
            if readings:
                return max(r.current for r in readings)
    except (AttributeError, OSError):
        pass
    return None


def _read_battery() -> Optional[float]:
    """Return battery percent or None if on AC / unavailable."""
    try:
        b = psutil.sensors_battery()
        if b is not None and not b.power_plugged:
            return b.percent
    except (AttributeError, OSError):
        pass
    return None


def compute_pressure(
    cpu: float,
    mem: float,
    temp: Optional[float],
    battery: Optional[float],
    w_cpu: float = 0.50,
    w_mem: float = 0.25,
    w_temp: float = 0.15,
    w_bat: float = 0.10,
    temp_critical: float = 90.0,
    temp_warn: float = 70.0,
) -> float:
    """
    Weighted pressure index R(t) in [0, 1].

    Weights (defaults):
        CPU utilisation  50%
        Memory           25%
        Thermal          15%
        Battery          10%

    Temperature is normalised between warn and critical thresholds.
    Battery pressure rises as charge drops (low battery = high pressure).
    If a signal is unavailable its weight is redistributed to CPU.
    """
    active_w = w_cpu
    pressure = (cpu / 100.0) * w_cpu + (mem / 100.0) * w_mem
    active_w += w_mem

    if temp is not None:
        t_norm = max(0.0, min(1.0, (temp - temp_warn) / (temp_critical - temp_warn)))
        pressure += t_norm * w_temp
        active_w += w_temp
    else:
        # redistribute thermal weight to CPU
        pressure += (cpu / 100.0) * w_temp

    if battery is not None:
        bat_pressure = 1.0 - (battery / 100.0)
        pressure += bat_pressure * w_bat
        active_w += w_bat
    else:
        pressure += (cpu / 100.0) * w_bat

    return min(1.0, max(0.0, pressure))


class ResourceMonitor:
    """
    Background thread that polls system resources at `hz` Hz.
    Access the latest snapshot via `.snapshot` or `.pressure`.
    """

    def __init__(self, hz: float = 10.0):
        self.hz = hz
        self._interval = 1.0 / hz
        self._snapshot: Optional[ResourceSnapshot] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        # warm up psutil cpu_percent (first call always returns 0)
        psutil.cpu_percent(interval=None)
        time.sleep(0.1)

        while self._running:
            t0 = time.monotonic()
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            temp = _read_cpu_temp()
            battery = _read_battery()
            pressure = compute_pressure(cpu, mem, temp, battery)

            snap = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu,
                memory_percent=mem,
                cpu_temp=temp,
                battery_percent=battery,
                pressure_index=pressure,
            )
            with self._lock:
                self._snapshot = snap

            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self._interval - elapsed)
            time.sleep(sleep_for)

    @property
    def snapshot(self) -> Optional[ResourceSnapshot]:
        with self._lock:
            return self._snapshot

    @property
    def pressure(self) -> float:
        snap = self.snapshot
        return snap.pressure_index if snap else 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
