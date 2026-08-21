"""Resource monitoring for RAMS.

The monitor exposes CPU, memory, thermal, battery, and optional accelerator
signals as one calibrated pressure input.  NVIDIA support is intentionally
optional: a CPU-only host keeps the original pressure semantics.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol

import psutil

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcceleratorSnapshot:
    """Best-effort accelerator telemetry, normalized only where noted."""

    utilization_percent: Optional[float] = None
    memory_fraction: Optional[float] = None
    temperature_c: Optional[float] = None
    clock_mhz: Optional[float] = None
    source: str = "none"


class AcceleratorProvider(Protocol):
    """Small provider contract so platform telemetry never becomes required."""

    def sample(self) -> AcceleratorSnapshot:
        ...


class NullAcceleratorProvider:
    def sample(self) -> AcceleratorSnapshot:
        return AcceleratorSnapshot()


class NvmlAcceleratorProvider:
    """NVIDIA telemetry provider. Importing ``pynvml`` remains optional."""

    def __init__(self) -> None:
        import pynvml  # type: ignore[import-not-found]

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample(self) -> AcceleratorSnapshot:
        n = self._nvml
        try:
            util = n.nvmlDeviceGetUtilizationRates(self._handle).gpu
            memory = n.nvmlDeviceGetMemoryInfo(self._handle)
            temp = n.nvmlDeviceGetTemperature(self._handle, n.NVML_TEMPERATURE_GPU)
            clock = n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_GRAPHICS)
            return AcceleratorSnapshot(
                utilization_percent=float(util),
                memory_fraction=(memory.used / memory.total) if memory.total else None,
                temperature_c=float(temp),
                clock_mhz=float(clock),
                source="nvml",
            )
        except Exception as exc:  # telemetry must never stop inference
            logger.debug("[RAMS] NVML sample unavailable: %s", exc)
            return AcceleratorSnapshot(source="nvml")


class TegraStatsAcceleratorProvider:
    """Jetson fallback using one bounded ``tegrastats`` sample per poll."""

    _GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
    _RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
    _TEMP_RE = re.compile(r"GPU@(\d+(?:\.\d+)?)C")
    _CLOCK_RE = re.compile(r"GR3D_FREQ\s+\d+%@?(\d+)?")

    def sample(self) -> AcceleratorSnapshot:
        try:
            completed = subprocess.run(
                ["tegrastats", "--interval", "100", "--count", "1"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            line = (completed.stdout or completed.stderr).strip().splitlines()[-1]
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            logger.debug("[RAMS] tegrastats sample unavailable: %s", exc)
            return AcceleratorSnapshot(source="tegrastats")

        gpu = self._GPU_RE.search(line)
        ram = self._RAM_RE.search(line)
        temp = self._TEMP_RE.search(line)
        clock = self._CLOCK_RE.search(line)
        return AcceleratorSnapshot(
            utilization_percent=float(gpu.group(1)) if gpu else None,
            memory_fraction=(float(ram.group(1)) / float(ram.group(2))) if ram else None,
            temperature_c=float(temp.group(1)) if temp else None,
            clock_mhz=float(clock.group(1)) if clock and clock.group(1) else None,
            source="tegrastats",
        )


@lru_cache(maxsize=1)
def make_accelerator_provider() -> AcceleratorProvider:
    """Prefer NVML, fall back to Jetson telemetry, otherwise expose no GPU."""
    try:
        return NvmlAcceleratorProvider()
    except Exception:
        if shutil.which("tegrastats"):
            return TegraStatsAcceleratorProvider()
    return NullAcceleratorProvider()


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    cpu_temp: Optional[float]
    battery_percent: Optional[float]
    pressure_index: float
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_fraction: Optional[float] = None
    gpu_temp: Optional[float] = None
    gpu_clock_mhz: Optional[float] = None
    accelerator_source: str = "none"


def _read_cpu_temp() -> Optional[float]:
    """Read CPU temperature across platforms."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
            if key in temps and temps[key]:
                return max(reading.current for reading in temps[key])
        for readings in temps.values():
            if readings:
                return max(reading.current for reading in readings)
    except (AttributeError, OSError):
        pass
    return None


def _read_battery() -> Optional[float]:
    """Return battery percent or ``None`` if on AC or unavailable."""
    try:
        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged:
            return battery.percent
    except (AttributeError, OSError):
        pass
    return None


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def compute_pressure(
    cpu: float,
    mem: float,
    temp: Optional[float],
    battery: Optional[float],
    gpu_util: Optional[float] = None,
    gpu_mem: Optional[float] = None,
    w_cpu: float = 0.50,
    w_mem: float = 0.25,
    w_temp: float = 0.15,
    w_bat: float = 0.10,
    w_gpu_util: float = 0.20,
    w_gpu_mem: float = 0.10,
    temp_critical: float = 90.0,
    temp_warn: float = 70.0,
) -> float:
    """Return a scalar pressure index in ``[0, 1]``.

    GPU weights apply only when an accelerator signal is available, preserving
    historical CPU-only behaviour. Missing individual signals are charged to
    CPU, matching RAMS's conservative existing fallback.
    """
    # Preserve the original final-clamp semantics for malformed/over-range host
    # samples; only normalized accelerator fractions are clamped individually.
    cpu_unit = cpu / 100.0
    mem_unit = mem / 100.0
    pressure = cpu_unit * w_cpu + mem_unit * w_mem

    if temp is None:
        pressure += cpu_unit * w_temp
    else:
        pressure += _clamp_unit((temp - temp_warn) / (temp_critical - temp_warn)) * w_temp

    if battery is None:
        pressure += cpu_unit * w_bat
    else:
        pressure += (1.0 - _clamp_unit(battery / 100.0)) * w_bat

    accelerator_present = gpu_util is not None or gpu_mem is not None
    if accelerator_present:
        pressure += (cpu_unit if gpu_util is None else _clamp_unit(gpu_util / 100.0)) * w_gpu_util
        pressure += (cpu_unit if gpu_mem is None else _clamp_unit(gpu_mem)) * w_gpu_mem

    return _clamp_unit(pressure)


class ResourceMonitor:
    """Background monitor that samples host and optional accelerator pressure."""

    def __init__(
        self,
        hz: float = 10.0,
        gpu_util_weight: float = 0.20,
        gpu_memory_weight: float = 0.10,
    ):
        if hz <= 0:
            raise ValueError(f"ResourceMonitor hz must be positive, got {hz}")
        self.hz = hz
        self._interval = 1.0 / hz
        self.gpu_util_weight = gpu_util_weight
        self.gpu_memory_weight = gpu_memory_weight
        self._accelerator = make_accelerator_provider()
        self._snapshot: Optional[ResourceSnapshot] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            logger.warning("[RAMS] ResourceMonitor already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        psutil.cpu_percent(interval=None)
        time.sleep(0.1)
        while self._running:
            started = time.monotonic()
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent
            cpu_temp = _read_cpu_temp()
            battery = _read_battery()
            accelerator = self._accelerator.sample()
            # A Jetson GPU temperature is a better accelerator thermal signal,
            # but pressure intentionally receives one thermal contribution only.
            thermal = accelerator.temperature_c if accelerator.temperature_c is not None else cpu_temp
            pressure = compute_pressure(
                cpu, memory, thermal, battery,
                accelerator.utilization_percent, accelerator.memory_fraction,
                w_gpu_util=self.gpu_util_weight,
                w_gpu_mem=self.gpu_memory_weight,
            )
            snapshot = ResourceSnapshot(
                timestamp=time.time(), cpu_percent=cpu, memory_percent=memory,
                cpu_temp=cpu_temp, battery_percent=battery, pressure_index=pressure,
                gpu_utilization_percent=accelerator.utilization_percent,
                gpu_memory_fraction=accelerator.memory_fraction,
                gpu_temp=accelerator.temperature_c,
                gpu_clock_mhz=accelerator.clock_mhz,
                accelerator_source=accelerator.source,
            )
            with self._lock:
                self._snapshot = snapshot
            time.sleep(max(0.0, self._interval - (time.monotonic() - started)))

    @property
    def snapshot(self) -> Optional[ResourceSnapshot]:
        with self._lock:
            return self._snapshot

    @property
    def pressure(self) -> float:
        snapshot = self.snapshot
        return snapshot.pressure_index if snapshot else 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
