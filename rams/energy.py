"""Telemetry-conditioned, TDP-based energy estimates for RAMS.

The model is deliberately an estimate, not a physical power measurement:

    P = P_idle + P_cpu*u_cpu + P_mem*u_mem + P_gpu*u_gpu
    E = P * t_end_to_end

Every profile must name its source and operating mode.  Results are suitable
only for language such as "estimated energy under the documented power model".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


def _unit(value: Optional[float], divisor: float = 1.0) -> float:
    if value is None:
        return 0.0
    return min(1.0, max(0.0, float(value) / divisor))


@dataclass(frozen=True)
class TDPEstimate:
    """Backward-compatible constant-power estimate."""

    watts: float
    latency_ms: float
    joules: float
    label: str
    is_measured: bool = False


@dataclass(frozen=True)
class PowerProfile:
    """A documented device/backend power model.

    Dynamic watt terms are multiplied by normalized telemetry. ``max_power_w``
    caps the model at the declared TDP/power-mode budget.
    """

    name: str
    source: str
    idle_w: float
    cpu_dynamic_w: float
    memory_dynamic_w: float
    gpu_dynamic_w: float = 0.0
    max_power_w: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PowerProfile":
        required = ("name", "source", "idle_w", "cpu_dynamic_w", "memory_dynamic_w")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Energy profile missing required fields: {missing}")
        profile = cls(
            name=str(data["name"]), source=str(data["source"]),
            idle_w=float(data["idle_w"]), cpu_dynamic_w=float(data["cpu_dynamic_w"]),
            memory_dynamic_w=float(data["memory_dynamic_w"]),
            gpu_dynamic_w=float(data.get("gpu_dynamic_w", 0.0)),
            max_power_w=(float(data["max_power_w"]) if data.get("max_power_w") is not None else None),
        )
        if not profile.source.strip():
            raise ValueError("Energy profile source must identify the documented power basis")
        if any(value < 0 for value in (
            profile.idle_w, profile.cpu_dynamic_w, profile.memory_dynamic_w,
            profile.gpu_dynamic_w,
        )):
            raise ValueError("Energy profile watt fields must be non-negative")
        if profile.max_power_w is not None and profile.max_power_w <= 0:
            raise ValueError("max_power_w must be positive when supplied")
        return profile


def load_power_profile(path: str | Path) -> PowerProfile:
    """Load a JSON power profile supplied for one documented device/mode."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return PowerProfile.from_mapping(json.load(handle))


@dataclass(frozen=True)
class TelemetryEnergyEstimate:
    profile_name: str
    profile_source: str
    latency_ms: float
    estimated_power_w: float
    estimated_energy_j: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    dynamic_scale: float = 1.0
    is_measured: bool = False


def estimate_profile_energy(
    latency_ms: float,
    profile: PowerProfile,
    *,
    cpu_percent: Optional[float],
    memory_percent: Optional[float],
    gpu_utilization_percent: Optional[float],
    dynamic_scale: float = 1.0,
) -> TelemetryEnergyEstimate:
    """Estimate energy from one profile and the telemetry recorded per frame."""
    if latency_ms < 0:
        raise ValueError(f"latency_ms must be non-negative, got {latency_ms}")
    if dynamic_scale <= 0:
        raise ValueError(f"dynamic_scale must be positive, got {dynamic_scale}")
    cpu = _unit(cpu_percent, 100.0)
    memory = _unit(memory_percent, 100.0)
    gpu = _unit(gpu_utilization_percent, 100.0)
    dynamic_w = dynamic_scale * (
        profile.cpu_dynamic_w * cpu
        + profile.memory_dynamic_w * memory
        + profile.gpu_dynamic_w * gpu
    )
    power_w = profile.idle_w + dynamic_w
    if profile.max_power_w is not None:
        power_w = min(power_w, profile.max_power_w)
    return TelemetryEnergyEstimate(
        profile_name=profile.name,
        profile_source=profile.source,
        latency_ms=float(latency_ms),
        estimated_power_w=power_w,
        estimated_energy_j=power_w * float(latency_ms) / 1000.0,
        cpu_utilization=cpu,
        memory_utilization=memory,
        gpu_utilization=gpu,
        dynamic_scale=dynamic_scale,
    )


def estimate_tdp_energy(latency_ms: float, watts: Optional[float],
                        label: str = "user-supplied TDP") -> Optional[TDPEstimate]:
    """Return legacy constant-power ``P * t`` estimate, or ``None``."""
    if watts is None:
        return None
    if watts <= 0:
        raise ValueError(f"TDP watts must be positive, got {watts}")
    if latency_ms < 0:
        raise ValueError(f"latency_ms must be non-negative, got {latency_ms}")
    return TDPEstimate(
        watts=float(watts), latency_ms=float(latency_ms),
        joules=float(watts) * float(latency_ms) / 1000.0, label=label,
    )
