"""
Tests for rams.monitor — resource pressure computation, snapshots, and
the background ResourceMonitor thread.

Fixtures used
-------------
- patched_psutil (conftest) — deterministic mocks for psutil / monitor internals
- patched_psutil_loaded (conftest) — high-load variant of the above
- mock_snapshot (conftest) — factory that builds ResourceSnapshot instances
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Helpers
# ===================================================================


def _wait_for_snapshot(mon: Any, timeout: float = 2.0) -> Any:
    """Spin until ``mon.snapshot`` is not ``None``, then return it."""
    from rams.monitor import ResourceSnapshot

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = mon.snapshot
        if snap is not None:
            return snap
        time.sleep(0.02)
    pytest.fail("ResourceMonitor did not produce a snapshot within "
                f"{timeout:.1f}s timeout")


# ===================================================================
# compute_pressure — basic calculation
# ===================================================================


class TestComputePressure:
    """``compute_pressure()`` basic calculation and clamping."""

    def test_idle_system(self) -> None:
        from rams.monitor import compute_pressure

        # CPU=10, MEM=40, temp=None, battery=None
        p = compute_pressure(10.0, 40.0, None, None)
        # CPU: 0.10 * 0.50 = 0.05
        # MEM: 0.40 * 0.25 = 0.10
        # temp redist to CPU: 0.10 * 0.15 = 0.015
        # battery redist to CPU: 0.10 * 0.10 = 0.01
        # Total = 0.175
        assert p == pytest.approx(0.175, rel=1e-3)

    def test_loaded_system(self) -> None:
        from rams.monitor import compute_pressure

        # CPU=85, MEM=75, temp=82, battery=15
        p = compute_pressure(85.0, 75.0, 82.0, 15.0)
        # CPU: 0.85 * 0.50 = 0.425
        # MEM: 0.75 * 0.25 = 0.1875
        # temp: (82-70)/(90-70)=0.6  →  0.6 * 0.15 = 0.09
        # battery: (1-0.15) * 0.10 = 0.085
        # Total = 0.7875
        assert p > 0.6
        assert p == pytest.approx(0.7875, rel=1e-3)

    def test_clamps_to_zero(self) -> None:
        from rams.monitor import compute_pressure

        p = compute_pressure(-10.0, -20.0, None, None)
        assert p == 0.0

    def test_clamps_to_one(self) -> None:
        from rams.monitor import compute_pressure

        p = compute_pressure(200.0, 200.0, 100.0, 100.0)
        assert p == 1.0

    def test_temp_normalization(self) -> None:
        from rams.monitor import compute_pressure

        # temp=80, warn=70, critical=90 → (80-70)/(90-70) = 0.5
        p = compute_pressure(0.0, 0.0, 80.0, None)
        # CPU=0, MEM=0, temp=0.5*0.15=0.075, battery redist=0
        assert p == pytest.approx(0.075, rel=1e-3)

    def test_temp_below_warn_is_zero(self) -> None:
        from rams.monitor import compute_pressure

        # temp=50 < warn=70 → t_norm clamped to 0
        p = compute_pressure(0.0, 0.0, 50.0, None)
        assert p == 0.0

    def test_temp_above_critical_is_one(self) -> None:
        from rams.monitor import compute_pressure

        # temp=100 > critical=90 → t_norm clamped to 1
        p = compute_pressure(0.0, 0.0, 100.0, None)
        # pressure = 0 + 0 + 1.0 * 0.15 + 0 = 0.15
        assert p == pytest.approx(0.15, rel=1e-3)

    def test_battery_full_gives_zero_pressure(self) -> None:
        from rams.monitor import compute_pressure

        p = compute_pressure(0.0, 0.0, None, 100.0)
        # battery pressure = 1 - 100/100 = 0
        assert p == 0.0

    def test_battery_empty_gives_full_weight(self) -> None:
        from rams.monitor import compute_pressure

        p = compute_pressure(0.0, 0.0, None, 0.0)
        # battery pressure = 1 - 0/100 = 1.0  →  1.0 * 0.10 = 0.10
        assert p == pytest.approx(0.10, rel=1e-3)


# ===================================================================
# compute_pressure — weight redistribution
# ===================================================================


class TestComputePressureWeightRedistribution:
    """Verify that missing signals redistribute their weight to CPU."""

    def test_temp_none_redistributes_to_cpu(self) -> None:
        from rams.monitor import compute_pressure

        # With temp=70 at the warn boundary, temp contribution is 0.
        # CPU=50, MEM=0, battery=None (redistributed)
        p_with = compute_pressure(50.0, 0.0, 70.0, None)
        # CPU 0.25 + MEM 0 + temp 0 + bat redist 0.05 = 0.30
        assert p_with == pytest.approx(0.30, rel=1e-3)

        # Without temp at all, the 0.15 weight goes to CPU.
        p_without = compute_pressure(50.0, 0.0, None, None)
        # CPU 0.25 + MEM 0 + temp redist 0.075 + bat redist 0.05 = 0.375
        assert p_without == pytest.approx(0.375, rel=1e-3)

        # The difference should be exactly (50/100) * 0.15 = 0.075
        assert p_without - p_with == pytest.approx(0.075, rel=1e-3)

    def test_battery_none_redistributes_to_cpu(self) -> None:
        from rams.monitor import compute_pressure

        # With battery=100% its contribution is 0.
        # CPU=50, MEM=0, temp=None (redistributed)
        p_with = compute_pressure(50.0, 0.0, None, 100.0)
        # CPU 0.25 + MEM 0 + temp redist 0.075 + bat 0 = 0.325
        assert p_with == pytest.approx(0.325, rel=1e-3)

        # Without battery at all, the 0.10 weight goes to CPU.
        p_without = compute_pressure(50.0, 0.0, None, None)
        # CPU 0.25 + MEM 0 + temp redist 0.075 + bat redist 0.05 = 0.375
        assert p_without == pytest.approx(0.375, rel=1e-3)

        # The difference should be exactly (50/100) * 0.10 = 0.05
        assert p_without - p_with == pytest.approx(0.05, rel=1e-3)


# ===================================================================
# ResourceSnapshot dataclass
# ===================================================================


class TestResourceSnapshot:
    """``ResourceSnapshot`` dataclass field types and bounds."""

    def test_fields_are_correct_types(self, mock_snapshot: Any) -> None:
        snap = mock_snapshot(cpu=25.0, mem=50.0, temp=70.0, battery=80.0,
                             ts=1234.0)
        assert isinstance(snap.timestamp, float)
        assert snap.timestamp == 1234.0
        assert isinstance(snap.cpu_percent, float)
        assert snap.cpu_percent == 25.0
        assert isinstance(snap.memory_percent, float)
        assert snap.memory_percent == 50.0
        assert snap.cpu_temp == 70.0
        assert snap.battery_percent == 80.0
        assert isinstance(snap.pressure_index, float)

    def test_pressure_index_in_range(self, mock_snapshot: Any) -> None:
        for cpu, mem in [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)]:
            snap = mock_snapshot(cpu=cpu, mem=mem)
            assert 0.0 <= snap.pressure_index <= 1.0

    def test_optional_fields_can_be_none(self, mock_snapshot: Any) -> None:
        snap = mock_snapshot(temp=None, battery=None)
        assert snap.cpu_temp is None
        assert snap.battery_percent is None


# ===================================================================
# ResourceMonitor thread lifecycle
# ===================================================================


class TestResourceMonitor:
    """``ResourceMonitor`` start / stop / snapshot / pressure lifecycle."""

    def test_start_creates_daemon_thread(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=50)
        mon.start()
        assert mon._thread is not None
        assert mon._thread.daemon is True
        mon.stop()

    def test_snapshot_is_none_before_start(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=50)
        assert mon.snapshot is None

    def test_pressure_is_zero_before_start(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=50)
        assert mon.pressure == 0.0

    def test_snapshot_available_after_start(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=50)
        mon.start()
        snap = _wait_for_snapshot(mon)
        mon.stop()
        assert snap is not None
        assert isinstance(snap.pressure_index, float)

    def test_pressure_returns_value_after_start(
        self, patched_psutil: dict
    ) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=50)
        mon.start()
        _wait_for_snapshot(mon)
        p = mon.pressure
        mon.stop()
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_pressure_value_matches_idle_system(
        self, patched_psutil: dict
    ) -> None:
        from rams.monitor import ResourceMonitor

        # patched_psutil gives CPU=10, MEM=40, temp=None, battery=None
        mon = ResourceMonitor(hz=50)
        mon.start()
        snap = _wait_for_snapshot(mon)
        mon.stop()
        # Expected pressure ≈ 0.175
        assert snap.pressure_index == pytest.approx(0.175, abs=0.02)

    def test_stop_joins_thread(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=50)
        mon.start()
        mon.stop()
        assert not mon._running
        assert not mon._thread.is_alive()

    def test_interval_from_hz(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        mon = ResourceMonitor(hz=20)
        assert mon.hz == 20
        assert mon._interval == 0.05  # 1/20

    def test_context_manager(self, patched_psutil: dict) -> None:
        from rams.monitor import ResourceMonitor

        with ResourceMonitor(hz=50) as mon:
            assert mon._running
            snap = _wait_for_snapshot(mon)
            assert snap is not None
        assert not mon._running


# ===================================================================
# _read_cpu_temp
# ===================================================================


class TestReadCpuTemp:
    """``_read_cpu_temp()`` — sensor detection."""

    def test_returns_none_when_no_sensors(self) -> None:
        from rams.monitor import _read_cpu_temp

        with patch("rams.monitor.psutil.sensors_temperatures",
                   return_value={}, create=True):
            assert _read_cpu_temp() is None

    def test_returns_float_when_sensors_present(self) -> None:
        from rams.monitor import _read_cpu_temp

        t = MagicMock()
        t.current = 75.0
        with patch("rams.monitor.psutil.sensors_temperatures",
                   return_value={"coretemp": [t]}, create=True):
            result = _read_cpu_temp()
            assert isinstance(result, float)
            assert result == 75.0

    def test_uses_max_across_multiple_readings(self) -> None:
        from rams.monitor import _read_cpu_temp

        t1 = MagicMock()
        t1.current = 60.0
        t2 = MagicMock()
        t2.current = 82.0
        with patch("rams.monitor.psutil.sensors_temperatures",
                   return_value={"coretemp": [t1, t2]}, create=True):
            result = _read_cpu_temp()
            assert result == 82.0

    def test_fallback_to_first_available_sensor(self) -> None:
        from rams.monitor import _read_cpu_temp

        t = MagicMock()
        t.current = 71.0
        # No known sensor key, should fall back to the first sensor group
        with patch("rams.monitor.psutil.sensors_temperatures",
                   return_value={"unknown_sensor": [t]}, create=True):
            result = _read_cpu_temp()
            assert result == 71.0

    def test_handles_attribute_error_gracefully(self) -> None:
        from rams.monitor import _read_cpu_temp

        with patch("rams.monitor.psutil.sensors_temperatures",
                   side_effect=AttributeError, create=True):
            assert _read_cpu_temp() is None

    def test_handles_os_error_gracefully(self) -> None:
        from rams.monitor import _read_cpu_temp

        with patch("rams.monitor.psutil.sensors_temperatures",
                   side_effect=OSError, create=True):
            assert _read_cpu_temp() is None


# ===================================================================
# _read_battery
# ===================================================================


class TestReadBattery:
    """``_read_battery()`` — AC / discharging detection."""

    def test_returns_none_when_on_ac(self) -> None:
        from rams.monitor import _read_battery

        bat = MagicMock()
        bat.power_plugged = True
        bat.percent = 80.0
        with patch("rams.monitor.psutil.sensors_battery",
                   return_value=bat):
            assert _read_battery() is None

    def test_returns_float_when_discharging(self) -> None:
        from rams.monitor import _read_battery

        bat = MagicMock()
        bat.power_plugged = False
        bat.percent = 65.0
        with patch("rams.monitor.psutil.sensors_battery",
                   return_value=bat):
            result = _read_battery()
            assert isinstance(result, float)
            assert result == 65.0

    def test_returns_none_when_sensors_battery_is_none(self) -> None:
        from rams.monitor import _read_battery

        with patch("rams.monitor.psutil.sensors_battery",
                   return_value=None):
            assert _read_battery() is None

    def test_handles_attribute_error_gracefully(self) -> None:
        from rams.monitor import _read_battery

        with patch("rams.monitor.psutil.sensors_battery",
                   side_effect=AttributeError):
            assert _read_battery() is None

    def test_handles_os_error_gracefully(self) -> None:
        from rams.monitor import _read_battery

        with patch("rams.monitor.psutil.sensors_battery",
                   side_effect=OSError):
            assert _read_battery() is None
