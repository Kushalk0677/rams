"""
pytest shared fixtures for the RAMS test suite.

Fixtures provided:
    - patched_psutil: mocks psutil for deterministic ResourceMonitor tests
    - mock_snapshot: builds a ResourceSnapshot with known values
    - sample_pressures: common pressure values used across policy tests
    - policy_kwargs: sensible constructor defaults for each policy
    - isolated_config: mutable copy of default.yaml for config tests
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RAMS_ROOT", str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Autouse: keep the config cache clean between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """
    Clear the ``load_config`` LRU cache before and after each test
    to prevent cross-test pollution from the module-level cache.
    """
    from rams.config import reload_config
    reload_config()
    yield
    reload_config()


# ---------------------------------------------------------------------------
# Fixtures: config
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_config(tmp_path: Path) -> Path:
    """
    Create an isolated config file that can be safely mutated during a test.

    Returns the path to a temporary default.yaml with standard content.
    Tests that modify config should use this fixture to avoid side effects.
    """
    src = PROJECT_ROOT / "configs" / "default.yaml"
    dst = tmp_path / "default.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


@pytest.fixture
def isolated_config(tmp_path: Path) -> Path:
    """
    Create an isolated config file that can be safely mutated during a test.

    Returns the path to a temporary default.yaml with standard content.
    Tests that modify config should use this fixture to avoid side effects.
    """
    src = PROJECT_ROOT / "configs" / "default.yaml"
    dst = tmp_path / "default.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


# ---------------------------------------------------------------------------
# Fixtures: psutil mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_psutil() -> dict[str, MagicMock]:
    """
    Mock all psutil functions used by ResourceMonitor.

    Returns a dict of mocks so callers can configure return values:
        mock_cpu, mock_mem, mock_temp, mock_battery

    Default values simulate an idle system:
        CPU ~10%, MEM ~40%, no temp sensor, no battery.
    """
    with patch("rams.monitor.psutil.cpu_percent", return_value=10.0) as mcpu:
        mock_mem = MagicMock()
        mock_mem.percent = 40.0
        with patch("rams.monitor.psutil.virtual_memory", return_value=mock_mem) as mmem:
            with patch("rams.monitor._read_cpu_temp", return_value=None) as mtemp:
                with patch("rams.monitor._read_battery", return_value=None) as mbat:
                    yield {
                        "cpu_percent": mcpu,
                        "virtual_memory": mmem,
                        "cpu_temp": mtemp,
                        "battery": mbat,
                    }


@pytest.fixture
def patched_psutil_loaded() -> dict[str, MagicMock]:
    """
    Like *patched_psutil* but with high CPU/MEM to simulate system load.
        CPU ~85%, MEM ~75%, temp 82°C, battery 15%
    """
    with patch("rams.monitor.psutil.cpu_percent", return_value=85.0) as mcpu:
        mock_mem = MagicMock()
        mock_mem.percent = 75.0
        with patch("rams.monitor.psutil.virtual_memory", return_value=mock_mem) as mmem:
            with patch("rams.monitor._read_cpu_temp", return_value=82.0) as mtemp:
                with patch("rams.monitor._read_battery", return_value=15.0) as mbat:
                    yield {
                        "cpu_percent": mcpu,
                        "virtual_memory": mmem,
                        "cpu_temp": mtemp,
                        "battery": mbat,
                    }


# ---------------------------------------------------------------------------
# Fixtures: resource snapshots
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_snapshot():
    """Return a factory that builds ResourceSnapshots with sensible defaults."""
    from rams.monitor import ResourceSnapshot

    def _make(
        cpu: float = 10.0,
        mem: float = 40.0,
        temp: float | None = None,
        battery: float | None = None,
        ts: float = 1_000_000.0,
    ) -> ResourceSnapshot:
        # Import the real compute_pressure so snapshots have consistent pressure
        from rams.monitor import compute_pressure
        pressure = compute_pressure(cpu, mem, temp, battery)
        return ResourceSnapshot(
            timestamp=ts,
            cpu_percent=cpu,
            memory_percent=mem,
            cpu_temp=temp,
            battery_percent=battery,
            pressure_index=pressure,
        )

    return _make


# ---------------------------------------------------------------------------
# Fixtures: sample pressure levels
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pressures() -> dict[str, float]:
    """
    Canonical pressure values for policy tests.
    These correspond to the default thresholds (lo=0.45, hi=0.72).
    """
    return {
        "idle":      0.08,   # well below lo → MEDIUM
        "light":     0.28,   # below lo → MEDIUM
        "moderate":  0.52,   # between lo and hi → SMALL
        "heavy":     0.72,   # at hi threshold → boundary
        "burst":     0.93,   # above hi → NANO
    }


# ---------------------------------------------------------------------------
# Fixtures: policy fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy_kwargs() -> dict[str, dict[str, Any]]:
    """
    Default constructor arguments for each policy type.

    These match the defaults in configs/default.yaml but are explicitly
    provided so tests are not coupled to the config file.
    """
    return {
        "threshold": {
            "lo_thresh": 0.45,
            "hi_thresh": 0.72,
            "hysteresis_window": 3,
        },
        "predictive": {
            "history_len": 20,
            "alpha": 0.35,
            "lo_thresh": 0.40,
            "hi_thresh": 0.68,
            "use_lstm": False,
        },
        "safety": {
            "lo_thresh": 0.45,
            "hi_thresh": 0.72,
            "hysteresis_window": 3,
            "proximity_window_s": 0.5,
            "min_conf": 0.40,
        },
        "adaptive": {
            "alpha_min": 0.10,
            "alpha_max": 0.70,
            "var_window": 15,
            "lo_thresh": 0.40,
            "hi_thresh": 0.68,
        },
        "safety2": {
            "lo_thresh": 0.45,
            "hi_thresh": 0.72,
            "hysteresis_window": 3,
            "proximity_window_s": 0.5,
            "min_conf": 0.40,
            "near_area_thresh": 8_000.0,
        },
    }


@pytest.fixture
def make_threshold_policy():
    """Factory fixture for ThresholdPolicy."""
    from rams.policy import ThresholdPolicy

    def _make(**overrides):
        kwargs = {
            "lo_thresh": 0.45,
            "hi_thresh": 0.72,
            "hysteresis_window": 3,
        }
        kwargs.update(overrides)
        return ThresholdPolicy(**kwargs)

    return _make


@pytest.fixture
def make_safety_policy():
    """Factory fixture for SafetyPolicy."""
    from rams.policy import SafetyPolicy

    def _make(**overrides):
        kwargs = {
            "lo_thresh": 0.45,
            "hi_thresh": 0.72,
            "hysteresis_window": 3,
            "proximity_window_s": 0.5,
            "min_conf": 0.25,
        }
        kwargs.update(overrides)
        return SafetyPolicy(**kwargs)

    return _make


@pytest.fixture
def make_safety2_policy():
    """Factory fixture for SafetyTwoLevelPolicy."""
    from rams.policy import SafetyTwoLevelPolicy

    def _make(**overrides):
        kwargs = {
            "lo_thresh": 0.45,
            "hi_thresh": 0.72,
            "hysteresis_window": 3,
            "proximity_window_s": 0.5,
            "min_conf": 0.25,
            "near_area_thresh": 8_000.0,
        }
        kwargs.update(overrides)
        return SafetyTwoLevelPolicy(**kwargs)

    return _make


@pytest.fixture
def make_predictive_policy():
    """Factory fixture for PredictivePolicy (EWMA mode)."""
    from rams.policy import PredictivePolicy

    def _make(**overrides):
        kwargs = {
            "history_len": 20,
            "alpha": 0.35,
            "lo_thresh": 0.40,
            "hi_thresh": 0.68,
            "use_lstm": False,
        }
        kwargs.update(overrides)
        return PredictivePolicy(**kwargs)

    return _make


@pytest.fixture
def make_adaptive_policy():
    """Factory fixture for AdaptivePredictivePolicy."""
    from rams.policy import AdaptivePredictivePolicy

    def _make(**overrides):
        kwargs = {
            "alpha_min": 0.10,
            "alpha_max": 0.70,
            "var_window": 15,
            "lo_thresh": 0.40,
            "hi_thresh": 0.68,
        }
        kwargs.update(overrides)
        return AdaptivePredictivePolicy(**kwargs)

    return _make
