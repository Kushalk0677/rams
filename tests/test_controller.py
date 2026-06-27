"""
Tests for rams.controller — RAMSController orchestrator.

All tests use simulation mode so no real model files are needed.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from rams.models import Tier
from rams.policy import ThresholdPolicy


# ===================================================================
# Constructor / __init__
# ===================================================================


class TestConstructor:
    """``RAMSController.__init__`` parameter resolution."""

    def test_default_constructor(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController()
        assert ctrl.simulate is False  # config default
        assert ctrl.policy.name == "safety"  # config default
        assert ctrl.monitor.hz == 10.0  # config default

    def test_simulate_true(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True)
        assert ctrl.simulate is True
        assert ctrl.library.simulate is True
        for wrapper in ctrl.library._models.values():
            assert wrapper.simulate is True

    def test_policy_string_creates_correct_policy(self) -> None:
        from rams.controller import RAMSController
        from rams.policy import ThresholdPolicy

        ctrl = RAMSController(simulate=True, policy="threshold")
        assert isinstance(ctrl.policy, ThresholdPolicy)
        assert ctrl.policy.name == "threshold"

    def test_policy_instance_used_directly(self) -> None:
        from rams.controller import RAMSController

        custom = ThresholdPolicy(lo_thresh=0.30, hi_thresh=0.60,
                                 hysteresis_window=1)
        ctrl = RAMSController(simulate=True, policy=custom)
        assert ctrl.policy is custom
        assert ctrl.policy.lo == 0.30

    def test_policy_kwargs_passed_to_factory(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"lo_thresh": 0.20,
                                             "hi_thresh": 0.50,
                                             "hysteresis_window": 1})
        assert ctrl.policy.lo == 0.20
        assert ctrl.policy.hi == 0.50
        assert ctrl.policy.window == 1

    def test_custom_monitor_hz(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, monitor_hz=25.0)
        assert ctrl.monitor.hz == 25.0
        assert ctrl.monitor._interval == pytest.approx(0.04)

    def test_initial_current_tier(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True)
        assert ctrl._current_tier == Tier.SMALL

    def test_initial_switch_log_empty(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True)
        assert ctrl.switch_log == []

    def test_invalid_policy_name_raises(self) -> None:
        from rams.controller import RAMSController

        with pytest.raises(ValueError, match="Unknown policy"):
            RAMSController(simulate=True, policy="nonexistent")

    def test_simulate_none_resolves_to_config_default(self) -> None:
        from rams.controller import RAMSController

        # The real config has simulate: false
        ctrl = RAMSController(simulate=None)
        assert ctrl.simulate is False


# ===================================================================
# Lifecycle: start / stop / context manager
# ===================================================================


class TestLifecycle:
    """``start()``, ``stop()``, and context manager."""

    def test_start_loads_models_and_starts_monitor(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        ctrl.start()
        try:
            assert ctrl.monitor._running is True
            assert ctrl.monitor._thread is not None
            for wrapper in ctrl.library._models.values():
                assert wrapper._loaded is True
        finally:
            ctrl.stop()

    def test_start_is_idempotent(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        ctrl.start()
        ctrl.start()  # second call should not raise
        ctrl.stop()

    def test_stop_stops_monitor_and_unloads_models(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        ctrl.start()
        ctrl.stop()
        assert ctrl.monitor._running is False
        for wrapper in ctrl.library._models.values():
            assert wrapper._loaded is False
            assert wrapper._model is None

    def test_context_manager(self) -> None:
        from rams.controller import RAMSController

        with RAMSController(simulate=True, policy="threshold") as ctrl:
            assert ctrl.monitor._running is True
            for wrapper in ctrl.library._models.values():
                assert wrapper._loaded is True
        # After exit
        assert ctrl.monitor._running is False
        for wrapper in ctrl.library._models.values():
            assert wrapper._loaded is False

    def test_context_manager_exception_still_stops(self) -> None:
        from rams.controller import RAMSController

        with pytest.raises(RuntimeError, match="test error"):
            with RAMSController(simulate=True, policy="threshold") as ctrl:
                raise RuntimeError("test error")
        # Should still be stopped
        assert ctrl.monitor._running is False


# ===================================================================
# infer() — result structure and basic behaviour
# ===================================================================


EXPECTED_INFER_KEYS = {
    "tier", "latency_ms", "pressure", "cpu_pct", "mem_pct",
    "cpu_temp", "battery",
    "detections", "backend", "simulated", "accuracy_proxy",
}
"""Keys that should appear in every infer() result when a snapshot is available."""

EXPECTED_INFER_KEYS_NO_SNAP = {
    "tier", "latency_ms", "pressure",
    "detections", "backend", "simulated", "accuracy_proxy",
}
"""Keys available when the monitor hasn't produced a snapshot yet."""


class TestInferBasic:
    """``infer()`` result shape and basic correctness."""

    def test_infer_returns_dict_with_expected_keys(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        with ctrl:
            time.sleep(0.3)  # wait for monitor snapshot
            result = ctrl.infer()
        assert isinstance(result, dict)
        # cpu_pct, mem_pct, cpu_temp, battery come from monitor snapshot
        for key in EXPECTED_INFER_KEYS:
            assert key in result, f"Missing expected key: {key}"

    def test_infer_without_start_still_works(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        # No start — monitor not running, snapshot is None
        # ModelWrapper.infer() auto-loads so this should still work
        result = ctrl.infer()
        assert isinstance(result, dict)
        for key in EXPECTED_INFER_KEYS_NO_SNAP:
            assert key in result, f"Missing key: {key}"
        # pressure should be 0.0 (fallback when no snapshot and no override)
        assert result["pressure"] == 0.0
        # cpu_pct / mem_pct / cpu_temp / battery should be absent
        for key in ("cpu_pct", "mem_pct", "cpu_temp", "battery"):
            assert key not in result, f"Unexpected key without snapshot: {key}"

    def test_infer_simulated_flag_true(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        result = ctrl.infer()
        assert result["simulated"] is True
        assert result["backend"] == "simulation"

    def test_infer_latency_positive(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        result = ctrl.infer()
        assert result["latency_ms"] > 0

    def test_infer_detections_is_list(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        result = ctrl.infer()
        assert isinstance(result["detections"], list)

    def test_infer_accuracy_proxy_in_range(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        for _ in range(5):
            result = ctrl.infer()
            assert 0.0 <= result["accuracy_proxy"] <= 1.0

    def test_infer_with_frame_none(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        result = ctrl.infer(frame=None)
        assert result["simulated"] is True

    def test_multiple_infer_calls_work(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        for i in range(5):
            result = ctrl.infer()
            assert result["simulated"] is True
            assert result["latency_ms"] > 0

    def test_infer_after_stop_reloads(self) -> None:
        """``infer()`` after ``stop()`` auto-loads the current tier's wrapper."""
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        ctrl.start()
        ctrl.stop()
        # All models are now unloaded
        for wrapper in ctrl.library._models.values():
            assert wrapper._loaded is False

        # infer() should auto-load the current tier's wrapper
        result = ctrl.infer()
        assert result["simulated"] is True
        # Only the wrapper for the current tier should have been re-loaded
        current_tier = ctrl._current_tier
        for tier, wrapper in ctrl.library._models.items():
            if tier == current_tier:
                assert wrapper._loaded is True, (
                    f"Wrapper for {tier} should be loaded"
                )
            else:
                assert wrapper._loaded is False, (
                    f"Wrapper for {tier} should still be unloaded"
                )


# ===================================================================
# Pressure override
# ===================================================================


class TestPressureOverride:
    """``set_pressure_override()`` injects synthetic pressure."""

    def test_high_pressure_forces_nano(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.90)
        result = ctrl.infer()
        assert result["tier"] == "NANO"
        assert result["pressure"] == 0.90

    def test_low_pressure_keeps_medium(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.10)
        result = ctrl.infer()
        assert result["tier"] == "MEDIUM"

    def test_moderate_pressure_produces_small(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.52)
        result = ctrl.infer()
        assert result["tier"] == "SMALL"

    def test_clear_override_returns_to_zero_pressure(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.90)
        ctrl.infer()  # establishes NANO

        ctrl.set_pressure_override(None)
        result = ctrl.infer()
        # With no override and no snapshot, pressure falls back to 0.0
        assert result["pressure"] == 0.0
        assert result["tier"] == "MEDIUM"  # 0.0 < lo_thresh

    def test_pressure_in_result_matches_override(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        for val in (0.0, 0.25, 0.50, 0.75, 1.0):
            ctrl.set_pressure_override(val)
            result = ctrl.infer()
            assert result["pressure"] == val


# ===================================================================
# Tier selection via pressure override
# ===================================================================


class TestTierSelection:
    """Verify tier changes correctly under pressure override."""

    def test_override_with_hysteresis_window_3(self) -> None:
        """With hysteresis_window=3, needs 3 calls to switch tiers."""
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 3})
        ctrl.set_pressure_override(0.90)  # would be NANO

        # First call: starts building candidate, stays SMALL
        result = ctrl.infer()
        assert result["tier"] == "SMALL"

        # Second call
        result = ctrl.infer()
        assert result["tier"] == "SMALL"

        # Third call triggers switch
        result = ctrl.infer()
        assert result["tier"] == "NANO"

    def test_switch_from_medium_to_small_to_nano(self) -> None:
        """Progressive tier switching via multiple override values."""
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})

        # Start with idle pressure → MEDIUM
        ctrl.set_pressure_override(0.08)
        result = ctrl.infer()
        assert result["tier"] == "MEDIUM"

        # Moderate pressure → SMALL
        ctrl.set_pressure_override(0.52)
        result = ctrl.infer()
        assert result["tier"] == "SMALL"

        # High pressure → NANO
        ctrl.set_pressure_override(0.93)
        result = ctrl.infer()
        assert result["tier"] == "NANO"

    def test_current_tier_property(self) -> None:
        """``current_tier`` property reflects latest selection."""
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        assert ctrl.current_tier == Tier.SMALL

        ctrl.set_pressure_override(0.08)
        ctrl.infer()
        assert ctrl.current_tier == Tier.MEDIUM

        ctrl.set_pressure_override(0.93)
        ctrl.infer()
        assert ctrl.current_tier == Tier.NANO


# ===================================================================
# status() diagnostics
# ===================================================================


class TestStatus:
    """``status()`` method."""

    STATUS_KEYS = {"policy", "current_tier", "pressure", "cpu_pct",
                   "mem_pct", "cpu_temp", "battery", "n_switches"}

    def test_status_returns_dict(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        s = ctrl.status()
        assert isinstance(s, dict)

    def test_status_has_expected_keys(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        s = ctrl.status()
        assert set(s.keys()) == self.STATUS_KEYS

    def test_status_before_start(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        s = ctrl.status()
        assert s["policy"] == "threshold"
        assert s["current_tier"] == "SMALL"
        assert s["n_switches"] == 0
        # No snapshot available → pressure/cpu/mem/etc are None
        assert s["pressure"] is None
        assert s["cpu_pct"] is None
        assert s["mem_pct"] is None
        assert s["cpu_temp"] is None
        assert s["battery"] is None

    def test_status_after_infer(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.52)
        ctrl.infer()
        s = ctrl.status()
        assert s["policy"] == "threshold"
        assert s["current_tier"] == "SMALL"
        # n_switches depends on how many switches happened:
        # started at SMALL, stayed SMALL with 0.52 (no switch needed)
        assert s["n_switches"] == 0

    def test_status_after_switch(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.08)  # MEDIUM
        ctrl.infer()
        s = ctrl.status()
        assert s["current_tier"] == "MEDIUM"
        assert s["n_switches"] == 1

    def test_status_policy_reflects_constructor(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="safety")
        s = ctrl.status()
        assert s["policy"] == "safety"


# ===================================================================
# switch_log
# ===================================================================


class TestSwitchLog:
    """``switch_log`` property."""

    def test_empty_initially(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold")
        assert ctrl.switch_log == []

    def test_records_single_switch(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.08)
        ctrl.infer()  # SMALL → MEDIUM
        log = ctrl.switch_log
        assert len(log) == 1
        entry = log[0]
        assert "ts" in entry
        assert "from" in entry
        assert "to" in entry
        assert "pressure" in entry
        assert entry["from"] == "SMALL"
        assert entry["to"] == "MEDIUM"
        assert entry["pressure"] == 0.08

    def test_records_multiple_switches(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.08)
        ctrl.infer()  # SMALL → MEDIUM
        ctrl.set_pressure_override(0.93)
        ctrl.infer()  # MEDIUM → NANO
        log = ctrl.switch_log
        assert len(log) == 2
        assert log[1]["from"] == "MEDIUM"
        assert log[1]["to"] == "NANO"

    def test_switch_log_returns_copy(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.08)
        ctrl.infer()
        log1 = ctrl.switch_log
        log2 = ctrl.switch_log
        assert log1 is not log2  # different list objects
        assert log1 == log2  # same content

    def test_modifying_returned_list_does_not_affect_internal(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.08)
        ctrl.infer()
        log = ctrl.switch_log
        log.clear()
        # Internal log should be unchanged
        assert len(ctrl.switch_log) == 1


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_pressure_override_attribute_set_dynamically(self) -> None:
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True)
        assert not hasattr(ctrl, "_pressure_override")
        ctrl.set_pressure_override(0.50)
        assert ctrl._pressure_override == 0.50
        ctrl.set_pressure_override(None)
        assert ctrl._pressure_override is None

    def test_preserves_switch_log_after_stop(self) -> None:
        """Switch log persists across stop/start cycles."""
        from rams.controller import RAMSController

        ctrl = RAMSController(simulate=True, policy="threshold",
                              policy_kwargs={"hysteresis_window": 1})
        ctrl.set_pressure_override(0.08)
        ctrl.infer()  # SMALL → MEDIUM
        ctrl.start()
        ctrl.stop()
        # Switch log should still be intact
        assert len(ctrl.switch_log) == 1

    def test_different_policy_names(self) -> None:
        """All built-in policy names can be constructed."""
        from rams.controller import RAMSController

        for name in ("threshold", "predictive", "safety", "adaptive",
                     "safety2"):
            ctrl = RAMSController(simulate=True, policy=name)
            assert ctrl.policy.name == name

    def test_pre_constructed_policy_with_custom_thresholds(self) -> None:
        from rams.controller import RAMSController

        custom = ThresholdPolicy(lo_thresh=0.10, hi_thresh=0.30,
                                 hysteresis_window=1)
        ctrl = RAMSController(simulate=True, policy=custom)
        ctrl.set_pressure_override(0.20)
        result = ctrl.infer()
        # 0.20 >= 0.10 and 0.20 < 0.30 → SMALL
        assert result["tier"] == "SMALL"

    def test_policy_observe_called_with_detections(self) -> None:
        """Policy.observe is invoked with the infer result's detections."""
        from rams.controller import RAMSController
        from unittest.mock import MagicMock

        mock_policy = MagicMock(wraps=ThresholdPolicy(
            lo_thresh=0.45, hi_thresh=0.72, hysteresis_window=1))
        mock_policy.name = "mock"
        mock_policy.select_tier.return_value = Tier.NANO

        ctrl = RAMSController(simulate=True, policy=mock_policy)
        ctrl.infer()
        # observe should have been called with a list of detections
        assert mock_policy.observe.called
        args, _ = mock_policy.observe.call_args
        assert isinstance(args[0], list)
