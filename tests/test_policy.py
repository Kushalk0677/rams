"""
Tests for rams.policy — all 5 switching policies and the factory.

Fixtures used (from conftest)
-----------------------------
- make_threshold_policy
- make_predictive_policy
- make_safety_policy
- make_safety2_policy
- make_adaptive_policy
- sample_pressures
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import patch

import pytest

from rams.models import Tier


# ===================================================================
# Helpers
# ===================================================================


def _drive_hysteresis(policy: Any, pressure: float, target: Tier,
                      *, start: Tier = Tier.MEDIUM) -> list[Tier]:
    """Call ``select_tier`` repeatedly until it returns *target*.

    Returns the full sequence of returned tiers so callers can inspect
    intermediate values.
    """
    tiers: list[Tier] = []
    last = start
    for _ in range(20):  # safety limit
        last = policy.select_tier(pressure, last)
        tiers.append(last)
        if last == target:
            break
    return tiers


# ===================================================================
# BasePolicy
# ===================================================================


class TestBasePolicy:
    """``BasePolicy`` abstract interface."""

    def test_cannot_instantiate_directly(self) -> None:
        from rams.policy import BasePolicy

        bp = BasePolicy()
        with pytest.raises(NotImplementedError):
            bp.select_tier(0.5, Tier.MEDIUM)

    def test_observe_returns_none(self) -> None:
        from rams.policy import BasePolicy

        bp = BasePolicy()
        assert bp.observe() is None
        assert bp.observe([]) is None
        assert bp.observe([{"class": "person", "conf": 0.9}]) is None

    def test_reset_does_not_raise(self) -> None:
        from rams.policy import BasePolicy

        BasePolicy().reset()


# ===================================================================
# ThresholdPolicy
# ===================================================================


class TestThresholdPolicyBasic:
    """Basic tier selection — no hysteresis edge cases."""

    def test_idle_pressure_returns_medium(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)

        tier = policy.select_tier(0.08, Tier.MEDIUM)
        assert tier == Tier.MEDIUM

    def test_moderate_pressure_returns_small(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)

        tier = policy.select_tier(0.52, Tier.MEDIUM)
        assert tier == Tier.SMALL

    def test_burst_pressure_returns_nano(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)

        tier = policy.select_tier(0.93, Tier.MEDIUM)
        assert tier == Tier.NANO

    def test_boundary_at_lo_thresh(self, make_threshold_policy: Any) -> None:
        """Pressure exactly at lo_thresh → SMALL."""
        policy = make_threshold_policy(lo_thresh=0.45, hi_thresh=0.72,
                                       hysteresis_window=1)

        tier = policy.select_tier(0.45, Tier.MEDIUM)
        assert tier == Tier.SMALL

    def test_boundary_at_hi_thresh(self, make_threshold_policy: Any) -> None:
        """Pressure exactly at hi_thresh → NANO."""
        policy = make_threshold_policy(lo_thresh=0.45, hi_thresh=0.72,
                                       hysteresis_window=1)

        tier = policy.select_tier(0.72, Tier.MEDIUM)
        assert tier == Tier.NANO

    def test_just_below_lo_stays_medium(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)

        tier = policy.select_tier(0.44, Tier.MEDIUM)
        assert tier == Tier.MEDIUM

    def test_just_below_hi_is_small(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)

        tier = policy.select_tier(0.71, Tier.MEDIUM)
        assert tier == Tier.SMALL

    def test_using_sample_pressures(
        self, make_threshold_policy: Any, sample_pressures: dict
    ) -> None:
        policy = make_threshold_policy(hysteresis_window=1)

        assert policy.select_tier(sample_pressures["idle"], Tier.MEDIUM) == Tier.MEDIUM
        assert policy.select_tier(sample_pressures["moderate"], Tier.MEDIUM) == Tier.SMALL
        assert policy.select_tier(sample_pressures["burst"], Tier.MEDIUM) == Tier.NANO


class TestThresholdBoundaryPressures:
    """Edge cases around threshold values."""

    def test_pressure_exactly_zero(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)
        assert policy.select_tier(0.0, Tier.MEDIUM) == Tier.MEDIUM

    def test_pressure_exactly_one(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)
        assert policy.select_tier(1.0, Tier.MEDIUM) == Tier.NANO

    def test_negative_pressure(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=1)
        # Pressure below 0 should be treated as below lo_thresh
        assert policy.select_tier(-0.1, Tier.MEDIUM) == Tier.MEDIUM


class TestThresholdHysteresis:
    """Hysteresis window behaviour."""

    def test_requires_window_before_switch(self, make_threshold_policy: Any) -> None:
        """Must see ``window`` consecutive different-tier samples before switching."""
        policy = make_threshold_policy(hysteresis_window=3)
        last = Tier.MEDIUM

        # First call with moderate pressure
        t1 = policy.select_tier(0.52, last)
        assert t1 == Tier.MEDIUM  # still pending

        # Second consecutive call
        t2 = policy.select_tier(0.52, t1)
        assert t2 == Tier.MEDIUM  # still pending

        # Third call triggers the switch
        t3 = policy.select_tier(0.52, t2)
        assert t3 == Tier.SMALL  # committed

    def test_partial_window_then_back_to_idle_resets(
        self, make_threshold_policy: Any
    ) -> None:
        """If pressure drops back before the window fills, the counter resets."""
        policy = make_threshold_policy(hysteresis_window=3)
        last = Tier.MEDIUM

        # Two moderate readings (not enough to switch)
        last = policy.select_tier(0.52, last)
        assert last == Tier.MEDIUM
        last = policy.select_tier(0.52, last)
        assert last == Tier.MEDIUM

        # Pressure drops back to idle
        last = policy.select_tier(0.08, last)
        assert last == Tier.MEDIUM  # still MEDIUM (and candidate reset)

        # Now another moderate reading should start a fresh count
        last = policy.select_tier(0.52, last)
        assert last == Tier.MEDIUM  # count=1, need 3

    def test_hysteresis_across_tiers(self, make_threshold_policy: Any) -> None:
        """MEDIUM→SMALL completes its window, then SMALL→NANO needs its own window."""
        policy = make_threshold_policy(hysteresis_window=3)
        last = Tier.MEDIUM

        # ---- Drive MEDIUM → SMALL ----
        # 3 calls with moderate pressure (0.52) to trigger the switch
        for _ in range(3):
            last = policy.select_tier(0.52, last)
        assert last == Tier.SMALL

        # ---- Drive SMALL → NANO ----
        # First burst reading should not immediately switch
        t1 = policy.select_tier(0.93, last)
        assert t1 == Tier.SMALL  # count=1

        t2 = policy.select_tier(0.93, t1)
        assert t2 == Tier.SMALL  # count=2

        t3 = policy.select_tier(0.93, t2)
        assert t3 == Tier.NANO  # count=3 → committed

    def test_window_of_one_switches_immediately(
        self, make_threshold_policy: Any
    ) -> None:
        policy = make_threshold_policy(hysteresis_window=1)
        assert policy.select_tier(0.52, Tier.MEDIUM) == Tier.SMALL

    def test_window_of_five(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=5)
        last = Tier.MEDIUM
        for i in range(4):
            last = policy.select_tier(0.52, last)
            assert last == Tier.MEDIUM, f"failed on iteration {i}"
        last = policy.select_tier(0.52, last)
        assert last == Tier.SMALL


class TestThresholdCustom:
    """Custom thresholds and reset."""

    def test_custom_lo_hi(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(
            lo_thresh=0.20, hi_thresh=0.50, hysteresis_window=1
        )
        assert policy.select_tier(0.30, Tier.MEDIUM) == Tier.SMALL
        assert policy.select_tier(0.60, Tier.MEDIUM) == Tier.NANO

    def test_reset_clears_candidate(self, make_threshold_policy: Any) -> None:
        policy = make_threshold_policy(hysteresis_window=3)
        last = Tier.MEDIUM

        # Build partial hysteresis
        last = policy.select_tier(0.52, last)
        assert policy._candidate == Tier.SMALL
        assert policy._candidate_count == 1

        policy.reset()
        assert policy._candidate is None
        assert policy._candidate_count == 0

        # After reset, need full window again
        for _ in range(2):
            last = policy.select_tier(0.52, last)
            assert last == Tier.MEDIUM
        last = policy.select_tier(0.52, last)
        assert last == Tier.SMALL


# ===================================================================
# PredictivePolicy (EWMA mode)
# ===================================================================


class TestPredictiveBasic:
    """PredictivePolicy with EWMA (use_lstm=False)."""

    def test_first_call_returns_medium(self, make_predictive_policy: Any) -> None:
        """EWMA hasn't warmed up yet — first idle reading stays MEDIUM."""
        policy = make_predictive_policy(alpha=0.35, lo_thresh=0.40, hi_thresh=0.68)
        tier = policy.select_tier(0.08, Tier.MEDIUM)
        assert tier == Tier.MEDIUM

    def test_forecast_converges_to_nano(self, make_predictive_policy: Any) -> None:
        """After several high-pressure calls the forecast exceeds hi_thresh."""
        policy = make_predictive_policy(alpha=0.35, lo_thresh=0.40, hi_thresh=0.68)

        # Start with an idle reading
        last = policy.select_tier(0.08, Tier.MEDIUM)
        assert last == Tier.MEDIUM

        # Feed sustained high pressure — forecast should climb
        for i in range(10):
            last = policy.select_tier(0.80, last)

        # After enough calls ewma should exceed hi_thresh (0.68)
        assert last == Tier.NANO, f"Expected NANO after 10 high-pressure calls, got {last}"

    def test_forecast_smooths_spike(self, make_predictive_policy: Any) -> None:
        """A single high-pressure spike is smoothed by EWMA alpha."""
        policy = make_predictive_policy(alpha=0.35, lo_thresh=0.40, hi_thresh=0.68)

        last = policy.select_tier(0.08, Tier.MEDIUM)  # ewma = 0.08
        # Single spike
        last = policy.select_tier(0.80, last)  # ewma = 0.35*0.80 + 0.65*0.08 = 0.332
        # 0.332 < 0.40 → MEDIUM
        assert last == Tier.MEDIUM

    def test_sustained_idle_stays_medium(self, make_predictive_policy: Any) -> None:
        """After sustained idle pressure, forecast stays below lo_thresh."""
        policy = make_predictive_policy(alpha=0.35, lo_thresh=0.40, hi_thresh=0.68)
        last = Tier.MEDIUM
        for _ in range(10):
            last = policy.select_tier(0.05, last)
        assert last == Tier.MEDIUM

    def test_converges_to_small_with_moderate_pressure(
        self, make_predictive_policy: Any
    ) -> None:
        """Sustained moderate pressure lands in the SMALL band."""
        policy = make_predictive_policy(alpha=0.35, lo_thresh=0.40, hi_thresh=0.68)
        last = Tier.MEDIUM
        for _ in range(10):
            last = policy.select_tier(0.55, last)
        # ewma should converge toward 0.55 (between 0.40 and 0.68)
        assert last == Tier.SMALL, f"Expected SMALL, got {last}"

    def test_high_alpha_responds_faster(self, make_predictive_policy: Any) -> None:
        """Higher alpha → forecast reaches hi_thresh in fewer calls."""
        fast = make_predictive_policy(alpha=0.70, lo_thresh=0.40, hi_thresh=0.68)
        slow = make_predictive_policy(alpha=0.20, lo_thresh=0.40, hi_thresh=0.68)

        # Start with a low-pressure reading so EWMA initialises low
        lf = fast.select_tier(0.05, Tier.MEDIUM)
        ls = slow.select_tier(0.05, Tier.MEDIUM)

        # Count how many high-pressure calls each needs to reach NANO
        calls_fast = 0
        calls_slow = 0
        for _ in range(20):
            if lf != Tier.NANO:
                lf = fast.select_tier(0.80, lf)
                calls_fast += 1
            if ls != Tier.NANO:
                ls = slow.select_tier(0.80, ls)
                calls_slow += 1

        assert calls_fast < calls_slow, (
            f"Fast alpha should reach NANO faster ({calls_fast} vs {calls_slow})"
        )

    def test_reset_clears_state(self, make_predictive_policy: Any) -> None:
        policy = make_predictive_policy(alpha=0.35)
        # Build some EWMA state
        policy.select_tier(0.80, Tier.MEDIUM)
        policy.select_tier(0.80, Tier.MEDIUM)
        assert policy._ewma is not None
        assert len(policy._history) == 2

        policy.reset()
        assert policy._ewma is None
        assert len(policy._history) == 0


# ===================================================================
# SafetyPolicy
# ===================================================================


class TestSafetyBasic:
    """SafetyPolicy with no VRU — same as ThresholdPolicy."""

    def test_idle_pressure_no_vru(self, make_safety_policy: Any) -> None:
        policy = make_safety_policy(hysteresis_window=1)
        assert policy.select_tier(0.08, Tier.MEDIUM) == Tier.MEDIUM

    def test_moderate_pressure_no_vru(self, make_safety_policy: Any) -> None:
        policy = make_safety_policy(hysteresis_window=1)
        last = _drive_hysteresis(policy, 0.52, Tier.SMALL)
        assert last[-1] == Tier.SMALL

    def test_burst_pressure_no_vru(self, make_safety_policy: Any) -> None:
        policy = make_safety_policy(hysteresis_window=1)
        assert policy.select_tier(0.93, Tier.MEDIUM) == Tier.NANO


class TestSafetyVruOverride:
    """VRU detection triggers the safety override."""

    VRU_DET = [{"class": "person", "conf": 0.9}]

    def test_vru_override_to_small(self, make_safety_policy: Any) -> None:
        """When VRU detected and base tier is NANO, returns SMALL."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=self.VRU_DET)
        assert tier == Tier.SMALL

    def test_vru_not_override_when_already_medium(
        self, make_safety_policy: Any
    ) -> None:
        """If base tier is already MEDIUM, VRU doesn't force SMALL."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)

        tier = policy.select_tier(0.08, Tier.MEDIUM,
                                  recent_detections=self.VRU_DET)
        assert tier == Tier.MEDIUM

    def test_vru_timer_expiry(self, make_safety_policy: Any) -> None:
        """Override expires after ``proximity_window_s``."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=1.0)
        vru = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 10, 10]}]

        with patch("time.monotonic", side_effect=[100.0, 100.0, 101.5]):
            # At t=100: first call updates VRU timer
            tier1 = policy.select_tier(0.93, Tier.MEDIUM,
                                       recent_detections=vru)
            assert tier1 == Tier.SMALL  # VRU override active

            # At t=101.5: 1.5s > proximity_window_s=1.0, override expired
            tier2 = policy.select_tier(0.93, Tier.SMALL)
            assert tier2 == Tier.NANO  # Back to threshold-based selection

    def test_expiry_with_zero_window(self, make_safety_policy: Any) -> None:
        """With proximity_window_s=0 the override expires immediately."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=0.0)
        vru = [{"class": "person", "conf": 0.9}]

        with patch("time.monotonic", side_effect=[100.0, 100.0, 100.001]):
            # At t=100: VRU detected, window=0 so active at exact same time
            tier1 = policy.select_tier(0.93, Tier.MEDIUM,
                                       recent_detections=vru)
            assert tier1 == Tier.SMALL  # VRU override active

            # At t=100.001: 0.001ms > 0.0s, override expired
            tier2 = policy.select_tier(0.93, Tier.SMALL)
            assert tier2 == Tier.NANO  # VRU expired

    def test_min_conf_filter(self, make_safety_policy: Any) -> None:
        """Detections below ``min_conf`` do NOT trigger the override."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100,
                                    min_conf=0.50)
        low_conf = [{"class": "person", "conf": 0.30}]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=low_conf)
        assert tier == Tier.NANO  # No override

    def test_min_conf_boundary(self, make_safety_policy: Any) -> None:
        """Detection exactly at ``min_conf`` DOES trigger the override."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100,
                                    min_conf=0.50)
        at_boundary = [{"class": "person", "conf": 0.50}]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=at_boundary)
        assert tier == Tier.SMALL  # Override active

    def test_non_vru_classes_do_not_override(self, make_safety_policy: Any) -> None:
        """Car/truck detections do NOT trigger the override."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)
        non_vru = [{"class": "car", "conf": 0.9}]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=non_vru)
        assert tier == Tier.NANO  # No override

    def test_vulnerable_class_names(self, make_safety_policy: Any) -> None:
        """All names in VULNERABLE_CLASSES trigger the override."""
        from rams.policy import VULNERABLE_CLASSES

        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)

        for cls_name in VULNERABLE_CLASSES:
            det = [{"class": cls_name, "conf": 0.9}]
            tier = policy.select_tier(0.93, Tier.MEDIUM,
                                      recent_detections=det)
            assert tier == Tier.SMALL, f"Class '{cls_name}' should trigger override"


class TestSafetyObserve:
    """``observe()`` method integration."""

    VRU_DET = [{"class": "person", "conf": 0.9}]

    def test_observe_updates_vru_state(self, make_safety_policy: Any) -> None:
        """Calling ``observe()`` with VRU detections sets the timer."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)

        policy.observe(self.VRU_DET)
        assert policy._last_vru_time is not None

    def test_observe_empty_does_not_clear(self, make_safety_policy: Any) -> None:
        """Empty ``observe()`` call does NOT clear existing VRU state."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)

        # Freeze time.monotonic so the VRU never expires
        with patch("time.monotonic", return_value=1000.0):
            policy.observe(self.VRU_DET)
            assert policy._last_vru_time == 1000.0

            # Then observe with empty detections
            policy.observe([])
            assert policy._last_vru_time is not None

            # The VRU should still be active
            tier = policy.select_tier(0.93, Tier.SMALL)
            assert tier == Tier.SMALL

    def test_observe_none_does_not_clear(self, make_safety_policy: Any) -> None:
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)
        with patch("time.monotonic", return_value=1000.0):
            policy.observe(self.VRU_DET)
        policy.observe(None)
        assert policy._last_vru_time is not None

    def test_select_tier_with_detections_updates_vru(
        self, make_safety_policy: Any
    ) -> None:
        """Passing ``recent_detections`` to ``select_tier`` also updates VRU."""
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)

        policy.select_tier(0.93, Tier.MEDIUM, recent_detections=self.VRU_DET)
        assert policy._last_vru_time is not None


class TestSafetyReset:
    """``reset()`` behaviour."""

    def test_reset_clears_vru_state(self, make_safety_policy: Any) -> None:
        policy = make_safety_policy(hysteresis_window=1, proximity_window_s=100)
        vru = [{"class": "person", "conf": 0.9}]

        policy.observe(vru)
        assert policy._last_vru_time is not None

        policy.reset()
        assert policy._last_vru_time is None

    def test_reset_clears_base_candidate(self, make_safety_policy: Any) -> None:
        policy = make_safety_policy(hysteresis_window=3)
        policy.select_tier(0.52, Tier.MEDIUM)  # builds partial hysteresis

        policy.reset()
        assert policy._base._candidate is None
        assert policy._base._candidate_count == 0


# ===================================================================
# AdaptivePredictivePolicy
# ===================================================================


class TestAdaptiveBasic:
    """AdaptivePredictivePolicy basic forecast behaviour."""

    def test_first_call_returns_medium(self, make_adaptive_policy: Any) -> None:
        policy = make_adaptive_policy()
        assert policy.select_tier(0.08, Tier.MEDIUM) == Tier.MEDIUM

    def test_converges_to_nano(self, make_adaptive_policy: Any) -> None:
        """Sustained high pressure drives forecast to NANO."""
        policy = make_adaptive_policy(lo_thresh=0.40, hi_thresh=0.68)
        last = Tier.MEDIUM
        for _ in range(20):
            last = policy.select_tier(0.85, last)
        assert last == Tier.NANO

    def test_converges_to_small(self, make_adaptive_policy: Any) -> None:
        policy = make_adaptive_policy(lo_thresh=0.40, hi_thresh=0.68)
        last = Tier.MEDIUM
        for _ in range(20):
            last = policy.select_tier(0.55, last)
        assert last == Tier.SMALL

    def test_sustained_idle_stays_medium(self, make_adaptive_policy: Any) -> None:
        policy = make_adaptive_policy(lo_thresh=0.40, hi_thresh=0.68)
        last = Tier.MEDIUM
        for _ in range(20):
            last = policy.select_tier(0.05, last)
        assert last == Tier.MEDIUM


class TestAdaptiveAlpha:
    """Alpha adapts based on rolling variance."""

    def test_low_variance_low_alpha(self, make_adaptive_policy: Any) -> None:
        """Stable pressure → low EWMA alpha."""
        policy = make_adaptive_policy(alpha_min=0.10, alpha_max=0.70,
                                      var_window=15)
        # Feed very stable pressure
        for _ in range(10):
            policy.select_tier(0.30, Tier.MEDIUM)

        alpha = policy._adaptive_alpha()
        assert alpha == pytest.approx(0.10, abs=0.05), f"Expected low alpha, got {alpha}"

    def test_high_variance_high_alpha(self, make_adaptive_policy: Any) -> None:
        """Volatile pressure → higher EWMA alpha."""
        policy = make_adaptive_policy(alpha_min=0.10, alpha_max=0.70,
                                      var_window=15)
        # Feed volatile pressure
        for p in [0.05, 0.80, 0.05, 0.80, 0.05, 0.80, 0.05, 0.80]:
            policy.select_tier(p, Tier.MEDIUM)

        alpha = policy._adaptive_alpha()
        assert alpha > 0.50, f"Expected high alpha, got {alpha}"

    def test_alpha_ramps_with_variance(self, make_adaptive_policy: Any) -> None:
        """Alpha increases as variance increases."""
        policy = make_adaptive_policy(alpha_min=0.10, alpha_max=0.70,
                                      var_window=15)

        # Feed stable values first
        for _ in range(10):
            policy.select_tier(0.30, Tier.MEDIUM)
        alpha_stable = policy._adaptive_alpha()

        # Then volatile values
        for p in [0.05, 0.85, 0.05, 0.85, 0.05, 0.85]:
            policy.select_tier(p, Tier.MEDIUM)
        alpha_volatile = policy._adaptive_alpha()

        assert alpha_volatile > alpha_stable, (
            f"Volatile alpha ({alpha_volatile}) should exceed stable alpha "
            f"({alpha_stable})"
        )

    def test_reset_clears_state(self, make_adaptive_policy: Any) -> None:
        policy = make_adaptive_policy()
        policy.select_tier(0.80, Tier.MEDIUM)
        policy.select_tier(0.80, Tier.MEDIUM)
        assert policy._ewma is not None
        assert len(policy._history) == 2

        policy.reset()
        assert policy._ewma is None
        assert len(policy._history) == 0


# ===================================================================
# SafetyTwoLevelPolicy
# ===================================================================


class TestSafety2Basic:
    """SafetyTwoLevelPolicy with no VRU — same as ThresholdPolicy."""

    def test_no_vru_idle(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=1)
        assert policy.select_tier(0.08, Tier.MEDIUM) == Tier.MEDIUM

    def test_no_vru_burst(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=1)
        assert policy.select_tier(0.93, Tier.MEDIUM) == Tier.NANO

    def test_no_vru_moderate(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=1)
        last = _drive_hysteresis(policy, 0.52, Tier.SMALL)
        assert last[-1] == Tier.SMALL


class TestSafety2VruOverride:
    """Two-level VRU override based on bounding-box area."""

    def test_distant_vru_locks_small(self, make_safety2_policy: Any) -> None:
        """VRU with area < near_area_thresh → SMALL."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        vru_distant = [
            {"class": "person", "conf": 0.9, "xyxy": [0, 0, 50, 50]}
        ]  # area = 2500

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=vru_distant)
        assert tier == Tier.SMALL

    def test_near_vru_locks_medium(self, make_safety2_policy: Any) -> None:
        """VRU with area >= near_area_thresh → MEDIUM."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        vru_near = [
            {"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200]}
        ]  # area = 40000

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=vru_near)
        assert tier == Tier.MEDIUM

    def test_area_exactly_at_boundary(self, make_safety2_policy: Any) -> None:
        """Area exactly equal to near_area_thresh → MEDIUM."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        # area = sqrt(8000) ≈ 89.44 → use 89.44 × 89.44
        side = 8000**0.5
        vru_at = [
            {"class": "person", "conf": 0.9,
             "xyxy": [0, 0, side, side]}
        ]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=vru_at)
        assert tier == Tier.MEDIUM

    def test_area_just_below_boundary(self, make_safety2_policy: Any) -> None:
        """Area just below near_area_thresh → SMALL."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        vru_small = [
            {"class": "person", "conf": 0.9,
             "xyxy": [0, 0, 89, 89]}  # area = 7921 < 8000
        ]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=vru_small)
        assert tier == Tier.SMALL

    def test_vru_timer_expiry(self, make_safety2_policy: Any) -> None:
        """Override expires after proximity_window_s."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=1.0,
                                     near_area_thresh=8000)
        vru = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200]}]

        with patch("time.monotonic", side_effect=[100.0, 100.0, 101.5]):
            # At t=100: VRU detected, near lock → MEDIUM
            tier1 = policy.select_tier(0.93, Tier.MEDIUM,
                                       recent_detections=vru)
            assert tier1 == Tier.MEDIUM  # VRU active (near lock)

            # At t=101.5: 1.5s > proximity_window_s=1.0, override expired
            tier2 = policy.select_tier(0.93, Tier.SMALL)
            assert tier2 == Tier.NANO  # VRU expired

    def test_min_conf_filter(self, make_safety2_policy: Any) -> None:
        """Low-confidence VRU detections do NOT trigger override."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     min_conf=0.50, near_area_thresh=8000)
        low_conf = [{"class": "person", "conf": 0.30, "xyxy": [0, 0, 200, 200]}]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=low_conf)
        assert tier == Tier.NANO

    def test_min_conf_boundary_triggers(self, make_safety2_policy: Any) -> None:
        """Detection exactly at min_conf triggers the override."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     min_conf=0.50, near_area_thresh=8000)
        at_boundary = [{"class": "person", "conf": 0.50, "xyxy": [0, 0, 200, 200]}]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=at_boundary)
        assert tier == Tier.MEDIUM

    def test_non_vru_classes_do_not_override(self, make_safety2_policy: Any) -> None:
        """Non-VRU classes do not trigger override."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        non_vru = [{"class": "car", "conf": 0.9, "xyxy": [0, 0, 200, 200]}]

        tier = policy.select_tier(0.93, Tier.MEDIUM,
                                  recent_detections=non_vru)
        assert tier == Tier.NANO

    def test_near_vru_overrides_small_base_to_medium(
        self, make_safety2_policy: Any
    ) -> None:
        """Near VRU upgrades SMALL base tier to MEDIUM."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        # Use time freeze so VRU stays active
        with patch("time.monotonic", return_value=1000.0):
            vru_near = [
                {"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200]}
            ]
            # Base tier would be SMALL (pressure=0.52, lo=0.45, hi=0.72)
            tier = policy.select_tier(0.52, Tier.MEDIUM,
                                      recent_detections=vru_near)
            # base=SMALL(2), lock=MEDIUM(3), base<lock → MEDIUM
            assert tier == Tier.MEDIUM

    def test_distant_vru_overrides_nano_base_to_small(
        self, make_safety2_policy: Any
    ) -> None:
        """Distant VRU upgrades NANO base tier to SMALL."""
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100,
                                     near_area_thresh=8000)
        with patch("time.monotonic", return_value=2000.0):
            vru_distant = [
                {"class": "person", "conf": 0.9, "xyxy": [0, 0, 50, 50]}
            ]
            # Base tier would be NANO (pressure=0.93, lo=0.45, hi=0.72)
            tier = policy.select_tier(0.93, Tier.MEDIUM,
                                      recent_detections=vru_distant)
            # base=NANO(1), lock=SMALL(2), base<lock → SMALL
            assert tier == Tier.SMALL


class TestSafety2Observe:
    """``observe()`` with SafetyTwoLevelPolicy."""

    def test_observe_updates_vru_state(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100)
        vru = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200]}]

        policy.observe(vru)
        assert policy._last_vru_time is not None
        assert policy._last_vru_area == pytest.approx(40000.0)

    def test_observe_empty_does_not_clear(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100)
        vru = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200]}]

        policy.observe(vru)
        policy.observe([])
        assert policy._last_vru_time is not None


class TestSafety2Reset:
    """``reset()`` for SafetyTwoLevelPolicy."""

    def test_reset_clears_vru_state(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=1, proximity_window_s=100)
        vru = [{"class": "person", "conf": 0.9, "xyxy": [0, 0, 200, 200]}]

        policy.observe(vru)
        assert policy._last_vru_time is not None
        assert policy._last_vru_area > 0

        policy.reset()
        assert policy._last_vru_time is None
        assert policy._last_vru_area == 0.0

    def test_reset_clears_base_candidate(self, make_safety2_policy: Any) -> None:
        policy = make_safety2_policy(hysteresis_window=3)
        policy.select_tier(0.52, Tier.MEDIUM)

        policy.reset()
        assert policy._base._candidate is None
        assert policy._base._candidate_count == 0


# ===================================================================
# Factory: make_policy
# ===================================================================


class TestMakePolicy:
    """``make_policy()`` factory function."""

    def test_threshold(self) -> None:
        from rams.policy import ThresholdPolicy, make_policy

        p = make_policy("threshold")
        assert isinstance(p, ThresholdPolicy)

    def test_predictive(self) -> None:
        from rams.policy import PredictivePolicy, make_policy

        p = make_policy("predictive")
        assert isinstance(p, PredictivePolicy)

    def test_safety(self) -> None:
        from rams.policy import SafetyPolicy, make_policy

        p = make_policy("safety")
        assert isinstance(p, SafetyPolicy)

    def test_adaptive(self) -> None:
        from rams.policy import AdaptivePredictivePolicy, make_policy

        p = make_policy("adaptive")
        assert isinstance(p, AdaptivePredictivePolicy)

    def test_safety2(self) -> None:
        from rams.policy import SafetyTwoLevelPolicy, make_policy

        p = make_policy("safety2")
        assert isinstance(p, SafetyTwoLevelPolicy)

    def test_unknown_policy_raises_value_error(self) -> None:
        from rams.policy import make_policy

        with pytest.raises(ValueError) as exc:
            make_policy("nonexistent")
        assert "Unknown policy" in str(exc.value)
        assert "nonexistent" in str(exc.value)

    def test_empty_string_raises(self) -> None:
        from rams.policy import make_policy

        with pytest.raises(ValueError):
            make_policy("")

    def test_kwargs_passed_to_constructor(self) -> None:
        from rams.policy import ThresholdPolicy, make_policy

        p = make_policy("threshold", lo_thresh=0.30, hi_thresh=0.60,
                        hysteresis_window=5)
        assert isinstance(p, ThresholdPolicy)
        assert p.lo == 0.30
        assert p.hi == 0.60
        assert p.window == 5

    def test_kwargs_for_predictive(self) -> None:
        from rams.policy import PredictivePolicy, make_policy

        p = make_policy("predictive", alpha=0.50, lo_thresh=0.30)
        assert isinstance(p, PredictivePolicy)
        assert p.alpha == 0.50
        assert p.lo == 0.30

    def test_kwargs_for_safety(self) -> None:
        from rams.policy import SafetyPolicy, make_policy

        p = make_policy("safety", min_conf=0.50, proximity_window_s=2.0)
        assert isinstance(p, SafetyPolicy)
        assert p.min_conf == 0.50
        assert p.proximity_window_s == 2.0
