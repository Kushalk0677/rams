"""
Tests for rams.models — Tier enum, TierProfile, PROFILES, COCO_NAMES,
ModelWrapper (simulation mode), ModelLibrary (simulation mode),
and _parse_onnx_output.

Fast tests only — no real model files are loaded.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from enum import IntEnum
from typing import Any

import pytest

from rams.models import Tier


# ===================================================================
# Tier IntEnum
# ===================================================================


class TestTier:
    """``Tier`` IntEnum member values and properties."""

    def test_nano_value(self) -> None:
        assert Tier.NANO == 1

    def test_small_value(self) -> None:
        assert Tier.SMALL == 2

    def test_medium_value(self) -> None:
        assert Tier.MEDIUM == 3

    def test_is_int_enum(self) -> None:
        assert isinstance(Tier.NANO, int)
        assert isinstance(Tier.NANO, IntEnum)

    def test_iteration_yields_all_three(self) -> None:
        members = list(Tier)
        assert members == [Tier.NANO, Tier.SMALL, Tier.MEDIUM]

    def test_names(self) -> None:
        assert Tier.NANO.name == "NANO"
        assert Tier.SMALL.name == "SMALL"
        assert Tier.MEDIUM.name == "MEDIUM"

    def test_comparison_ordering(self) -> None:
        assert Tier.NANO < Tier.SMALL < Tier.MEDIUM


# ===================================================================
# TierProfile dataclass & PROFILES dict
# ===================================================================


class TestTierProfile:
    """``TierProfile`` dataclass fields and ``PROFILES`` dict."""

    def test_profiles_has_all_tiers(self) -> None:
        from rams.models import PROFILES

        assert set(PROFILES) == {Tier.NANO, Tier.SMALL, Tier.MEDIUM}

    def test_imgsz_values(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].imgsz == 320
        assert PROFILES[Tier.SMALL].imgsz == 416
        assert PROFILES[Tier.MEDIUM].imgsz == 640

    def test_non_zero_latency_stats(self) -> None:
        from rams.models import PROFILES

        for tier in Tier:
            p = PROFILES[tier]
            assert p.latency_mean_ms > 0, f"{tier} latency_mean_ms is zero"
            assert p.latency_std_ms > 0, f"{tier} latency_std_ms is zero"

    def test_non_zero_map50(self) -> None:
        from rams.models import PROFILES

        for tier in Tier:
            assert PROFILES[tier].map50 > 0

    def test_map50_increases_with_tier(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].map50 < PROFILES[Tier.SMALL].map50
        assert PROFILES[Tier.SMALL].map50 < PROFILES[Tier.MEDIUM].map50

    def test_latency_increases_with_tier(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].latency_mean_ms < PROFILES[Tier.SMALL].latency_mean_ms
        assert PROFILES[Tier.SMALL].latency_mean_ms < PROFILES[Tier.MEDIUM].latency_mean_ms

    def test_std_increases_with_tier(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].latency_std_ms < PROFILES[Tier.SMALL].latency_std_ms
        assert PROFILES[Tier.SMALL].latency_std_ms < PROFILES[Tier.MEDIUM].latency_std_ms

    def test_frozen_dataclass_cannot_modify(self) -> None:
        from rams.models import PROFILES

        with pytest.raises(FrozenInstanceError):
            PROFILES[Tier.NANO].imgsz = 999

    def test_model_ids(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].model_id == "yolov8n.pt"
        assert PROFILES[Tier.SMALL].model_id == "yolov8s.pt"
        assert PROFILES[Tier.MEDIUM].model_id == "yolov8m.pt"

    def test_onnx_ids(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].onnx_id == "yolov8n.onnx"
        assert PROFILES[Tier.SMALL].onnx_id == "yolov8s.onnx"
        assert PROFILES[Tier.MEDIUM].onnx_id == "yolov8m.onnx"

    def test_labels(self) -> None:
        from rams.models import PROFILES

        assert PROFILES[Tier.NANO].label == "YOLOv8-nano"
        assert PROFILES[Tier.SMALL].label == "YOLOv8-small"
        assert PROFILES[Tier.MEDIUM].label == "YOLOv8-medium"


# ===================================================================
# ModelWrapper — simulation mode
# ===================================================================


class TestModelWrapperInit:
    """``ModelWrapper.__init__`` with ``simulate=True``."""

    def test_sets_profile(self) -> None:
        from rams.models import ModelWrapper

        for tier in Tier:
            mw = ModelWrapper(tier, simulate=True)
            assert mw.profile is not None
            assert mw.profile.tier == tier

    def test_sets_backend_to_simulation(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        assert mw._backend == "simulation"
        assert mw.simulate is True

    def test_sets_imgsz_from_profile(self) -> None:
        from rams.models import PROFILES, ModelWrapper

        for tier in Tier:
            mw = ModelWrapper(tier, simulate=True)
            assert mw.imgsz == PROFILES[tier].imgsz

    def test_not_loaded_on_init(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        assert mw._loaded is False
        assert mw._model is None

    def test_simulate_false_also_starts_not_loaded(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=False)
        assert mw._loaded is False


class TestModelWrapperLoad:
    """``ModelWrapper.load()`` in simulation mode."""

    def test_load_sets_loaded_flag(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        assert mw._loaded is True

    def test_load_sets_backend(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        assert mw._backend == "simulation"

    def test_load_idempotent(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        mw.load()  # second call should not raise
        assert mw._loaded is True

    def test_load_for_all_tiers(self) -> None:
        from rams.models import ModelWrapper

        for tier in Tier:
            mw = ModelWrapper(tier, simulate=True)
            mw.load()
            assert mw._loaded is True
            assert mw._backend == "simulation"


class TestModelWrapperInfer:
    """``ModelWrapper.infer()`` in simulation mode."""

    RESULT_KEYS = {"tier", "simulated", "backend", "latency_ms",
                   "detections", "accuracy_proxy"}

    @pytest.fixture(autouse=True)
    def _loaded_wrapper(self) -> Any:
        """Return a pre-loaded NANO ModelWrapper for convenience."""
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        return mw

    def test_returns_dict(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert set(result.keys()) == self.RESULT_KEYS, (
            f"Missing keys: {self.RESULT_KEYS - set(result)}"
        )

    def test_tier_matches_wrapper(self) -> None:
        from rams.models import ModelWrapper

        for tier in Tier:
            mw = ModelWrapper(tier, simulate=True)
            mw.load()
            result = mw.infer()
            assert result["tier"] == tier.name

    def test_simulated_flag_true(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert result["simulated"] is True

    def test_backend_is_simulation(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert result["backend"] == "simulation"

    def test_latency_ms_positive(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert result["latency_ms"] > 0

    def test_detections_is_list(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert isinstance(result["detections"], list)

    def test_detection_items_have_expected_keys(
        self, _loaded_wrapper: Any
    ) -> None:
        result = _loaded_wrapper.infer()
        for det in result["detections"]:
            assert "class" in det
            assert "conf" in det
            assert "xyxy" in det
            assert isinstance(det["class"], str)
            assert isinstance(det["conf"], float)
            assert len(det["xyxy"]) == 4

    def test_accuracy_proxy_in_range(self, _loaded_wrapper: Any) -> None:
        result = _loaded_wrapper.infer()
        assert 0.0 <= result["accuracy_proxy"] <= 1.0

    def test_infer_without_explicit_load_still_works(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        # Do NOT call load() — infer() should auto-load
        result = mw.infer()
        assert result["simulated"] is True
        assert result["backend"] == "simulation"

    def test_infer_returns_different_latencies(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        latencies = [mw.infer()["latency_ms"] for _ in range(5)]
        # Gaussian randomness should produce some variation
        assert len(set(round(l, 1) for l in latencies)) > 1, (
            "Expected variation in simulated latency"
        )

    @pytest.mark.slow
    def test_latency_approximately_gaussian(self) -> None:
        from rams.models import PROFILES, ModelWrapper

        """Mean of 10 inferences should be within 3σ of the profile mean."""
        mw = ModelWrapper(Tier.NANO, simulate=True)
        n = 10
        latencies = [mw.infer()["latency_ms"] for _ in range(n)]
        mean = sum(latencies) / n
        expected_mean = PROFILES[Tier.NANO].latency_mean_ms
        expected_std = PROFILES[Tier.NANO].latency_std_ms
        assert abs(mean - expected_mean) < 3 * expected_std, (
            f"Mean latency {mean:.1f}ms not within 3σ of "
            f"{expected_mean}±{expected_std}ms"
        )


class TestModelWrapperUnload:
    """``ModelWrapper.unload()``."""

    def test_unload_sets_loaded_false(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        assert mw._loaded is True
        mw.unload()
        assert mw._loaded is False

    def test_unload_sets_model_none(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        mw.unload()
        assert mw._model is None

    def test_infer_after_unload_reloads(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.load()
        mw.unload()
        # infer() should call load() automatically
        result = mw.infer()
        assert result["simulated"] is True
        assert mw._loaded is True

    def test_unload_idempotent(self) -> None:
        from rams.models import ModelWrapper

        mw = ModelWrapper(Tier.NANO, simulate=True)
        mw.unload()
        mw.unload()  # second call should not raise
        assert mw._loaded is False


# ===================================================================
# ModelLibrary — simulation mode
# ===================================================================


class TestModelLibrary:
    """``ModelLibrary`` with ``simulate=True``."""

    RESULT_KEYS = {"tier", "simulated", "backend", "latency_ms",
                   "detections", "accuracy_proxy"}

    def test_init_creates_three_wrappers(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        assert len(lib._models) == 3
        for tier in Tier:
            assert tier in lib._models

    def test_get_returns_correct_tier(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        for tier in Tier:
            wrapper = lib.get(tier)
            assert wrapper.tier == tier
            assert wrapper.simulate is True

    def test_load_all_succeeds(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        lib.load_all()
        for tier in Tier:
            assert lib.get(tier)._loaded is True

    def test_load_all_idempotent(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        lib.load_all()
        lib.load_all()  # second call should not raise
        for tier in Tier:
            assert lib.get(tier)._loaded is True

    def test_infer_nano_returns_valid_dict(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        result = lib.infer(Tier.NANO)
        assert isinstance(result, dict)
        assert set(result.keys()) == self.RESULT_KEYS
        assert result["tier"] == "NANO"
        assert result["simulated"] is True

    def test_infer_small_returns_valid_dict(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        result = lib.infer(Tier.SMALL)
        assert result["tier"] == "SMALL"
        assert result["simulated"] is True

    def test_infer_medium_returns_valid_dict(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        result = lib.infer(Tier.MEDIUM)
        assert result["tier"] == "MEDIUM"
        assert result["simulated"] is True

    def test_infer_with_frame_none(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        result = lib.infer(Tier.NANO, frame=None)
        assert result["simulated"] is True
        assert result["backend"] == "simulation"

    def test_unload_all_unloads_all_tiers(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        lib.load_all()
        lib.unload_all()
        for tier in Tier:
            wrapper = lib.get(tier)
            assert wrapper._loaded is False
            assert wrapper._model is None

    def test_infer_after_unload_all_reloads(self) -> None:
        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        lib.load_all()
        lib.unload_all()
        # infer() should auto-load
        result = lib.infer(Tier.NANO)
        assert result["simulated"] is True
        assert lib.get(Tier.NANO)._loaded is True

    def test_infer_latency_grows_with_tier_size(self) -> None:
        import statistics

        from rams.models import ModelLibrary

        lib = ModelLibrary(simulate=True)
        means = []
        for tier in Tier:
            lats = [lib.infer(tier)["latency_ms"] for _ in range(5)]
            means.append(statistics.mean(lats))
        # NANO should be fastest, MEDIUM slowest on average
        assert means[0] <= means[2], (
            f"NANO mean {means[0]:.1f}ms > MEDIUM mean {means[2]:.1f}ms"
        )


# ===================================================================
# COCO_NAMES
# ===================================================================


class TestCocoNames:
    """``COCO_NAMES`` list."""

    def test_length_is_80(self) -> None:
        from rams.models import COCO_NAMES

        assert len(COCO_NAMES) == 80

    def test_contains_person(self) -> None:
        from rams.models import COCO_NAMES

        assert "person" in COCO_NAMES

    def test_contains_bicycle(self) -> None:
        from rams.models import COCO_NAMES

        assert "bicycle" in COCO_NAMES

    def test_contains_car(self) -> None:
        from rams.models import COCO_NAMES

        assert "car" in COCO_NAMES

    def test_all_strings(self) -> None:
        from rams.models import COCO_NAMES

        for name in COCO_NAMES:
            assert isinstance(name, str), f"Expected str, got {type(name)}"

    def test_no_duplicates(self) -> None:
        from rams.models import COCO_NAMES

        assert len(COCO_NAMES) == len(set(COCO_NAMES))


# ===================================================================
# _parse_onnx_output
# ===================================================================


class TestParseOnnxOutput:
    """``_parse_onnx_output()`` — requires numpy.

    The real ONNX output has shape ``(1, 84, 8400)`` where the 84 rows
    are: 4 bbox coords + 80 class scores.  The function transposes to
    ``(8400, 84)`` so each row is a candidate detection.
    """

    def _make_mock_output(
        self, n_boxes: int = 5, conf: float = 0.9, cls_id: int = 0
    ) -> tuple:
        """Build a tuple matching ONNX output shape ``(1, 84, 8400)``."""
        import numpy as np

        preds = np.zeros((1, 84, 8400), dtype=np.float32)
        for i in range(n_boxes):
            cx = 100.0 + i * 50
            cy = 100.0 + i * 30
            w = 40.0 + i * 5
            h = 60.0 + i * 4
            preds[0, :4, i] = [cx, cy, w, h]
            preds[0, 4 + cls_id, i] = conf
        return (preds,)

    def test_returns_list(self) -> None:
        from rams.models import _parse_onnx_output

        out = self._make_mock_output()
        dets = _parse_onnx_output(out)
        assert isinstance(dets, list)

    def test_each_detection_has_expected_keys(self) -> None:
        from rams.models import _parse_onnx_output

        out = self._make_mock_output()
        dets = _parse_onnx_output(out)
        for det in dets:
            assert "class" in det
            assert "conf" in det
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4

    def test_class_name_from_coco(self) -> None:
        from rams.models import COCO_NAMES, _parse_onnx_output

        cls_id = COCO_NAMES.index("person")  # 0
        out = self._make_mock_output(n_boxes=1, cls_id=cls_id)
        dets = _parse_onnx_output(out)
        assert dets[0]["class"] == "person"

    def test_conf_threshold_filters_low_conf(self) -> None:
        from rams.models import _parse_onnx_output

        # Two boxes: one above threshold, one below
        import numpy as np

        preds = np.zeros((1, 84, 8400), dtype=np.float32)
        # First grid cell: bbox + high score for class 0
        preds[0, :4, 0] = [100.0, 100.0, 40.0, 60.0]
        preds[0, 4 + 0, 0] = 0.90
        # Second grid cell: bbox + low score for class 0
        preds[0, :4, 1] = [200.0, 200.0, 50.0, 70.0]
        preds[0, 4 + 0, 1] = 0.10
        out = (preds,)
        dets = _parse_onnx_output(out, conf_thresh=0.25)
        assert len(dets) == 1
        assert dets[0]["conf"] == pytest.approx(0.90, abs=1e-5)

    def test_class_id_beyond_coco_falls_back_to_string(self) -> None:
        """When argmax returns a class ID >= len(COCO_NAMES), fall back to
        the string representation of the ID.  We force this by temporarily
        shortening ``COCO_NAMES`` to 1 element."""
        from unittest.mock import patch

        import numpy as np

        preds = np.zeros((1, 84, 8400), dtype=np.float32)
        preds[0, :4, 0] = [100.0, 100.0, 40.0, 60.0]
        preds[0, 4 + 5, 0] = 0.90  # class_id = 5 (beyond the 1-element list)
        out = (preds,)

        with patch("rams.models.COCO_NAMES", ["only_person"]):
            from rams.models import _parse_onnx_output

            dets = _parse_onnx_output(out, conf_thresh=0.25)
            assert len(dets) == 1
            assert dets[0]["class"] == "5"

    def test_conf_is_float(self) -> None:
        from rams.models import _parse_onnx_output

        out = self._make_mock_output()
        dets = _parse_onnx_output(out)
        assert isinstance(dets[0]["conf"], float)

    def test_xyxy_coordinates(self) -> None:
        from rams.models import _parse_onnx_output

        # Single box at center (100, 100) with size (40, 60)
        import numpy as np

        preds = np.zeros((1, 84, 8400), dtype=np.float32)
        preds[0, :4, 0] = [100.0, 100.0, 40.0, 60.0]
        preds[0, 4 + 0, 0] = 0.90
        out = (preds,)
        dets = _parse_onnx_output(out)
        # xyxy = [cx-w/2, cy-h/2, cx+w/2, cy+h/2] = [80, 70, 120, 130]
        assert dets[0]["xyxy"] == [80.0, 70.0, 120.0, 130.0]

    def test_returns_multiple_detections(self) -> None:
        from rams.models import _parse_onnx_output

        out = self._make_mock_output(n_boxes=10)
        dets = _parse_onnx_output(out)
        assert len(dets) == 10
