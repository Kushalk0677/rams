"""
Tests for rams.config — configuration loading, caching, and accessors.

Fixtures used
-------------
- isolated_config (conftest) — temporary copy of default.yaml for mutation
- tmp_path (built-in) — temporary directory for isolated config files
- monkeypatch (built-in) — override module-level constants

Note: The ``_reset_config_cache`` autouse fixture is defined in
``conftest.py`` and clears the LRU config cache before and after
*every* test, so tests in this file are automatically isolated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ===================================================================
# load_config
# ===================================================================


class TestLoadConfig:
    """Unit tests for ``load_config()``."""

    def test_returns_dict(self) -> None:
        from rams.config import load_config

        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_has_expected_top_level_keys(self) -> None:
        from rams.config import load_config

        cfg = load_config()
        expected = {"monitor", "policy", "models", "benchmark"}
        assert expected.issubset(cfg.keys()), f"Missing keys: {expected - cfg.keys()}"

    def test_caching_returns_same_object(self) -> None:
        from rams.config import load_config

        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1 is cfg2, "LRU cache should return the same dict object"

    def test_reload_config_clears_cache(self) -> None:
        from rams.config import load_config, reload_config

        cfg1 = load_config()
        reload_config()
        cfg2 = load_config()
        assert cfg1 is not cfg2, "reload_config() should return a fresh object"
        assert cfg1 == cfg2, "content should be identical after reload"

    def test_returns_empty_dict_when_file_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rams.config import load_config

        fake = Path("/does/not/exist.yaml")
        monkeypatch.setattr("rams.config.CONFIG_PATH", fake)
        cfg = load_config()
        assert cfg == {}


# ===================================================================
# get_monitor_hz
# ===================================================================


class TestGetMonitorHz:
    """Unit tests for ``get_monitor_hz()``."""

    def test_returns_hz_from_config(self) -> None:
        from rams.config import get_monitor_hz

        # Real config has monitor.hz: 10.0
        assert get_monitor_hz() == 10.0

    def test_returns_custom_default_when_config_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rams.config import get_monitor_hz

        fake = Path("/does/not/exist.yaml")
        monkeypatch.setattr("rams.config.CONFIG_PATH", fake)
        assert get_monitor_hz(default=20.0) == 20.0


# ===================================================================
# get_default_simulate
# ===================================================================


class TestGetDefaultSimulate:
    """Unit tests for ``get_default_simulate()``."""

    def test_returns_false(self) -> None:
        from rams.config import get_default_simulate

        assert get_default_simulate() is False

    def test_returns_false_when_config_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rams.config import get_default_simulate

        fake = Path("/does/not/exist.yaml")
        monkeypatch.setattr("rams.config.CONFIG_PATH", fake)
        assert get_default_simulate() is False


# ===================================================================
# get_default_policy_name
# ===================================================================


class TestGetDefaultPolicyName:
    """Unit tests for ``get_default_policy_name()``."""

    def test_returns_safety(self) -> None:
        from rams.config import get_default_policy_name

        assert get_default_policy_name() == "safety"

    def test_returns_safety_when_config_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rams.config import get_default_policy_name

        fake = Path("/does/not/exist.yaml")
        monkeypatch.setattr("rams.config.CONFIG_PATH", fake)
        assert get_default_policy_name() == "safety"


# ===================================================================
# get_policy_kwargs
# ===================================================================


class TestGetPolicyKwargs:
    """Unit tests for ``get_policy_kwargs()``."""

    def test_threshold_policy_returns_expected_keys(self) -> None:
        from rams.config import get_policy_kwargs

        kwargs = get_policy_kwargs("threshold")
        assert "lo_thresh" in kwargs
        assert "hi_thresh" in kwargs
        assert "hysteresis_window" in kwargs
        assert isinstance(kwargs["lo_thresh"], (int, float))
        assert isinstance(kwargs["hi_thresh"], (int, float))
        assert isinstance(kwargs["hysteresis_window"], int)

    def test_unknown_policy_returns_empty_dict(self) -> None:
        from rams.config import get_policy_kwargs

        assert get_policy_kwargs("nonexistent_policy_xyz") == {}

    def test_empty_config_returns_empty_dict(
        self, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A policy name that exists but has a null/empty value."""
        from rams.config import CONFIG_PATH, get_policy_kwargs, load_config, reload_config

        import yaml

        data = yaml.safe_load(isolated_config.read_text(encoding="utf-8"))
        data["policy"]["threshold"] = None  # null in YAML
        isolated_config.write_text(yaml.dump(data), encoding="utf-8")

        monkeypatch.setattr("rams.config.CONFIG_PATH", isolated_config)
        reload_config()  # clears cache & loads from the isolated path
        kwargs = get_policy_kwargs("threshold")
        assert kwargs == {}

    def test_empty_dict_value_returns_empty_dict(
        self, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A policy name mapped to ``{}`` instead of ``null``."""
        from rams.config import get_policy_kwargs, reload_config

        import yaml

        data = yaml.safe_load(isolated_config.read_text(encoding="utf-8"))
        data["policy"]["threshold"] = {}
        isolated_config.write_text(yaml.dump(data), encoding="utf-8")

        monkeypatch.setattr("rams.config.CONFIG_PATH", isolated_config)
        reload_config()
        kwargs = get_policy_kwargs("threshold")
        assert kwargs == {}


# ===================================================================
# Everything returns defaults when config is missing
# ===================================================================


class TestAllGettersWithMissingConfig:
    """Verify every getter falls back to its default when no config exists."""

    def test_all_return_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rams.config import (
            get_default_policy_name,
            get_default_simulate,
            get_monitor_hz,
            get_policy_kwargs,
        )

        fake = Path("/does/not/exist.yaml")
        monkeypatch.setattr("rams.config.CONFIG_PATH", fake)

        assert get_monitor_hz() == 10.0
        assert get_default_simulate() is False
        assert get_default_policy_name() == "safety"
        assert get_policy_kwargs("threshold") == {}
