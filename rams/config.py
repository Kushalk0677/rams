from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'default.yaml'


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the RAMS YAML configuration.

    Results are LRU-cached and reused on subsequent calls.
    Returns an empty dict when the file is missing or PyYAML is not installed.
    Use reload_config() to clear the cache and re-read from disk.

    Args:
        path: Optional explicit path to a YAML config file.
              Defaults to configs/default.yaml relative to the project root.

    Returns:
        Parsed config dictionary, or {} on failure.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    if yaml is None or not cfg_path.exists():
        return {}
    with open(cfg_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def reload_config(path: str | None = None) -> dict[str, Any]:
    """Clear the cached config and reload from disk.

    Returns:
        Freshly parsed config dictionary.
    """
    load_config.cache_clear()
    return load_config(path)


def get_monitor_hz(default: float = 10.0) -> float:
    """Return the resource monitor sampling frequency from config.

    Args:
        default: Fallback value when config key is missing.

    Returns:
        Sampling rate in Hz.
    """
    cfg = load_config()
    return float(cfg.get('monitor', {}).get('hz', default))


def get_default_simulate(default: bool = False) -> bool:
    """Return whether to run in simulation-only mode from config.

    Args:
        default: Fallback value when config key is missing.

    Returns:
        True if no real model files should be loaded.
    """
    cfg = load_config()
    return bool(cfg.get('models', {}).get('simulate', default))


def get_default_policy_name(default: str = 'safety') -> str:
    """Return the name of the default switching policy from config.

    Args:
        default: Fallback value when config key is missing.

    Returns:
        Policy name string, e.g. 'threshold', 'predictive', 'safety'.
    """
    cfg = load_config()
    return str(cfg.get('policy', {}).get('default', default))


def get_policy_kwargs(policy_name: str) -> dict[str, Any]:
    """Return the config dictionary for a named policy.

    Args:
        policy_name: Key in the config's policy section, e.g. 'threshold'.

    Returns:
        Dict of keyword arguments for the policy constructor, or {} if
        the policy_name is not found or the value is not a dict.
    """
    cfg = load_config()
    policy_cfg = cfg.get('policy', {})
    values = policy_cfg.get(policy_name, {})
    return dict(values) if isinstance(values, dict) else {}
