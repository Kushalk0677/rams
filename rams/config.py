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
    cfg_path = Path(path) if path else CONFIG_PATH
    if yaml is None or not cfg_path.exists():
        return {}
    with open(cfg_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def reload_config(path: str | None = None) -> dict[str, Any]:
    load_config.cache_clear()
    return load_config(path)


def get_monitor_hz(default: float = 10.0) -> float:
    cfg = load_config()
    return float(cfg.get('monitor', {}).get('hz', default))


def get_default_simulate(default: bool = False) -> bool:
    cfg = load_config()
    return bool(cfg.get('models', {}).get('simulate', default))


def get_default_policy_name(default: str = 'safety') -> str:
    cfg = load_config()
    return str(cfg.get('policy', {}).get('default', default))


def get_policy_kwargs(policy_name: str) -> dict[str, Any]:
    cfg = load_config()
    policy_cfg = cfg.get('policy', {})
    values = policy_cfg.get(policy_name, {})
    return dict(values) if isinstance(values, dict) else {}
