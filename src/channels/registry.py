"""Channel class registry.

YAML config
```
channels:
  fast:
    type: code_exec
    max_latency_s: 5
```
is resolved via `build_channel_from_config("code_exec", {...})` → callable.

Matches verl's own registry-for-everything pattern. Keeps reward channels
pluggable without making the reward manager depend on every channel class.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Type

_REGISTRY: Dict[str, Type] = {}


def register_channel(name: str):
    """Decorator: registers a channel class under `name`.

    Usage:
        @register_channel("code_exec")
        class CodeExecChannel:
            is_async = False
            def __call__(self, data): ...
    """
    def _wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"channel {name!r} already registered")
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_channel_class(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(
            f"channel {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_registered() -> list[str]:
    return sorted(_REGISTRY.keys())


def build_channel_from_config(type_name: str, config: dict[str, Any]):
    """Construct a channel instance from a YAML sub-config.

    `config` is the dict under channels.<name> in the YAML, e.g.
    `{type: code_exec, max_latency_s: 5, lambda_weight: 1.0}`.

    The `type` key is consumed; remaining keys are passed as kwargs.
    """
    cfg = dict(config)
    cfg.pop("type", None)
    cls = get_channel_class(type_name)
    return cls(**cfg)
