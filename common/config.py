"""Configuration helpers for local config files plus env-based API keys.

Keys present in the local top-level ``config`` file are overridden by matching
environment variables. Environment-only values are included automatically for
variables ending with ``_API_KEY``. Callers that need additional env-only keys
can pass them explicitly to ``load_config``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from .paths import repo_root


def config_path() -> Path:
    return repo_root() / "config"


def load_config(extra_env_keys: Iterable[str] | None = None) -> dict[str, Any]:
    """Load the local JSON config file and apply env overrides.

    Existing config keys are overridden by matching environment variables.
    Environment-only ``*_API_KEY`` values are also added automatically.
    Pass ``extra_env_keys`` to include additional env-only keys.
    """
    data: dict[str, Any] = {}
    path = config_path()

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)

        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a JSON object: {path}")

        data.update(loaded)

    return _apply_environment_overrides(data, extra_env_keys=extra_env_keys)


def get_api_key(name: str) -> str:
    """Return an API key from environment variables or local config."""
    key_name = name.strip()
    if not key_name:
        raise ValueError("API key name must not be empty")

    for candidate in _candidate_names(key_name):
        value = os.getenv(candidate)
        if value:
            return value

    config = load_config()
    for candidate in _candidate_names(key_name):
        value = config.get(candidate)
        if isinstance(value, str) and value:
            return value

    candidates = ", ".join(_candidate_names(key_name))
    raise KeyError(
        f"Missing API key for '{key_name}'. Set one of [{candidates}] in the environment or the top-level config file."
    )


def _apply_environment_overrides(
    config: dict[str, Any],
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    merged = dict(config)
    for key in list(merged):
        env_value = _first_environment_value(key)
        if env_value is not None:
            merged[key] = env_value

    for key, value in _environment_backed_config().items():
        merged[key] = value

    if extra_env_keys is not None:
        for key in extra_env_keys:
            env_value = _first_environment_value(key)
            if env_value is not None:
                merged[key] = env_value

    return merged


def _first_environment_value(name: str) -> str | None:
    for candidate in _candidate_names(name):
        value = os.getenv(candidate)
        if value:
            return value
    return None


def _environment_backed_config() -> dict[str, str]:
    config: dict[str, str] = {}
    for name, value in os.environ.items():
        if not value:
            continue
        if name.lower().endswith("_api_key"):
            config[name.lower()] = value
    return config


def _candidate_names(name: str) -> list[str]:
    trimmed = name.strip()
    normalized = "".join(ch if ch.isalnum() else "_" for ch in trimmed)
    normalized = "_".join(part for part in normalized.split("_") if part)

    candidates: list[str] = []
    for candidate in (
        trimmed,
        trimmed.lower(),
        trimmed.upper(),
        normalized,
        normalized.lower(),
        normalized.upper(),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    upper_base = normalized.upper()
    if upper_base and not upper_base.endswith("_API_KEY"):
        candidates.append(f"{upper_base}_API_KEY")

    lower_base = normalized.lower()
    if lower_base and not lower_base.endswith("_api_key"):
        candidate = f"{lower_base}_api_key"
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates
