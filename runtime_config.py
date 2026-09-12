"""Single source of truth for how both runtimes reach their storage.

Before this module existed the two runtimes resolved the same information by
different means:

  * the trainer read a ``repository`` block embedded in each ``train_config``
  * the server assembled the same structure from flat environment variables
    (see the old ``load_env`` in ``server/main.py``)

Two independent paths meant the two could silently disagree about which
database they were using, so the server could serve a registry the trainer
never wrote to. Everything now resolves through :func:`load`.

Resolution order (first wins):

  1. an explicit ``path`` argument
  2. ``$MLOPS_CONFIG``
  3. ``./config/runtime.json``           (relative to the working directory)
  4. ``<repo root>/config/runtime.json``
  5. built-in defaults

Environment variables are then overlaid on top, so a deployment can point at a
different database without editing the file. Because both runtimes apply the
same overlay, an override can no longer desynchronise them.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = pathlib.Path("config") / "runtime.json"

DEFAULTS: dict[str, Any] = {
    "experiment_id": "experiment_001",
    "column_reference": "commodity_flow",
    "data": {
        "type": "sqlite",
        "properties": {"name": "example.db", "migrate": True},
    },
    "object": {
        "type": "sqlite",
        "properties": {"name": "example.db", "migrate": True},
    },
}

# Environment variable -> (section, dotted key). A None section means top level.
ENV_OVERRIDES: dict[str, tuple[Optional[str], str]] = {
    "EXPERIMENT_ID": (None, "experiment_id"),
    "COLUMN_REFERENCE": (None, "column_reference"),
    "REPOSITORY_DATA": ("data", "type"),
    "REPOSITORY_DATA_PATH": ("data", "properties.name"),
    "REPOSITORY_OBJECT": ("object", "type"),
    "REPOSITORY_OBJECT_PATH": ("object", "properties.name"),
}

SUPPORTED_BACKENDS = ("sqlite", "disk", "noop")


def _resolve_path(path: Optional[str] = None) -> Optional[pathlib.Path]:
    if path:
        return pathlib.Path(path).expanduser()
    from_env = os.getenv("MLOPS_CONFIG")
    if from_env:
        return pathlib.Path(from_env).expanduser()
    cwd_candidate = pathlib.Path.cwd() / DEFAULT_CONFIG_PATH
    if cwd_candidate.exists():
        return cwd_candidate
    root_candidate = REPO_ROOT / DEFAULT_CONFIG_PATH
    if root_candidate.exists():
        return root_candidate
    return None


def _set(cfg: dict, section: Optional[str], dotted: str, value: Any) -> None:
    target = cfg if section is None else cfg.setdefault(section, {})
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if key.startswith("_"):  # allow _comment keys in the json file
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _validate(cfg: dict) -> None:
    for section in ("data", "object"):
        backend = (cfg.get(section) or {}).get("type")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"runtime config: unsupported {section} backend {backend!r}; "
                f"expected one of {list(SUPPORTED_BACKENDS)}"
            )


def load(path: Optional[str] = None) -> dict:
    """Return the resolved runtime configuration as a Facade instruction dict."""
    cfg = copy.deepcopy(DEFAULTS)

    resolved = _resolve_path(path)
    if resolved is not None and resolved.exists():
        try:
            file_cfg = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"runtime config {resolved} is not valid JSON: {exc}") from exc
        _merge(cfg, file_cfg)

    for env_var, (section, dotted) in ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw not in (None, ""):
            _set(cfg, section, dotted, raw)

    # The sqlite repositories expect a boolean.
    for section in ("data", "object"):
        props = cfg.get(section, {}).get("properties")
        if isinstance(props, dict) and isinstance(props.get("migrate"), str):
            props["migrate"] = props["migrate"].strip().lower() in ("1", "true", "yes", "on")

    _validate(cfg)
    return cfg


def source(path: Optional[str] = None) -> str:
    """Where the configuration file was read from, for logs and /health."""
    resolved = _resolve_path(path)
    if resolved is not None and resolved.exists():
        return str(resolved)
    return "built-in defaults"


def applied_overrides() -> dict:
    """Environment variables that overrode the configuration file."""
    return {
        var: os.environ[var]
        for var in ENV_OVERRIDES
        if os.getenv(var) not in (None, "")
    }


def describe(path: Optional[str] = None) -> dict:
    """A summary suitable for logging or a health payload.

    Reports the file *and* the environment variables that overrode it. Knowing
    only the file is misleading: a deployment can set every repository value
    from the environment, leaving the file irrelevant.
    """
    cfg = load(path)
    return {
        "source": source(path),
        "environment_overrides": applied_overrides(),
        "experiment_id": cfg.get("experiment_id"),
        "column_reference": cfg.get("column_reference"),
        "data_backend": cfg.get("data", {}).get("type"),
        "data_path": cfg.get("data", {}).get("properties", {}).get("name"),
        "object_backend": cfg.get("object", {}).get("type"),
        "object_path": cfg.get("object", {}).get("properties", {}).get("name"),
    }
