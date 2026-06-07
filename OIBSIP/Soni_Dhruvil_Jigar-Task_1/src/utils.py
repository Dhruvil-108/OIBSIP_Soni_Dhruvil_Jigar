"""Shared utilities for configuration and path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = PROJECT_ROOT / "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration file."""

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    with config_file.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a project-relative or absolute path."""

    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_parent_directory(path_value: str | Path) -> Path:
    """Create the parent directory for a path and return the resolved path."""

    resolved_path = resolve_path(path_value)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path
