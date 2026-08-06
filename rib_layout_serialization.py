"""Strict JSON serialization and portable artifact-path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rib_layout_env import PROJECT_ROOT


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize JSON without JavaScript-only NaN or Infinity tokens."""
    if "allow_nan" in kwargs:
        raise TypeError("strict_json_dumps controls allow_nan")
    return json.dumps(value, allow_nan=False, **kwargs)


def portable_artifact_path(
    path: str | Path,
    project_root: str | Path | None = None,
) -> str:
    """Return a POSIX repository-relative path when *path* lies in the source tree."""
    resolved = Path(path).resolve()
    root = Path(project_root or PROJECT_ROOT).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


__all__ = ["portable_artifact_path", "strict_json_dumps"]
