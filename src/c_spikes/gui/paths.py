from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Union


PROJECT_ROOT_ENV = "C_SPIKES_PROJECT_ROOT"
PathInput = Union[str, Path]


def _looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "c_spikes").is_dir()


def configure_project_root(
    project_root: PathInput,
    *,
    environ: Optional[MutableMapping[str, str]] = None,
) -> Path:
    """Publish the checkout root for GUI code imported from a non-editable install."""
    active_environ = os.environ if environ is None else environ
    configured = str(active_environ.get(PROJECT_ROOT_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()

    resolved = Path(project_root).expanduser().resolve()
    active_environ[PROJECT_ROOT_ENV] = str(resolved)
    return resolved


def resolve_project_root(
    *,
    module_file: Optional[PathInput] = None,
    environ: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathInput] = None,
) -> Path:
    """Resolve GUI repository assets without assuming a fixed installed-package depth."""
    active_environ = os.environ if environ is None else environ
    configured = str(active_environ.get(PROJECT_ROOT_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()

    module_path = Path(__file__ if module_file is None else module_file).resolve()
    for candidate in module_path.parents:
        if _looks_like_project_root(candidate):
            return candidate

    working_dir = Path.cwd() if cwd is None else Path(cwd)
    return working_dir.expanduser().resolve()


__all__ = ["PROJECT_ROOT_ENV", "configure_project_root", "resolve_project_root"]
