from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

try:
    from ruamel.yaml import YAML
except Exception:  # pragma: no cover - optional import fallback
    YAML = None  # type: ignore


def list_cascade_available_models(model_root: Path) -> List[str]:
    yaml_path = Path(model_root) / "available_models.yaml"
    if not yaml_path.exists() or YAML is None:
        return []
    yaml = YAML(typ="safe")
    data = yaml.load(yaml_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return sorted(str(k) for k in data.keys())
    return []


def list_cascade_local_models(model_root: Path) -> List[str]:
    model_root = Path(model_root)
    models: List[str] = []
    if not model_root.exists():
        return models
    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_cascade_model_dir(child):
            models.append(child.name)
    return models


def list_ens2_model_dirs(model_root: Path) -> List[Path]:
    model_root = Path(model_root)
    dirs: List[Path] = []
    if not model_root.exists():
        return dirs
    if _has_ens2_weights(model_root):
        dirs.append(model_root)
    for child in sorted(model_root.iterdir()):
        if child.is_dir() and _has_ens2_weights(child):
            dirs.append(child)
    return dirs


def format_model_dir_label(root: Path, path: Path) -> str:
    root = Path(root)
    path = Path(path)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.name
    if str(rel) == ".":
        return f"{root.name} (root)"
    return str(rel)


def list_biophys_model_dirs(model_root: Path) -> List[Path]:
    model_root = Path(model_root)
    dirs: List[Path] = []
    if not model_root.exists():
        return dirs
    if detect_biophys_model_kind(model_root):
        dirs.append(model_root)
    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        if detect_biophys_model_kind(child):
            dirs.append(child)
    return dirs


def detect_biophys_model_kind(path: Path) -> Optional[str]:
    path = Path(path)
    if _is_cascade_model_dir(path):
        return "cascade"
    if _has_ens2_weights(path):
        return "ens2"
    return None


def _is_cascade_model_dir(path: Path) -> bool:
    return (
        (path / "config.yaml").exists()
        or any(path.glob("*.keras"))
        or any(path.glob("*.h5"))
    )


def _has_ens2_weights(path: Path) -> bool:
    return (path / "exc_ens2_pub.pt").exists() or (path / "inh_ens2_pub.pt").exists()
