from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class NoisePreprocessSpec:
    source_dir: Path
    target_fs: float
    method: str = "interp_like_inference_v1"

    def fs_tag(self) -> str:
        fs = float(self.target_fs)
        if np.isclose(fs, round(fs)):
            return f"{int(round(fs))}Hz"
        token = f"{fs:.2f}".rstrip("0").rstrip(".")
        return token.replace(".", "p") + "Hz"


def _sha1_token(value: object, *, n: int = 10) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return digest[:n]


def _iter_noise_files(noise_dir: Path) -> Iterable[Path]:
    return sorted(noise_dir.glob("*.mat"))


def _load_cattached(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    if "CAttached" not in data:
        raise KeyError(f"Missing 'CAttached' in noise file: {path}")
    cattached = np.asarray(data["CAttached"])
    if cattached.ndim != 2 or cattached.shape[0] != 1:
        raise ValueError(f"Unexpected CAttached shape {cattached.shape} in {path}")
    return cattached


def _extract_noise_and_time(inner: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if inner.dtype.names is None or "gt_noise" not in inner.dtype.names:
        raise KeyError("Noise entry missing required field 'gt_noise'.")
    noise = np.asarray(inner["gt_noise"][0][0], dtype=np.float64).ravel()
    if "fluo_time" in inner.dtype.names:
        time = np.asarray(inner["fluo_time"][0][0], dtype=np.float64).ravel()
    else:
        # Fall back to the historical default (used in synth_gen when fluo_time is missing).
        default_fs = 121.9
        time = np.arange(noise.size, dtype=np.float64) / default_fs
    mask = np.isfinite(time) & np.isfinite(noise)
    return noise[mask], time[mask]


def _downsample_series(times: np.ndarray, values: np.ndarray, target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    if times.size < 2 or values.size < 2:
        return times, values
    dt = np.median(np.diff(times))
    if not np.isfinite(dt) or dt <= 0:
        return times, values
    fs = 1.0 / dt
    if fs <= target_fs:
        return times, values

    # Match the repo's default downsampling semantics (used in inference):
    # - if the ratio is ~integer, average contiguous blocks
    # - otherwise, resample by linear interpolation on an evenly spaced time grid
    ratio = fs / float(target_fs)
    block = int(round(ratio))
    if np.isclose(ratio, block, atol=1e-6):
        n_trim = (values.size // block) * block
        if n_trim == 0:
            return times[:1], values[:1]
        values_trim = values[:n_trim].reshape(-1, block)
        times_trim = times[:n_trim].reshape(-1, block)
        return times_trim.mean(axis=1), values_trim.mean(axis=1)

    duration = float(times[-1] - times[0])
    if duration <= 0:
        return times, values
    n_target = max(2, int(np.round(duration * target_fs)) + 1)
    new_times = np.linspace(float(times[0]), float(times[-1]), n_target, dtype=np.float64)
    new_values = np.interp(new_times, times, values).astype(np.float64, copy=False)
    return new_times, new_values


def _ensure_field(inner: np.ndarray, field: str) -> np.ndarray:
    names = tuple(inner.dtype.names or ())
    if field in names:
        return inner
    new_dtype = np.dtype(list(inner.dtype.descr) + [(field, "|O")])
    expanded = np.empty(inner.shape, dtype=new_dtype)
    for name in names:
        expanded[name] = inner[name]
    return expanded


def _rewrite_noise_entry(inner: np.ndarray, *, target_fs: float) -> np.ndarray:
    inner = np.asarray(inner)
    inner = _ensure_field(inner, "fluo_time")
    noise, time = _extract_noise_and_time(inner)
    ds_time, ds_noise = _downsample_series(time, noise, target_fs)
    inner_out = inner.copy()
    inner_out["gt_noise"][0][0] = np.asarray(ds_noise, dtype=np.float32)
    inner_out["fluo_time"][0][0] = np.asarray(ds_time, dtype=np.float64)
    return inner_out


def _materialize_downsampled_noise_file(src_path: Path, dst_path: Path, *, target_fs: float) -> None:
    cattached = _load_cattached(src_path)
    rewritten = np.empty_like(cattached)
    for idx in range(cattached.shape[1]):
        rewritten[0, idx] = _rewrite_noise_entry(cattached[0, idx], target_fs=target_fs)
    sio.savemat(dst_path, {"CAttached": rewritten})


def prepare_noise_dir(
    noise_dir: Path,
    *,
    target_fs: float,
    cache_root: Path = Path("results") / "inference_cache" / "noise_downsample",
) -> Path:
    """
    Create (or reuse) a cached, downsampled copy of a syn_gen noise directory.

    The output directory contains the same *.mat filenames as the source, but
    with each CAttached entry's (gt_noise, fluo_time) resampled to target_fs.
    """
    noise_dir = Path(noise_dir).expanduser().resolve()
    if not noise_dir.exists():
        raise FileNotFoundError(noise_dir)
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")

    spec = NoisePreprocessSpec(source_dir=noise_dir, target_fs=float(target_fs))
    src_files = list(_iter_noise_files(noise_dir))
    if not src_files:
        raise FileNotFoundError(f"No .mat files found in noise_dir={noise_dir}")

    token = _sha1_token((str(noise_dir), float(spec.target_fs), spec.method), n=10)
    out_dir = cache_root.expanduser().resolve() / f"{noise_dir.name}__{spec.fs_tag()}__{token}"
    manifest_path = out_dir / "manifest.json"

    expected_names = [p.name for p in src_files]
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source_dir") == str(noise_dir)
                and float(manifest.get("target_fs")) == float(spec.target_fs)
                and manifest.get("method") == spec.method
                and manifest.get("files") == expected_names
            ):
                if all((out_dir / name).exists() for name in expected_names):
                    return out_dir
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    for src_path in src_files:
        dst_path = out_dir / src_path.name
        if dst_path.exists():
            continue
        _materialize_downsampled_noise_file(src_path, dst_path, target_fs=spec.target_fs)

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(noise_dir),
        "target_fs": float(spec.target_fs),
        "method": spec.method,
        "output_dir": str(out_dir),
        "files": expected_names,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir
