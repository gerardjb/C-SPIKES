#!/usr/bin/env python
"""Validate .mat files for C-SPIKES GUI ingestion."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

try:  # Optional for MATLAB v7.3 files
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore


REQUIRED_KEYS = ("time_stamps", "dff")
OPTIONAL_KEYS = ("ap_times",)


def _print_header(path: Path) -> None:
    print(f"\n== {path.name} ==")


def _fmt_shape(shape: Optional[Tuple[int, ...]]) -> str:
    if shape is None:
        return "<missing>"
    return "(" + ", ".join(str(int(s)) for s in shape) + ")"


def _infer_epochs(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 0
    if len(shape) == 1:
        return 1
    return int(shape[0])


def _safe_loadmat(path: Path) -> Dict[str, Any]:
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _inspect_with_h5py(path: Path, deep: bool) -> Dict[str, Any]:
    if h5py is None:
        raise RuntimeError("MATLAB v7.3 file requires h5py (not installed).")
    out: Dict[str, Any] = {"names": {}, "data": {}}
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            try:
                out["names"][key] = tuple(int(v) for v in h5[key].shape)
            except Exception:
                out["names"][key] = None
        if deep:
            for key in REQUIRED_KEYS + OPTIONAL_KEYS:
                if key in h5:
                    out["data"][key] = np.array(h5[key])
    return out


def _summarize_deep(
    time_stamps: np.ndarray,
    dff: np.ndarray,
    ap_times: Optional[np.ndarray],
) -> List[str]:
    lines: List[str] = []

    time_stamps = np.asarray(time_stamps)
    dff = np.asarray(dff)
    if time_stamps.shape != dff.shape:
        lines.append(f"WARNING: time_stamps shape {time_stamps.shape} != dff shape {dff.shape}")

    n_epochs = _infer_epochs(time_stamps.shape)
    if n_epochs == 0:
        lines.append("ERROR: could not infer epochs from time_stamps shape.")
        return lines

    if time_stamps.ndim == 1:
        time_stamps = time_stamps.reshape(1, -1)
    if dff.ndim == 1:
        dff = dff.reshape(1, -1)

    for idx in range(min(n_epochs, 3)):
        t = np.asarray(time_stamps[idx], dtype=np.float64).ravel()
        t = t[np.isfinite(t)]
        if t.size >= 2:
            dt = float(np.median(np.diff(t)))
            lines.append(f"Epoch {idx}: samples={t.size}, dt~{dt:.6f}s, t=[{t.min():.3f}, {t.max():.3f}]")
        elif t.size == 1:
            lines.append(f"Epoch {idx}: single time stamp at {t[0]:.3f}s")
        else:
            lines.append(f"Epoch {idx}: no finite time stamps")

    if ap_times is None:
        lines.append("NOTE: ap_times missing (ground truth spikes unavailable)")
        return lines

    arr = np.asarray(ap_times)
    if arr.size == 0:
        lines.append("NOTE: ap_times empty")
        return lines

    if arr.dtype == object:
        flat = arr.ravel()
        if flat.size == n_epochs:
            counts = [np.asarray(flat[i]).ravel().size for i in range(min(n_epochs, 3))]
            lines.append(f"ap_times per-epoch counts (first 3): {counts}")
        else:
            try:
                numeric = np.asarray(flat, dtype=np.float64)
                lines.append(f"ap_times count: {numeric.size}")
            except Exception:
                lines.append("NOTE: ap_times object array could not be summarized")
    else:
        numeric = np.asarray(arr, dtype=np.float64).ravel()
        lines.append(f"ap_times count: {numeric.size}")
    return lines


def inspect_file(path: Path, deep: bool) -> Tuple[int, int]:
    errors = 0
    warnings = 0

    try:
        vars_info = sio.whosmat(path)
        names = {name: shape for name, shape, _ in vars_info}
        _print_header(path)
        for key in REQUIRED_KEYS:
            shape = names.get(key)
            print(f"{key}: {_fmt_shape(shape)}")
            if shape is None:
                print(f"ERROR: missing required key '{key}'")
                errors += 1
        for key in OPTIONAL_KEYS:
            shape = names.get(key)
            if shape is None:
                print(f"{key}: <missing>")
                warnings += 1
            else:
                print(f"{key}: {_fmt_shape(shape)}")

        if deep and errors == 0:
            data = _safe_loadmat(path)
            dff = data.get("dff")
            time_stamps = data.get("time_stamps")
            ap_times = data.get("ap_times") if "ap_times" in data else None
            lines = _summarize_deep(time_stamps, dff, ap_times)
            for line in lines:
                print(line)
                if line.startswith("ERROR"):
                    errors += 1
                elif line.startswith("WARNING"):
                    warnings += 1
    except NotImplementedError:
        info = _inspect_with_h5py(path, deep)
        names = info.get("names", {})
        _print_header(path)
        for key in REQUIRED_KEYS:
            shape = names.get(key)
            print(f"{key}: {_fmt_shape(shape)}")
            if shape is None:
                print(f"ERROR: missing required key '{key}'")
                errors += 1
        for key in OPTIONAL_KEYS:
            shape = names.get(key)
            if shape is None:
                print(f"{key}: <missing>")
                warnings += 1
            else:
                print(f"{key}: {_fmt_shape(shape)}")
        if deep and errors == 0:
            dff = info.get("data", {}).get("dff")
            time_stamps = info.get("data", {}).get("time_stamps")
            ap_times = info.get("data", {}).get("ap_times")
            if dff is not None and time_stamps is not None:
                lines = _summarize_deep(time_stamps, dff, ap_times)
                for line in lines:
                    print(line)
                    if line.startswith("ERROR"):
                        errors += 1
                    elif line.startswith("WARNING"):
                        warnings += 1
    except Exception as exc:
        _print_header(path)
        print(f"ERROR: failed to inspect {path}: {exc}")
        errors += 1

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate .mat files for C-SPIKES GUI ingestion")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory with .mat files")
    parser.add_argument("--pattern", default="*.mat", help="Glob pattern for datasets")
    parser.add_argument("--deep", action="store_true", help="Load arrays to compute basic stats")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to inspect")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory does not exist: {data_dir}")
        return 2

    files = sorted(data_dir.glob(args.pattern))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        print("No .mat files found.")
        return 1

    total_errors = 0
    total_warnings = 0
    for path in files:
        errs, warns = inspect_file(path, args.deep)
        total_errors += errs
        total_warnings += warns

    print("\nSummary")
    print(f"  files: {len(files)}")
    print(f"  errors: {total_errors}")
    print(f"  warnings: {total_warnings}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
