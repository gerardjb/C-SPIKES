#!/usr/bin/env python3
"""
Inspect a syn_gen synthetic dataset directory (results/Ground_truth/synth_*) and
report basic timebase + spike-statistics sanity checks.

This is a lightweight, torch-free diagnostic to help catch timebase mismatches
and unintended spike-rate shifts across synth generations.

Example:
  python scripts/inspect_synth_dir.py \
    --synth-dir results/Ground_truth/synth_jGCaMP8f_ANM478349_cell04_ms2_trial1__k1_r6_s1p3_d0p35_sb0_rerun01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class EpochStats:
    file: str
    epoch: int
    fs_hz: float
    duration_s: float
    n_spikes: int
    mean_rate_hz: float
    spike_in_range_frac: float


def _iter_epochs(ca: object) -> Iterable[object]:
    if isinstance(ca, np.ndarray):
        return ca.flat
    return [ca]


def _load_epoch_arrays(ep: object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(getattr(ep, "fluo_time"), dtype=np.float64).squeeze()
    y = np.asarray(getattr(ep, "fluo_mean"), dtype=np.float64).squeeze()
    ev = np.asarray(getattr(ep, "events_AP"), dtype=np.float64).squeeze()
    return t, y, ev


def _epoch_stats(mat_path: Path) -> List[EpochStats]:
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "CAttached" not in data:
        raise KeyError(f"Missing 'CAttached' in {mat_path}")
    ca = data["CAttached"]

    out: List[EpochStats] = []
    for idx, ep in enumerate(_iter_epochs(ca)):
        t, _y, ev = _load_epoch_arrays(ep)
        if t.size < 2:
            continue
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            continue
        dt = float(np.median(diffs))
        fs = 1.0 / dt
        duration = float(t[-1] - t[0])
        if duration <= 0:
            continue

        # ENS2 convention: events_AP are in 10 kHz samples
        sp_s = ev.astype(np.float64, copy=False) / 1e4
        sp_s = sp_s[np.isfinite(sp_s)]
        in_range = (sp_s >= t[0]) & (sp_s <= t[-1])
        in_frac = float(in_range.mean()) if sp_s.size else 1.0

        n_spikes = int(sp_s.size)
        mean_rate = float(n_spikes / duration)

        out.append(
            EpochStats(
                file=mat_path.name,
                epoch=int(idx),
                fs_hz=float(fs),
                duration_s=duration,
                n_spikes=n_spikes,
                mean_rate_hz=mean_rate,
                spike_in_range_frac=in_frac,
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synth-dir", type=Path, required=True, help="Synthetic directory (results/Ground_truth/synth_*).")
    p.add_argument("--pattern", type=str, default="*.mat", help="Glob pattern within synth-dir (default: *.mat).")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of .mat files to scan.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit code 2) if any epoch has spikes outside its fluo_time range.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    synth_dir = args.synth_dir.expanduser().resolve()
    if not synth_dir.exists():
        raise FileNotFoundError(synth_dir)

    mats = sorted(synth_dir.glob(args.pattern))
    if not mats:
        raise FileNotFoundError(f"No files in {synth_dir} matching {args.pattern!r}")
    if args.limit is not None:
        mats = mats[: int(args.limit)]

    all_epochs: List[EpochStats] = []
    for mat in mats:
        all_epochs.extend(_epoch_stats(mat))

    if not all_epochs:
        raise RuntimeError("No valid epochs found (missing/invalid fluo_time).")

    fs_vals = np.asarray([e.fs_hz for e in all_epochs], dtype=float)
    rate_vals = np.asarray([e.mean_rate_hz for e in all_epochs], dtype=float)
    dur_vals = np.asarray([e.duration_s for e in all_epochs], dtype=float)
    in_fracs = np.asarray([e.spike_in_range_frac for e in all_epochs], dtype=float)

    print(f"[synth] dir={synth_dir}")
    print(f"[synth] files={len(mats)} epochs={len(all_epochs)}")
    print(
        "[synth] fs_hz: "
        f"median={np.median(fs_vals):.3f} mean={np.mean(fs_vals):.3f} "
        f"min={np.min(fs_vals):.3f} max={np.max(fs_vals):.3f}"
    )
    print(
        "[synth] duration_s: "
        f"median={np.median(dur_vals):.3f} mean={np.mean(dur_vals):.3f} "
        f"min={np.min(dur_vals):.3f} max={np.max(dur_vals):.3f}"
    )
    print(
        "[synth] mean_rate_hz: "
        f"median={np.median(rate_vals):.3f} mean={np.mean(rate_vals):.3f} "
        f"min={np.min(rate_vals):.3f} max={np.max(rate_vals):.3f}"
    )
    bad = np.where(in_fracs < 1.0)[0]
    if bad.size:
        n_bad = int(bad.size)
        worst = int(bad[np.argmin(in_fracs[bad])])
        e = all_epochs[worst]
        print(
            f"[warn] spikes out of range in {n_bad}/{len(all_epochs)} epochs; "
            f"worst={e.file} epoch={e.epoch} in_range_frac={e.spike_in_range_frac:.3f}"
        )
        if args.strict:
            raise SystemExit(2)
    else:
        print("[synth] spike_in_range_frac: all epochs OK (1.000)")


if __name__ == "__main__":
    main()

