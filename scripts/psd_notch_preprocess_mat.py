#!/usr/bin/env python3
"""Detect narrow PSD peaks and notch-filter one epoch in a Janelia-style .mat file."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/c_spikes_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal


DEFAULT_INPUT = Path("sample_data/janelia_8f/excitatory/jGCaMP8f_ANM478348_cell01.mat")


@dataclass(frozen=True)
class PeakTarget:
    frequency_hz: float
    psd_db: float
    prominence_db: float
    width_hz: float
    band_low_hz: float
    band_high_hz: float


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {arr.shape}.")
    return arr


def _sampling_rate_hz(times: np.ndarray) -> float:
    diffs = np.diff(np.asarray(times, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("Could not infer sampling rate from time_stamps.")
    return float(1.0 / np.median(diffs))


def _fill_invalid(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(y)
    if finite.all():
        return y
    if not finite.any():
        raise ValueError("Selected epoch has no finite dff values.")
    x = np.arange(y.size, dtype=np.float64)
    y_filled = y.copy()
    y_filled[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return y_filled


def compute_psd_db(
    values: np.ndarray,
    *,
    fs_hz: float,
    nperseg: int,
    noverlap: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    y = _fill_invalid(values)
    y = y - np.nanmean(y)
    nperseg = int(min(max(8, nperseg), y.size))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(max(0, noverlap), nperseg - 1))
    freq, pxx = signal.welch(y, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)
    psd_db = 10.0 * np.log10(np.maximum(pxx, np.finfo(np.float64).tiny))
    return freq, psd_db


def detect_targets(
    freq: np.ndarray,
    psd_db: np.ndarray,
    *,
    prominence_db: float,
    max_peak_width_hz: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    notch_width_hz: float | None,
    notch_width_scale: float,
    min_notch_width_hz: float,
    max_notch_width_hz: float,
    nyquist_hz: float,
) -> list[PeakTarget]:
    if freq.size < 2:
        return []
    peaks, props = signal.find_peaks(psd_db, prominence=float(prominence_db))
    if peaks.size == 0:
        return []

    width_samples, _, _, _ = signal.peak_widths(psd_db, peaks, rel_height=0.5)
    df = float(np.median(np.diff(freq)))
    targets: list[PeakTarget] = []
    for i, peak_idx in enumerate(peaks):
        frequency = float(freq[peak_idx])
        width_hz = float(width_samples[i] * df)
        if frequency < min_frequency_hz or frequency > max_frequency_hz:
            continue
        if width_hz > max_peak_width_hz:
            continue
        if notch_width_hz is None:
            filter_width = np.clip(
                width_hz * float(notch_width_scale),
                float(min_notch_width_hz),
                float(max_notch_width_hz),
            )
        else:
            filter_width = float(notch_width_hz)
        low = max(1e-6, frequency - filter_width / 2.0)
        high = min(nyquist_hz * 0.999, frequency + filter_width / 2.0)
        if high <= low:
            continue
        targets.append(
            PeakTarget(
                frequency_hz=frequency,
                psd_db=float(psd_db[peak_idx]),
                prominence_db=float(props["prominences"][i]),
                width_hz=width_hz,
                band_low_hz=float(low),
                band_high_hz=float(high),
            )
        )
    return targets


def apply_bandstop_filters(
    values: np.ndarray,
    targets: Sequence[PeakTarget],
    *,
    fs_hz: float,
    order: int,
) -> np.ndarray:
    y = _fill_invalid(values)
    filtered = y.copy()
    for target in targets:
        sos = signal.iirfilter(
            int(order),
            [target.band_low_hz, target.band_high_hz],
            btype="bandstop",
            fs=float(fs_hz),
            output="sos",
        )
        filtered = signal.sosfiltfilt(sos, filtered)
    return filtered


def _default_output_mat(input_mat: Path, epoch: int) -> Path:
    # GUI SMC-viz treats `_epoch\d+` as a formal epoch-id delimiter in cache tags.
    # Keep generated dataset stems free of that token so cache tags remain parseable.
    return input_mat.with_name(f"{input_mat.stem}_psdnotch_e{int(epoch):02d}{input_mat.suffix}")


def _has_gui_epoch_token(path: Path) -> bool:
    return re.search(r"_epoch\d+", path.stem) is not None


def _write_peaks_csv(path: Path, targets: Sequence[PeakTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frequency_hz",
                "psd_db",
                "prominence_db",
                "peak_width_hz",
                "band_low_hz",
                "band_high_hz",
            ]
        )
        for target in targets:
            writer.writerow(
                [
                    f"{target.frequency_hz:.9g}",
                    f"{target.psd_db:.9g}",
                    f"{target.prominence_db:.9g}",
                    f"{target.width_hz:.9g}",
                    f"{target.band_low_hz:.9g}",
                    f"{target.band_high_hz:.9g}",
                ]
            )


def _plot_psd(
    path: Path,
    freq: np.ndarray,
    raw_psd_db: np.ndarray,
    filtered_psd_db: np.ndarray,
    targets: Sequence[PeakTarget],
    *,
    xlim: tuple[float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(freq, raw_psd_db, linewidth=1.0, label="raw")
    ax.plot(freq, filtered_psd_db, linewidth=1.0, label="notch filtered")
    for target in targets:
        ax.axvspan(target.band_low_hz, target.band_high_hz, color="tab:red", alpha=0.15)
        ax.plot(target.frequency_hz, target.psd_db, "o", color="tab:red", markersize=4)
        ax.annotate(
            f"{target.frequency_hz:.2f} Hz\n{target.prominence_db:.1f} dB",
            xy=(target.frequency_hz, target.psd_db),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlim(*xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Welch PSD and detected notch targets")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_timeseries(
    path: Path,
    times: np.ndarray,
    raw_values: np.ndarray,
    filtered_values: np.ndarray,
    *,
    xlim: tuple[float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(times, raw_values, color="0.35", linewidth=0.8, label="raw")
    ax.plot(times, filtered_values, color="tab:blue", linewidth=1.0, label="filtered")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    ax.set_title("Raw vs notch-filtered trace")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-mat", type=Path, default=DEFAULT_INPUT, help="Input .mat file.")
    parser.add_argument("--output-mat", type=Path, help="Output .mat path.")
    parser.add_argument("--plot-dir", type=Path, help="Directory for PNG/CSV diagnostics.")
    parser.add_argument("--epoch", type=int, default=1, help="1-based epoch index to filter.")
    parser.add_argument("--xlim", type=float, nargs=2, default=(355.0, 359.5), metavar=("START", "END"))
    parser.add_argument("--psd-xlim", type=float, nargs=2, default=(0.0, 30.0), metavar=("FMIN", "FMAX"))
    parser.add_argument("--welch-nperseg", type=int, default=2048, help="Welch segment length.")
    parser.add_argument("--welch-noverlap", type=int, help="Welch segment overlap. Defaults to nperseg/2.")
    parser.add_argument("--peak-prominence-db", type=float, default=10.0, help="Minimum PSD peak prominence.")
    parser.add_argument("--max-peak-width-hz", type=float, default=5.0, help="Maximum detected peak width.")
    parser.add_argument("--min-frequency-hz", type=float, default=0.5, help="Minimum frequency considered for filtering.")
    parser.add_argument("--max-frequency-hz", type=float, default=30.0, help="Maximum frequency considered for filtering.")
    parser.add_argument("--notch-width-hz", type=float, help="Fixed notch width. If omitted, width is derived from peak width.")
    parser.add_argument("--notch-width-scale", type=float, default=2.0, help="Multiplier applied to detected peak width.")
    parser.add_argument("--min-notch-width-hz", type=float, default=0.25, help="Minimum derived notch width.")
    parser.add_argument("--max-notch-width-hz", type=float, default=1.0, help="Maximum derived notch width.")
    parser.add_argument("--filter-order", type=int, default=4, help="IIR bandstop filter order.")
    parser.add_argument(
        "--allow-gui-ambiguous-name",
        action="store_true",
        help="Allow output .mat stems containing `_epoch<digits>`, which can confuse BiophysSMC viz cache resolution.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output .mat.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_mat = args.input_mat.resolve()
    if not input_mat.exists():
        print(f"[error] input .mat not found: {input_mat}", file=sys.stderr)
        return 2
    if args.epoch < 1:
        print("[error] --epoch is 1-based and must be >= 1.", file=sys.stderr)
        return 2

    output_mat = (args.output_mat or _default_output_mat(input_mat, args.epoch)).resolve()
    if _has_gui_epoch_token(output_mat) and not args.allow_gui_ambiguous_name:
        print(
            "[error] output .mat stem contains `_epoch<digits>`, which collides with GUI cache-tag parsing. "
            "Use a stem like `_psdnotch_e01` or pass --allow-gui-ambiguous-name.",
            file=sys.stderr,
        )
        return 2
    plot_dir = (args.plot_dir or output_mat.with_suffix("").with_name(f"{output_mat.stem}_plots")).resolve()
    if output_mat.exists() and not args.overwrite:
        print(f"[error] output exists; pass --overwrite to replace: {output_mat}", file=sys.stderr)
        return 2

    data = sio.loadmat(input_mat)
    if "time_stamps" not in data or "dff" not in data:
        print(f"[error] {input_mat} must contain time_stamps and dff.", file=sys.stderr)
        return 2

    time_stamps = _as_2d("time_stamps", np.asarray(data["time_stamps"], dtype=np.float64))
    dff_original = np.asarray(data["dff"])
    dff = _as_2d("dff", dff_original)
    if time_stamps.shape != dff.shape:
        print(f"[error] time_stamps shape {time_stamps.shape} != dff shape {dff.shape}.", file=sys.stderr)
        return 2
    epoch_idx = args.epoch - 1
    if epoch_idx >= dff.shape[0]:
        print(f"[error] epoch {args.epoch} out of range for {dff.shape[0]} epochs.", file=sys.stderr)
        return 2

    times = np.asarray(time_stamps[epoch_idx], dtype=np.float64)
    raw_values = np.asarray(dff[epoch_idx], dtype=np.float64)
    fs_hz = _sampling_rate_hz(times)
    nyquist = fs_hz / 2.0
    max_frequency = min(float(args.max_frequency_hz), nyquist * 0.999)

    freq, raw_psd_db = compute_psd_db(
        raw_values,
        fs_hz=fs_hz,
        nperseg=int(args.welch_nperseg),
        noverlap=args.welch_noverlap,
    )
    targets = detect_targets(
        freq,
        raw_psd_db,
        prominence_db=float(args.peak_prominence_db),
        max_peak_width_hz=float(args.max_peak_width_hz),
        min_frequency_hz=float(args.min_frequency_hz),
        max_frequency_hz=max_frequency,
        notch_width_hz=args.notch_width_hz,
        notch_width_scale=float(args.notch_width_scale),
        min_notch_width_hz=float(args.min_notch_width_hz),
        max_notch_width_hz=float(args.max_notch_width_hz),
        nyquist_hz=nyquist,
    )
    filtered_values = apply_bandstop_filters(
        raw_values,
        targets,
        fs_hz=fs_hz,
        order=int(args.filter_order),
    )
    _, filtered_psd_db = compute_psd_db(
        filtered_values,
        fs_hz=fs_hz,
        nperseg=int(args.welch_nperseg),
        noverlap=args.welch_noverlap,
    )

    payload = {key: value for key, value in data.items() if not key.startswith("__")}
    dff_out = np.array(dff, copy=True)
    dff_out[epoch_idx] = filtered_values.astype(dff_out.dtype, copy=False)
    payload["dff"] = dff_out
    payload["psd_notch_epoch"] = np.array([[args.epoch]], dtype=np.int32)
    payload["psd_notch_frequencies_hz"] = np.array([target.frequency_hz for target in targets], dtype=np.float64).reshape(1, -1)
    payload["psd_notch_prominence_db"] = np.array([target.prominence_db for target in targets], dtype=np.float64).reshape(1, -1)
    payload["psd_notch_width_hz"] = np.array([target.width_hz for target in targets], dtype=np.float64).reshape(1, -1)
    payload["psd_notch_bands_hz"] = np.array(
        [[target.band_low_hz, target.band_high_hz] for target in targets],
        dtype=np.float64,
    )

    output_mat.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_mat, payload, do_compression=True)

    psd_path = plot_dir / f"{output_mat.stem}_epoch{args.epoch}_psd.png"
    time_path = plot_dir / f"{output_mat.stem}_epoch{args.epoch}_timeseries.png"
    csv_path = plot_dir / f"{output_mat.stem}_epoch{args.epoch}_peaks.csv"
    _plot_psd(psd_path, freq, raw_psd_db, filtered_psd_db, targets, xlim=tuple(args.psd_xlim))
    _plot_timeseries(time_path, times, raw_values, filtered_values, xlim=tuple(args.xlim))
    _write_peaks_csv(csv_path, targets)

    print(f"[done] input={input_mat}")
    print(f"[done] output_mat={output_mat}")
    print(f"[done] plot_dir={plot_dir}")
    print(f"[done] epoch={args.epoch} fs_hz={fs_hz:.6g} targets={len(targets)}")
    for target in targets:
        print(
            "[target] "
            f"freq={target.frequency_hz:.6g}Hz "
            f"prom={target.prominence_db:.3g}dB "
            f"peak_width={target.width_hz:.3g}Hz "
            f"band=[{target.band_low_hz:.6g},{target.band_high_hz:.6g}]Hz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
