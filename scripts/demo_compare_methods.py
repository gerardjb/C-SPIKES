#!/usr/bin/env python3
"""
Demo: run PGAS, ENS2, and CASCADE on a user-specified dataset and compare outputs.

Features:
  - Optional smoothing/downsampling (set a target Hz or use native rate).
  - Optional PGAS/CASCADE resample overrides.
  - Optional trimming via edges file or start/end times.
  - Prints correlations and shows overlay plots (spike_prob + discrete spikes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from c_spikes.inference.workflow import (
    DatasetRunConfig,
    MethodSelection,
    SmoothingLevel,
    run_inference_for_dataset,
)
from c_spikes.inference.eval import build_ground_truth_series, compute_correlations
from c_spikes.inference.pgas import pgas_windows_from_result
from c_spikes.utils import load_Janelia_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to .mat file with time_stamps and dff.")
    parser.add_argument("--smoothing", type=float, default=None, help="Target Hz for pre-inference smoothing (None=raw).")
    parser.add_argument("--pgas-resample", type=float, default=None, help="PGAS resample Hz (None=use native).")
    parser.add_argument("--cascade-resample", type=float, default=None, help="CASCADE resample Hz (default model: 30).")
    parser.add_argument("--edges-file", type=Path, help="Optional edges npy (dict dataset->edges) for trimming.")
    parser.add_argument("--start-time", type=float, help="Manual trim start (sec).")
    parser.add_argument("--end-time", type=float, help="Manual trim end (sec).")
    parser.add_argument("--skip-pgas", action="store_true", help="Skip PGAS.")
    parser.add_argument("--skip-ens2", action="store_true", help="Skip ENS2.")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip CASCADE.")
    parser.add_argument("--plot", action="store_true", help="Show overlay plots.")
    return parser.parse_args()


def plot_overlay(methods, title: str, xlim: Optional[tuple[float, float]] = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, m in enumerate(methods):
        c = colors[idx % len(colors)]
        ax.plot(m.time_stamps, m.spike_prob + idx * 0.5, label=f"{m.name} spike_prob", color=c, alpha=0.8)
        if m.discrete_spikes is not None and m.discrete_spikes.size == m.time_stamps.size:
            mask = m.discrete_spikes > 0
            if np.any(mask):
                ax.vlines(
                    m.time_stamps[mask],
                    ymin=np.nanmin(m.spike_prob) + idx * 0.5,
                    ymax=np.nanmax(m.spike_prob) + idx * 0.5,
                    color=c,
                    alpha=0.2,
                    linewidth=0.8,
                )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike prob (offset per method)")
    ax.legend()
    if xlim:
        ax.set_xlim(*xlim)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    time_stamps, dff, spike_times = load_Janelia_data(str(args.dataset))
    dataset_tag = args.dataset.stem

    edges = None
    if args.start_time is not None and args.end_time is not None:
        if args.end_time <= args.start_time:
            raise ValueError("end-time must exceed start-time.")
        edges = np.array([[args.start_time, args.end_time]] * time_stamps.shape[0], dtype=float)
    elif args.edges_file and args.edges_file.exists():
        edges_lookup = np.load(args.edges_file, allow_pickle=True).item()
        if dataset_tag in edges_lookup:
            edges = np.asarray(edges_lookup[dataset_tag], dtype=float)
        else:
            print(f"[WARN] Dataset '{dataset_tag}' not in edges file {args.edges_file}; skipping trim.")

    smoothing = SmoothingLevel(target_fs=args.smoothing)
    selection = MethodSelection(
        run_pgas=not args.skip_pgas,
        run_ens2=not args.skip_ens2,
        run_cascade=not args.skip_cascade,
    )
    cfg = DatasetRunConfig(
        dataset_path=args.dataset,
        smoothing=smoothing,
        reference_fs=None,
        edges=edges,
        selection=selection,
        use_cache=True,
        bm_sigma_gap_s=0.15,
        pgas_resample_fs=args.pgas_resample,
        cascade_resample_fs=args.cascade_resample,
    )

    outputs = run_inference_for_dataset(
        cfg,
        pgas_constants=Path("parameter_files/constants_GCaMP8_soma.json"),
        pgas_gparam=Path("src/c_spikes/pgas/20230525_gold.dat"),
        pgas_output_root=Path("results/pgas_output/demo"),
        ens2_pretrained_root=Path("results/Pretrained_models/ens2_published"),
        cascade_model_root=Path("results/Pretrained_models"),
    )

    methods = outputs["methods"]
    print("Methods run:", list(methods.keys()))

    # Correlations (if spike_times provided)
    if spike_times.size > 0 and methods:
        ref_fs = args.smoothing if args.smoothing is not None else 1.0 / np.median(np.diff(outputs["raw_time"]))
        global_start = min(outputs["raw_time"])
        global_end = max(outputs["raw_time"])
        ref_time, ref_trace = build_ground_truth_series(spike_times, global_start, global_end, reference_fs=ref_fs)
        pgas_windows = pgas_windows_from_result(methods["pgas"]) if "pgas" in methods else None
        corr = compute_correlations(methods.values(), ref_time, ref_trace, windows=pgas_windows)
        print("Correlations vs GT:", corr)
    else:
        print("No spike_times provided; skipping correlation.")

    if args.plot and methods:
        plot_overlay(list(methods.values()), title=f"{dataset_tag}: method comparison")


if __name__ == "__main__":
    main()

