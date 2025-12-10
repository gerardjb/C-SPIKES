#!/usr/bin/env python3
"""
Utility to compare correlation summaries across multiple inference runs.

It scans `results/full_evaluation/<run_tag>/<dataset>/<smoothing>/summary.json`
and prints the correlation values for each requested run tag so different PGAS/CASCADE
configurations can be compared side-by-side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/full_evaluation_by_run"),
        help="Root directory containing run_tag/dataset/smoothing summaries.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="TAG",
        help="Restrict comparison to specific dataset tag(s). Repeat to add more.",
    )
    parser.add_argument(
        "--smoothing",
        action="append",
        metavar="LABEL",
        help="Restrict to specific smoothing labels (e.g., raw, 30Hz, 10Hz).",
    )
    parser.add_argument(
        "--run",
        action="append",
        metavar="TAG",
        help="Only include specific run tags (e.g., pgas120_cascade30 or custom labels).",
    )
    parser.add_argument(
        "--methods",
        action="append",
        metavar="NAME",
        help="Limit output to particular correlation keys (e.g., pgas, ens2).",
    )
    parser.add_argument(
        "--as-csv",
        action="store_true",
        help="Emit CSV rows instead of a formatted table.",
    )
    parser.add_argument(
        "--missing",
        choices=["skip", "warn"],
        default="warn",
        help="How to handle missing summary.json files (default: warn).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a dotplot with per-method correlations, median, and IQR indicators.",
    )
    parser.add_argument(
        "--plot-save",
        type=Path,
        help="If provided, save the plot to this path instead of/min addition to showing it.",
    )
    parser.add_argument(
        "--show-methods",
        action="store_true",
        help="Also print method entries from comparison manifests when available.",
    )
    return parser.parse_args(argv)


def normalize_filter(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not values:
        return None
    seen = []
    for val in values:
        token = val.strip()
        if token and token not in seen:
            seen.append(token)
    return seen or None


def load_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_entries(
    root: Path,
    dataset_filter: Optional[List[str]],
    smoothing_filter: Optional[List[str]],
    run_filter: Optional[List[str]],
    method_filter: Optional[List[str]],
    on_missing: str,
    include_methods: bool,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not root.exists():
        raise FileNotFoundError(f"Summary root '{root}' does not exist.")

    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        run_tag = run_dir.name
        if run_filter and run_tag not in run_filter:
            continue
        dataset_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
        for dataset_dir in dataset_dirs:
            dataset_tag = dataset_dir.name
            if dataset_filter and dataset_tag not in dataset_filter:
                continue
            smoothing_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
            for smoothing_dir in smoothing_dirs:
                smoothing_label = smoothing_dir.name
                if smoothing_filter and smoothing_label not in smoothing_filter:
                    continue

                summary_path = smoothing_dir / "summary.json"
                summary = load_summary(summary_path)
                if summary is None:
                    if on_missing == "warn":
                        print(f"[WARN] Missing summary: {summary_path}")
                    continue
                correlations = summary.get("correlations", {})
                if not isinstance(correlations, dict):
                    continue
                filtered_corr = (
                    {k: correlations.get(k) for k in method_filter}
                    if method_filter
                    else correlations
                )
                entry: Dict[str, object] = {
                    "dataset": dataset_tag,
                    "smoothing": smoothing_label,
                    "run": run_tag,
                    "correlations": filtered_corr,
                }
                if include_methods:
                    manifest_path = smoothing_dir / "comparison.json"
                    if manifest_path.exists():
                        try:
                            with manifest_path.open("r", encoding="utf-8") as mf:
                                entry["manifest"] = json.load(mf)
                        except Exception:
                            entry["manifest"] = {"error": f"Failed to load {manifest_path}"}
                entries.append(entry)
    return entries


def print_table(entries: List[Dict[str, object]], show_methods: bool) -> None:
    if not entries:
        print("No matching summaries found.")
        return

    for entry in entries:
        print(f"\nDataset: {entry['dataset']}")
        print(f"Smoothing: {entry['smoothing']}")
        print(f"Run tag: {entry['run']}")
        for method, value in entry["correlations"].items():
            if value is None:
                continue
            print(f"  {method:>8}: {value:.4f}")
        if show_methods and entry.get("manifest"):
            methods = entry["manifest"].get("methods", [])
            if methods:
                print("  Methods:")
                for m in methods:
                    label = m.get("label", m.get("method", ""))
                    tag = m.get("cache_tag", "")
                    print(f"    - {label} ({m.get('method','')}): {tag}")


def print_csv(entries: List[Dict[str, object]]) -> None:
    if not entries:
        print("dataset,smoothing,run,method,correlation")
        return
    print("dataset,smoothing,run,method,correlation")
    for entry in entries:
        dataset = entry["dataset"]
        smoothing = entry["smoothing"]
        run_tag = entry["run"]
        for method, value in entry["correlations"].items():
            if value is None:
                continue
            print(f"{dataset},{smoothing},{run_tag},{method},{value}")


def build_plot_data(
    entries: List[Dict[str, object]]
) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """
    Group correlation values by (run_tag, method) pairing while tracking smoothing labels.
    """
    data: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for entry in entries:
        run_tag = entry["run"]
        smoothing = entry["smoothing"]
        for method, value in entry["correlations"].items():
            if value is None:
                continue
            key = (run_tag, method)
            data.setdefault(key, []).append((smoothing, float(value)))
    return data


def unique_smoothing_labels(entries: List[Dict[str, object]]) -> List[str]:
    labels: List[str] = []
    for entry in entries:
        label = entry["smoothing"]
        if label not in labels:
            labels.append(label)
    return labels


def plot_dotplot(
    plot_data: Dict[Tuple[str, str], List[Tuple[str, float]]],
    title: str,
    save_path: Optional[Path],
    show_plot: bool,
    smoothing_order: Sequence[str],
) -> None:
    if not plot_data:
        print("No data available for plotting.")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        print(f"Plotting requested but matplotlib/numpy not available: {exc}")
        return

    groups = sorted(plot_data.keys())
    smoothing_map = {label: idx for idx, label in enumerate(smoothing_order)}
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.2), 4))
    offset_step = 0.2
    for idx, (run_tag, method) in enumerate(groups):
        observations = plot_data[(run_tag, method)]
        if not observations:
            continue
        color = ax._get_lines.get_next_color()  # type: ignore[attr-defined]
        smoothing_bucket: Dict[str, List[float]] = {}
        for smooth_label, val in observations:
            smoothing_bucket.setdefault(smooth_label, []).append(val)
        for smooth_label in smoothing_order:
            values = np.asarray(smoothing_bucket.get(smooth_label, []), dtype=float)
            if values.size == 0:
                continue
            bucket_idx = smoothing_map.get(smooth_label, 0)
            base_x = idx + (bucket_idx - (len(smoothing_order) - 1) / 2.0) * offset_step
            jitter = (np.random.rand(values.size) - 0.5) * offset_step * 0.6
            ax.scatter(
                np.full(values.shape, base_x) + jitter,
                values,
                color=color,
                alpha=0.7,
                s=30,
            )
            median = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            ax.hlines(
                median,
                base_x - offset_step * 0.3,
                base_x + offset_step * 0.3,
                colors="black",
                linewidth=2,
                zorder=5,
            )
            ax.vlines(base_x, q1, q3, colors=color, linewidth=2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{method}-{run}" for run, method in groups], rotation=45, ha="right")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_filter = normalize_filter(args.dataset)
    smoothing_filter = normalize_filter(args.smoothing)
    run_filter = normalize_filter(args.run)
    method_filter = normalize_filter(args.methods)

    entries = collect_entries(
        root=args.root,
        dataset_filter=dataset_filter,
        smoothing_filter=smoothing_filter,
        run_filter=run_filter,
        method_filter=method_filter,
        on_missing=args.missing,
        include_methods=args.show_methods,
    )
    plotting_requested = args.plot or args.plot_save
    if plotting_requested:
        plot_data = build_plot_data(entries)
        filters = []
        if dataset_filter:
            filters.append(f"datasets={','.join(dataset_filter)}")
        if smoothing_filter:
            filters.append(f"smoothing={','.join(smoothing_filter)}")
        if run_filter:
            filters.append(f"runs={','.join(run_filter)}")
        title = "Correlation comparison"
        if filters:
            title += " (" + "; ".join(filters) + ")"
        smoothing_order = unique_smoothing_labels(entries)
        plot_dotplot(
            plot_data,
            title=title,
            save_path=args.plot_save,
            show_plot=args.plot,
            smoothing_order=smoothing_order,
        )
    else:
        if args.as_csv:
            print_csv(entries)
        else:
            print_table(entries, args.show_methods)


if __name__ == "__main__":
    main()
