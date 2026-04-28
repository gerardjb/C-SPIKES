#!/usr/bin/env python3
"""
Profile PGAS inference runtime across backends (e.g., Kokkos vs CPU) on a data snippet
and plot fold-acceleration on a log scale.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from c_spikes.inference.pgas import (
    PGAS_BM_SIGMA_DEFAULT,
    PGAS_BURNIN,
    PGAS_NITER,
    PgasConfig,
    run_pgas_inference,
)
from c_spikes.inference.types import TrialSeries
from c_spikes.utils import load_Janelia_data


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"none", "null", "auto", "estimate", "estimated"}:
        return None
    return float(value)


def _parse_backend_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, module = spec.split("=", 1)
    else:
        name, module = spec, "c_spikes.pgas.pgas_bound"
    name = name.strip()
    module = module.strip()
    if not name:
        raise ValueError(f"Invalid backend spec '{spec}': missing name.")
    if not module:
        raise ValueError(f"Invalid backend spec '{spec}': missing module.")
    return name, module


def _extract_spike_times(ap_times: np.ndarray, trial_index: int) -> np.ndarray:
    arr = np.asarray(ap_times)
    if arr.dtype == object:
        arr = arr.squeeze()
        if arr.ndim == 0:
            spikes = np.asarray(arr.item(), dtype=float).ravel()
        else:
            if trial_index >= arr.size:
                raise ValueError(f"trial_index {trial_index} out of range for ap_times shape {arr.shape}.")
            spikes = np.asarray(arr[trial_index], dtype=float).ravel()
    else:
        spikes = arr.astype(float).ravel()
    return spikes


def _select_snippet(
    time_stamps: np.ndarray,
    dff: np.ndarray,
    *,
    trial_index: int,
    start_s: float | None,
    duration_s: float | None,
    end_s: float | None,
    log: Callable[[str], None] | None = None,
) -> tuple[TrialSeries, tuple[float, float]]:
    if log is None:
        log = print
    if trial_index < 0 or trial_index >= time_stamps.shape[0]:
        raise ValueError(f"trial_index {trial_index} out of range for {time_stamps.shape[0]} trials.")

    times = np.asarray(time_stamps[trial_index], dtype=float)
    values = np.asarray(dff[trial_index], dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.size < 2:
        raise ValueError(f"Trial {trial_index} has insufficient samples after NaN filtering.")

    t_min = float(times[0])
    t_max = float(times[-1])

    if start_s is None:
        start_s = t_min
    if start_s < t_min:
        log(f"[WARN] snippet start {start_s:.3f}s < trial start {t_min:.3f}s; clamping.")
        start_s = t_min

    if end_s is None:
        if duration_s is None:
            end_s = t_max
        else:
            end_s = start_s + float(duration_s)
    if end_s > t_max:
        log(f"[WARN] snippet end {end_s:.3f}s > trial end {t_max:.3f}s; clamping.")
        end_s = t_max

    if end_s <= start_s:
        raise ValueError(f"Invalid snippet window: start={start_s}, end={end_s}.")

    win_mask = (times >= start_s) & (times <= end_s)
    if np.count_nonzero(win_mask) < 2:
        raise ValueError(
            f"Snippet window [{start_s:.3f}, {end_s:.3f}] has too few samples in trial {trial_index}."
        )

    trial = TrialSeries(times=times[win_mask].copy(), values=values[win_mask].copy())
    return trial, (float(start_s), float(end_s))


@contextlib.contextmanager
def _patch_pgas_module(module) -> Iterable[None]:
    import sys

    key = "c_spikes.pgas.pgas_bound"
    prev = sys.modules.get(key)
    sys.modules[key] = module
    try:
        yield
    finally:
        if prev is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = prev


def _run_backend(
    *,
    backend_name: str,
    module_path: str,
    trial: TrialSeries,
    spike_times: np.ndarray,
    raw_fs: float,
    dataset_tag: str,
    output_root: Path,
    constants_file: Path,
    gparam_file: Path,
    niter: int,
    burnin: int,
    bm_sigma: float | None,
    bm_sigma_gap_s: float,
    resample_fs: float | None,
    downsample_label: str,
    use_cache: bool,
) -> float:
    try:
        pgas_module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Backend '{backend_name}' could not import module '{module_path}'."
        ) from exc

    pgas_cfg = PgasConfig(
        dataset_tag=dataset_tag,
        output_root=output_root,
        constants_file=constants_file,
        gparam_file=gparam_file,
        resample_fs=resample_fs,
        niter=niter,
        burnin=burnin,
        downsample_label=downsample_label,
        maxspikes=None,
        bm_sigma=bm_sigma,
        bm_sigma_gap_s=bm_sigma_gap_s,
        edges=None,
        use_cache=use_cache,
    )

    start = time.perf_counter()
    with _patch_pgas_module(pgas_module):
        _ = run_pgas_inference(
            trials=[trial],
            raw_fs=raw_fs,
            spike_times=spike_times,
            config=pgas_cfg,
        )
    end = time.perf_counter()
    return end - start


def _run_backend_timings_inprocess(
    *,
    backend_name: str,
    module_path: str,
    trial: TrialSeries,
    spike_times: np.ndarray,
    raw_fs: float,
    dataset_tag: str,
    output_root: Path,
    constants_file: Path,
    gparam_file: Path,
    niter: int,
    burnin: int,
    bm_sigma: float | None,
    bm_sigma_gap_s: float,
    resample_fs: float | None,
    downsample_label: str,
    use_cache: bool,
    warmup: int,
    repeats: int,
) -> List[float]:
    if warmup > 0:
        for _ in range(warmup):
            _run_backend(
                backend_name=backend_name,
                module_path=module_path,
                trial=trial,
                spike_times=spike_times,
                raw_fs=raw_fs,
                dataset_tag=dataset_tag,
                output_root=output_root,
                constants_file=constants_file,
                gparam_file=gparam_file,
                niter=niter,
                burnin=burnin,
                bm_sigma=bm_sigma,
                bm_sigma_gap_s=bm_sigma_gap_s,
                resample_fs=resample_fs,
                downsample_label=downsample_label,
                use_cache=use_cache,
            )

    backend_times: List[float] = []
    for _ in range(repeats):
        elapsed = _run_backend(
            backend_name=backend_name,
            module_path=module_path,
            trial=trial,
            spike_times=spike_times,
            raw_fs=raw_fs,
            dataset_tag=dataset_tag,
            output_root=output_root,
            constants_file=constants_file,
            gparam_file=gparam_file,
            niter=niter,
            burnin=burnin,
            bm_sigma=bm_sigma,
            bm_sigma_gap_s=bm_sigma_gap_s,
            resample_fs=resample_fs,
            downsample_label=downsample_label,
            use_cache=use_cache,
        )
        backend_times.append(elapsed)
    return backend_times


def _run_backend_subprocess(
    *,
    backend_spec: str,
    args: argparse.Namespace,
) -> List[float]:
    import subprocess
    import sys

    run_name = args.run_name or "pgas_profile"
    run_root = args.pgas_output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    backend_name, _ = _parse_backend_spec(backend_spec)
    worker_out = run_root / f"worker_{backend_name}.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend_spec,
        "--worker-output",
        str(worker_out),
        "--dataset",
        str(args.dataset),
        "--trial-index",
        str(args.trial_index),
        "--snippet-duration-s",
        str(args.snippet_duration_s),
        "--pgas-constants",
        str(args.pgas_constants),
        "--pgas-gparam",
        str(args.pgas_gparam),
        "--pgas-output-root",
        str(args.pgas_output_root),
        "--pgas-bm-sigma",
        str(args.pgas_bm_sigma),
        "--niter",
        str(args.niter),
        "--bm-sigma-gap-s",
        str(args.bm_sigma_gap_s),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--run-name",
        str(args.run_name),
    ]
    if args.snippet_start_s is not None:
        cmd.extend(["--snippet-start-s", str(args.snippet_start_s)])
    if args.snippet_end_s is not None:
        cmd.extend(["--snippet-end-s", str(args.snippet_end_s)])
    if args.pgas_resample is not None:
        cmd.extend(["--pgas-resample", str(args.pgas_resample)])
    if args.use_cache:
        cmd.append("--use-cache")

    subprocess.run(cmd, check=True)
    if not worker_out.exists():
        raise RuntimeError(f"Worker did not write output file: {worker_out}")
    with worker_out.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["timings"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to .mat dataset (required unless --reuse-results).",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
        help="Trial/epoch index to profile (0-based).",
    )
    parser.add_argument("--snippet-start-s", type=float, default=None, help="Snippet start time (sec).")
    parser.add_argument(
        "--snippet-duration-s",
        type=float,
        default=5.0,
        help="Snippet duration (sec). Default: 5s.",
    )
    parser.add_argument("--snippet-end-s", type=float, default=None, help="Snippet end time (sec).")
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help=(
            "Backend spec as 'name=module'. Can repeat. "
            "If module omitted, defaults to c_spikes.pgas.pgas_bound."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Backend name to use as baseline for speedup (default: cpu if present else first backend).",
    )
    parser.add_argument(
        "--pgas-constants",
        type=Path,
        default=Path("parameter_files/constants_GCaMP8_soma.json"),
        help="PGAS constants JSON.",
    )
    parser.add_argument(
        "--pgas-gparam",
        type=Path,
        default=Path("src/c_spikes/pgas/20230525_gold.dat"),
        help="PGAS GCaMP parameter file.",
    )
    parser.add_argument(
        "--pgas-output-root",
        type=Path,
        default=Path("results/pgas_profile"),
        help="Output root for PGAS run files + plots.",
    )
    parser.add_argument(
        "--pgas-bm-sigma",
        type=str,
        default=str(PGAS_BM_SIGMA_DEFAULT),
        help="Fixed bm_sigma value, or 'auto' to estimate.",
    )
    parser.add_argument("--pgas-resample", type=float, default=None, help="PGAS resample Hz (None=raw).")
    parser.add_argument("--niter", type=int, default=PGAS_NITER, help="PGAS niter.")
    parser.add_argument(
        "--bm-sigma-gap-s",
        type=float,
        default=0.15,
        help="Gap (s) around spikes excluded when estimating bm_sigma.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats per backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per backend (not timed).")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow cache loads (default: disabled for timing).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name (default: timestamp-based).",
    )
    parser.add_argument(
        "--backend-mode",
        type=str,
        default="auto",
        choices=("auto", "inprocess", "subprocess"),
        help=(
            "Backend isolation mode. 'auto' uses subprocess when multiple backends are requested; "
            "'subprocess' always isolates; 'inprocess' runs in the current process."
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Reuse timings from an existing timings.json instead of rerunning inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
        return
    if args.reuse_results:
        if args.run_name is None:
            raise ValueError("--reuse-results requires --run-name to locate an existing timings.json.")
        run_root = args.pgas_output_root / args.run_name
        timings_path = run_root / "timings.json"
        if not timings_path.exists():
            raise FileNotFoundError(timings_path)
        with timings_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        timings = {k: list(v) for k, v in payload.get("timings", {}).items()}
        if not timings:
            raise RuntimeError(f"No timings found in {timings_path}.")
        baseline = payload.get("baseline")
        if not baseline:
            names = list(timings.keys())
            baseline = "cpu" if "cpu" in names else names[0]

        summary_rows = []
        for name, values in timings.items():
            data = np.asarray(values, dtype=float)
            summary_rows.append(
                {
                    "backend": name,
                    "n_runs": int(data.size),
                    "mean_s": float(np.mean(data)),
                    "median_s": float(np.median(data)),
                    "min_s": float(np.min(data)),
                    "max_s": float(np.max(data)),
                    "std_s": float(np.std(data)) if data.size > 1 else 0.0,
                }
            )
        baseline_row = next((row for row in summary_rows if row["backend"] == baseline), None)
        if baseline_row is None:
            raise ValueError(f"Baseline '{baseline}' not found in {list(timings)}.")
        baseline_mean = float(baseline_row["mean_s"])
        for row in summary_rows:
            mean_s = float(row["mean_s"])
            row["speedup_vs_baseline"] = float(baseline_mean / mean_s) if mean_s > 0 else float("nan")

        _write_summary_and_plot(
            run_root=run_root,
            summary_rows=summary_rows,
            timings=timings,
            baseline=baseline,
        )
        print("[DONE] Wrote timings to", run_root)
        print("[DONE] Plot:", run_root / "runtime_dotplot.png")
        return

    if args.dataset is None:
        raise ValueError("--dataset is required unless --reuse-results is set.")
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    time_stamps, dff, ap_times = load_Janelia_data(str(args.dataset))
    trial, (start_s, end_s) = _select_snippet(
        time_stamps,
        dff,
        trial_index=args.trial_index,
        start_s=args.snippet_start_s,
        duration_s=args.snippet_duration_s,
        end_s=args.snippet_end_s,
    )

    spike_times = _extract_spike_times(ap_times, args.trial_index)
    spike_times = spike_times[(spike_times >= start_s) & (spike_times <= end_s)]

    raw_fs = trial.current_fs()
    dataset_stem = args.dataset.stem
    snippet_label = f"snip{end_s - start_s:.2f}s"

    backend_specs = args.backend or ["kokkos=c_spikes.pgas.pgas_bound"]
    backends: List[Tuple[str, str]] = [_parse_backend_spec(spec) for spec in backend_specs]

    baseline = args.baseline
    if baseline is None:
        names = [name for name, _ in backends]
        baseline = "cpu" if "cpu" in names else names[0]

    run_name = args.run_name or f"{dataset_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.run_name = run_name
    run_root = args.pgas_output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    bm_sigma = _parse_optional_float(args.pgas_bm_sigma)

    timings: Dict[str, List[float]] = {}
    use_subprocess = (
        args.backend_mode == "subprocess"
        or (args.backend_mode == "auto" and len(backends) > 1)
    )
    if use_subprocess:
        for name, module in backends:
            backend_spec = f"{name}={module}"
            timings[name] = _run_backend_subprocess(backend_spec=backend_spec, args=args)
    else:
        for name, module in backends:
            backend_root = run_root / name
            backend_root.mkdir(parents=True, exist_ok=True)
            dataset_tag = f"{dataset_stem}_{name}"
            timings[name] = _run_backend_timings_inprocess(
                backend_name=name,
                module_path=module,
                trial=trial,
                spike_times=spike_times,
                raw_fs=raw_fs,
                dataset_tag=dataset_tag,
                output_root=backend_root,
                constants_file=args.pgas_constants,
                gparam_file=args.pgas_gparam,
                niter=args.niter,
                burnin=PGAS_BURNIN,
                bm_sigma=bm_sigma,
                bm_sigma_gap_s=args.bm_sigma_gap_s,
                resample_fs=args.pgas_resample,
                downsample_label=snippet_label,
                use_cache=args.use_cache,
                warmup=args.warmup,
                repeats=args.repeats,
            )

    summary_rows = []
    for name, _ in backends:
        data = np.asarray(timings[name], dtype=float)
        summary_rows.append(
            {
                "backend": name,
                "n_runs": int(data.size),
                "mean_s": float(np.mean(data)),
                "median_s": float(np.median(data)),
                "min_s": float(np.min(data)),
                "max_s": float(np.max(data)),
                "std_s": float(np.std(data)) if data.size > 1 else 0.0,
            }
        )

    baseline_row = next((row for row in summary_rows if row["backend"] == baseline), None)
    if baseline_row is None:
        raise ValueError(f"Baseline '{baseline}' not found in backends {list(timings)}.")
    baseline_mean = float(baseline_row["mean_s"])

    for row in summary_rows:
        mean_s = float(row["mean_s"])
        row["speedup_vs_baseline"] = float(baseline_mean / mean_s) if mean_s > 0 else float("nan")

    payload = {
        "dataset": str(args.dataset),
        "trial_index": int(args.trial_index),
        "snippet_start_s": float(start_s),
        "snippet_end_s": float(end_s),
        "raw_fs": float(raw_fs),
        "pgas_constants": str(args.pgas_constants),
        "pgas_gparam": str(args.pgas_gparam),
        "pgas_resample": args.pgas_resample,
        "niter": int(args.niter),
        "burnin": int(PGAS_BURNIN),
        "bm_sigma": bm_sigma,
        "bm_sigma_gap_s": float(args.bm_sigma_gap_s),
        "baseline": baseline,
        "timings": timings,
        "summary": summary_rows,
    }

    with (run_root / "timings.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    _write_summary_and_plot(
        run_root=run_root,
        summary_rows=summary_rows,
        timings=timings,
        baseline=baseline,
    )

    print("[DONE] Wrote timings to", run_root)
    print("[DONE] Plot:", run_root / "runtime_dotplot.png")


def _write_summary_and_plot(
    *,
    run_root: Path,
    summary_rows: List[Dict[str, object]],
    timings: Dict[str, List[float]],
    baseline: str,
) -> None:
    summary_csv = run_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8") as fh:
        header = ["backend", "n_runs", "mean_s", "median_s", "min_s", "max_s", "std_s", "speedup_vs_baseline"]
        fh.write(",".join(header) + "\n")
        for row in summary_rows:
            fh.write(",".join(str(row.get(col, "")) for col in header) + "\n")

    mpl_dir = run_root / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["backend"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    max_time = 0.0
    for idx, label in enumerate(labels):
        times = np.asarray(timings.get(label, []), dtype=float)
        if times.size == 0:
            continue
        x = np.full(times.shape, idx, dtype=float)
        ax.scatter(x, times, s=35, alpha=0.75, color="tab:blue")
        mean_val = float(np.mean(times))
        max_time = max(max_time, float(np.max(times)))
        ax.hlines(mean_val, idx - 0.25, idx + 0.25, colors="k", linewidth=2)
        ax.text(idx, mean_val, f"{mean_val:.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Time per run (s)\n5 s snippet, 10 iterations")
    ax.set_title("PGAS Backend Runtime per Run")
    ax.set_xlabel("Backend")

    # Annotate GPU speedup if present.
    if "gpu" in labels and baseline in labels:
        baseline_mean = next(
            (row["mean_s"] for row in summary_rows if row["backend"] == baseline),
            None,
        )
        gpu_mean = next((row["mean_s"] for row in summary_rows if row["backend"] == "gpu"), None)
        if baseline_mean and gpu_mean and gpu_mean > 0:
            speedup = float(baseline_mean) / float(gpu_mean)
            gpu_idx = labels.index("gpu")
            pad = max(0.5 * max_time, 0.5)
            y_pos = max_time - pad
            ax.text(gpu_idx, y_pos, f"speedup\n={speedup:.2f}x", ha="center", va="bottom", fontsize=10)
            #ax.set_ylim(0, y_pos + pad)

    ax.set_yscale("log")
    fig.tight_layout()
    plot_path = run_root / "runtime_dotplot.png"
    fig.savefig(plot_path, dpi=200)


def _worker_main(args: argparse.Namespace) -> None:
    import sys

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    backend_specs = args.backend or ["kokkos=c_spikes.pgas.pgas_bound"]
    backends: List[Tuple[str, str]] = [_parse_backend_spec(spec) for spec in backend_specs]
    if len(backends) != 1:
        raise ValueError("Worker mode requires exactly one --backend entry.")

    time_stamps, dff, ap_times = load_Janelia_data(str(args.dataset))
    trial, (start_s, end_s) = _select_snippet(
        time_stamps,
        dff,
        trial_index=args.trial_index,
        start_s=args.snippet_start_s,
        duration_s=args.snippet_duration_s,
        end_s=args.snippet_end_s,
        log=lambda msg: print(msg, file=sys.stderr),
    )
    spike_times = _extract_spike_times(ap_times, args.trial_index)
    spike_times = spike_times[(spike_times >= start_s) & (spike_times <= end_s)]
    raw_fs = trial.current_fs()

    dataset_stem = args.dataset.stem
    snippet_label = f"snip{end_s - start_s:.2f}s"
    run_name = args.run_name or f"{dataset_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.pgas_output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    bm_sigma = _parse_optional_float(args.pgas_bm_sigma)
    name, module = backends[0]
    backend_root = run_root / name
    backend_root.mkdir(parents=True, exist_ok=True)
    dataset_tag = f"{dataset_stem}_{name}"

    timings = _run_backend_timings_inprocess(
        backend_name=name,
        module_path=module,
        trial=trial,
        spike_times=spike_times,
        raw_fs=raw_fs,
        dataset_tag=dataset_tag,
        output_root=backend_root,
        constants_file=args.pgas_constants,
        gparam_file=args.pgas_gparam,
        niter=args.niter,
        burnin=PGAS_BURNIN,
        bm_sigma=bm_sigma,
        bm_sigma_gap_s=args.bm_sigma_gap_s,
        resample_fs=args.pgas_resample,
        downsample_label=snippet_label,
        use_cache=args.use_cache,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    payload = {"backend": name, "timings": timings}
    if args.worker_output is not None:
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        with args.worker_output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
