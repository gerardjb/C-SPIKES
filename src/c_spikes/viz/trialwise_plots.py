from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio


DEFAULT_COLORS: Dict[str, str] = {
    "pgas": "#009E73",
    "cascade": "#F3AE14",
    "ens2": "#A6780C",
}

DEFAULT_LABELS: Dict[str, str] = {
    "pgas": r"Biophys$_{SMC}$",
    "cascade": "CASCADE",
    "ens2": r"ENS$^2$",
    "mlspike": "MLspike",
    "biophys_ml": r"Biophys$_{ML}$",
}


def ensure_matplotlib_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """
    Ensure matplotlib uses a writable cache/config directory (common on HPC).

    Call this *before importing matplotlib* for best results.
    """
    if cache_dir is None:
        cache_dir = Path.cwd() / "tmp" / "mpl_cache"
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    return cache_dir


def read_trialwise_csv(path: Path) -> List[Dict[str, str]]:
    path = path.expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def _uniq(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for v in values:
        t = str(v).strip()
        if t and t not in out:
            out.append(t)
    return out or None


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _mean_sem(values: Sequence[float]) -> Tuple[float, float, int]:
    arr = _finite(values)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, n
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, sem, n


def _coerce_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _y_at_x(xs: np.ndarray, ys: np.ndarray, x_target: float) -> float:
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(ys[0])
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.interp(float(x_target), xs, ys))


def _place_right_labels(
    ax: Any,
    fig: Any,
    *,
    x: float,
    y_positions: List[float],
    labels: List[str],
    colors: List[str],
    min_sep: float,
    fontsize: float = 14,
) -> None:
    # Respect input ordering (caller can rank-order labels); apply offsets to avoid overlap.
    ys = [float(y) for y in y_positions]

    for i in range(1, len(ys)):
        if not np.isfinite(ys[i - 1]) or not np.isfinite(ys[i]):
            continue
        if ys[i - 1] - ys[i] < min_sep:
            ys[i] = ys[i - 1] - min_sep

    texts = []
    for y, lab, col in zip(ys, labels, colors):
        texts.append(
            ax.text(
                x,
                y,
                lab,
                color=col,
                fontsize=fontsize,
                fontweight="bold",
                ha="left",
                va="center",
                clip_on=False,
            )
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad_px = 2.0
    prev_bbox = None
    for text in texts:
        bbox = text.get_window_extent(renderer=renderer).expanded(1.0, 1.08)
        while prev_bbox is not None and bbox.overlaps(prev_bbox):
            shift_px = float(bbox.y1 - prev_bbox.y0 + pad_px)
            x_disp, y_disp = ax.transData.transform(text.get_position())
            new_y_disp = y_disp - shift_px
            _, new_y_data = ax.transData.inverted().transform((x_disp, new_y_disp))
            text.set_position((x, float(new_y_data)))
            fig.canvas.draw()
            bbox = text.get_window_extent(renderer=renderer).expanded(1.0, 1.08)
        prev_bbox = bbox

    ax_bbox = ax.get_window_extent(renderer=renderer)
    if texts:
        bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
        top = max(b.y1 for b in bboxes)
        bottom = min(b.y0 for b in bboxes)
        shift_up = 0.0
        if top > ax_bbox.y1:
            shift_up = ax_bbox.y1 - top
        if bottom + shift_up < ax_bbox.y0:
            shift_up = ax_bbox.y0 - bottom
        if shift_up != 0.0:
            for t in texts:
                x_disp, y_disp = ax.transData.transform(t.get_position())
                new_y_disp = y_disp + shift_up
                _, new_y_data = ax.transData.inverted().transform((x_disp, new_y_disp))
                t.set_position((x, float(new_y_data)))


def plot_corr_vs_sigma(
    *,
    csv_path: Path,
    out_path: Optional[Path] = None,
    runs: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    smoothings: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    reduce: str = "dataset",
    title: Optional[str] = None,
    ylabel: str = "Pearson correlation",
    ylim: Tuple[float, float] = (0.2, 1.0),
    figsize: Tuple[float, float] = (7.2, 2.8),
    dpi: int = 200,
    legend: bool = False,
    colors: Optional[Mapping[str, str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    ax: Any = None,
) -> Tuple[Any, Any]:
    """
    Mean ± SEM correlation vs corr_sigma_ms (ms), with right-side labels.

    Returns (fig, ax). If out_path is provided, saves the figure.
    """
    if reduce not in {"dataset", "trial"}:
        raise ValueError("reduce must be 'dataset' or 'trial'")

    # Lazy import so notebooks can configure matplotlib before importing.
    import matplotlib.pyplot as plt

    colors_map = dict(DEFAULT_COLORS)
    if colors:
        colors_map.update({str(k): str(v) for k, v in colors.items()})
    labels_map = dict(DEFAULT_LABELS)
    if labels:
        labels_map.update({str(k): str(v) for k, v in labels.items()})

    rows = read_trialwise_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    run_filter = set(_uniq(runs) or [])
    dataset_filter = set(_uniq(datasets) or [])
    smoothing_filter = set(_uniq(smoothings) or [])
    method_filter = set(_uniq(methods) or [])

    filtered: List[Dict[str, Any]] = []
    for r in rows:
        run = str(r.get("run", "")).strip()
        dataset = str(r.get("dataset", "")).strip()
        smoothing = str(r.get("smoothing", "")).strip()
        method = str(r.get("method", "")).strip()
        if run_filter and run not in run_filter:
            continue
        if dataset_filter and dataset not in dataset_filter:
            continue
        if smoothing_filter and smoothing not in smoothing_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        filtered.append(r)
    if not filtered:
        raise ValueError("No rows matched filters.")

    samples: Dict[Tuple[str, float], List[float]] = {}
    if reduce == "trial":
        for r in filtered:
            method = str(r.get("method", "")).strip()
            sigma = _coerce_float(r, "corr_sigma_ms")
            corr = _coerce_float(r, "correlation")
            if not method or not np.isfinite(sigma):
                continue
            samples.setdefault((method, float(sigma)), []).append(float(corr))
    else:
        by_bucket: Dict[Tuple[str, float, str, str, str], List[float]] = {}
        for r in filtered:
            method = str(r.get("method", "")).strip()
            sigma = _coerce_float(r, "corr_sigma_ms")
            dataset = str(r.get("dataset", "")).strip()
            run = str(r.get("run", "")).strip()
            smoothing = str(r.get("smoothing", "")).strip()
            corr = _coerce_float(r, "correlation")
            if not method or not dataset or not np.isfinite(sigma):
                continue
            by_bucket.setdefault((method, float(sigma), dataset, run, smoothing), []).append(float(corr))
        for (method, sigma, _dataset, _run, _smoothing), values in by_bucket.items():
            mean_val = float(np.mean(_finite(values))) if _finite(values).size else float("nan")
            samples.setdefault((method, float(sigma)), []).append(mean_val)

    method_list = sorted({m for (m, _sigma) in samples.keys()})
    sigma_list = sorted({sigma for (_m, sigma) in samples.keys()})
    if not method_list or not sigma_list:
        raise ValueError("No usable (method, corr_sigma_ms) samples found.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    else:
        fig = ax.figure

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 12,
        }
    )

    method_to_color: Dict[str, str] = {}
    method_to_label: Dict[str, str] = {}
    method_to_y100: Dict[str, float] = {}

    for method in method_list:
        xs: List[float] = []
        ys: List[float] = []
        sems: List[float] = []
        for sigma in sigma_list:
            vals = samples.get((method, sigma), [])
            mean, sem, n = _mean_sem(vals)
            xs.append(float(sigma))
            ys.append(float(mean) if n else float("nan"))
            sems.append(float(sem) if n else float("nan"))

        xs_arr = np.asarray(xs, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.float64)
        sem_arr = np.asarray(sems, dtype=np.float64)

        color = colors_map.get(method)
        if color is None:
            color = plt.cm.tab10(hash(method) % 10)  # type: ignore[arg-type]

        ax.plot(xs_arr, ys_arr, color=color, linewidth=2.5)
        ax.fill_between(xs_arr, ys_arr - sem_arr, ys_arr + sem_arr, color=color, alpha=0.18, linewidth=0)

        method_to_color[method] = str(color)
        method_to_label[method] = labels_map.get(method, method)
        method_to_y100[method] = _y_at_x(xs_arr, ys_arr, 100.0)

    ax.set_xlabel("Filter Width (ms)")
    ax.set_ylabel(str(ylabel))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if title:
        ax.set_title(str(title))

    x_min, x_max = float(min(sigma_list)), float(max(sigma_list))
    x_pad = max(5.0, 0.08 * (x_max - x_min))
    ax.set_xlim(x_min, x_max + 3.0 * x_pad)

    if legend:
        handles = list(ax.get_lines())
        legend_labels = [labels_map.get(m, m) for m in method_list]
        ax.legend(handles, legend_labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        ranked = sorted(
            ((m, method_to_y100.get(m, float("nan"))) for m in method_list),
            key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -1e9), kv[0]),
        )
        right_labels = [method_to_label.get(m, m) for m, _ in ranked]
        right_colors = [method_to_color.get(m, "#000000") for m, _ in ranked]
        right_y = [float(y) for _m, y in ranked]

        y_min, y_max = ax.get_ylim()
        min_sep = 0.09 * (y_max - y_min)
        _place_right_labels(
            ax,
            fig,
            x=x_max + 1.1 * x_pad,
            y_positions=right_y,
            labels=right_labels,
            colors=right_colors,
            min_sep=min_sep,
        )

    fig.tight_layout()
    if out_path is not None:
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig, ax


def _load_edges(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    edges_path = path.expanduser().resolve()
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)
    return np.load(edges_path, allow_pickle=True).item()


def _trial_windows_from_mat(
    dataset_path: Path,
    *,
    edges_lookup: Optional[Dict[str, Any]],
) -> Tuple[List[Tuple[float, float]], float]:
    from c_spikes.utils import load_Janelia_data
    from c_spikes.inference.types import TrialSeries, compute_sampling_rate

    time_stamps, dff, _spike_times = load_Janelia_data(str(dataset_path))
    trials: List[TrialSeries] = []
    for idx in range(time_stamps.shape[0]):
        t = np.asarray(time_stamps[idx], dtype=np.float64)
        y = np.asarray(dff[idx], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        if t.size == 0:
            continue
        trials.append(TrialSeries(times=t, values=y))
    if not trials:
        raise RuntimeError(f"No valid trials for dataset {dataset_path.stem}")

    raw_time = np.concatenate([tr.times for tr in trials])
    raw_fs = float(compute_sampling_rate(raw_time))

    if edges_lookup is not None and dataset_path.stem in edges_lookup:
        edges = np.asarray(edges_lookup[dataset_path.stem], dtype=np.float64)
        windows = [(float(s), float(e)) for s, e in edges]
    else:
        windows = [(float(tr.times[0]), float(tr.times[-1])) for tr in trials]
    return windows, raw_fs


def _reference_fs_from_label(label: str, raw_fs: float) -> float:
    token = str(label).strip().lower()
    if token == "raw":
        return float(raw_fs)
    if token in {"30hz", "30"}:
        return 30.0
    if token in {"10hz", "10"}:
        return 10.0
    if token.endswith("hz"):
        try:
            return float(token[:-2])
        except Exception:
            pass
    return float(raw_fs)


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros_like(v)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(v)
    out = (v - lo) / (hi - lo)
    out[~np.isfinite(out)] = np.nan
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _parse_run_by_method(items: Optional[Sequence[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not items:
        return mapping
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected METHOD=RUN_TAG, got: {item!r}")
        method, run = item.split("=", 1)
        method = method.strip()
        run = run.strip()
        if not method or not run:
            raise ValueError(f"Expected METHOD=RUN_TAG, got: {item!r}")
        mapping[method] = run
    return mapping


@dataclass(frozen=True)
class CacheSpec:
    method: str
    run_tag: str
    dataset: str
    smoothing: str


def _load_comparison_method_entry(eval_root: Path, spec: CacheSpec) -> Dict[str, Any]:
    cmp_path = eval_root / spec.run_tag / spec.dataset / spec.smoothing / "comparison.json"
    if not cmp_path.exists():
        raise FileNotFoundError(f"Missing comparison.json: {cmp_path}")
    obj = json.loads(cmp_path.read_text(encoding="utf-8"))
    for entry in obj.get("methods", []):
        if isinstance(entry, dict) and str(entry.get("method", "")).strip() == spec.method:
            return entry
    raise KeyError(f"Method {spec.method!r} not found in {cmp_path}")


def _cache_paths_from_entry(entry: Dict[str, Any], *, dataset_fallback: str) -> Tuple[str, str]:
    from c_spikes.inference.types import compute_config_signature

    method_name = str(entry.get("method", "")).strip()
    cache_tag_raw = entry.get("cache_tag")
    cache_tag = "" if cache_tag_raw is None else str(cache_tag_raw).strip()
    if cache_tag.lower() == "none":
        cache_tag = ""
    if not cache_tag:
        cache_tag = dataset_fallback

    cache_key_raw = entry.get("cache_key")
    cache_key = str(cache_key_raw).strip() if cache_key_raw is not None else ""
    if not cache_key:
        cfg = entry.get("config", {})
        if isinstance(cfg, dict) and cfg:
            cache_key, _ = compute_config_signature(cfg)
    if not method_name or not cache_key or not cache_tag:
        raise ValueError("Could not resolve method cache path from comparison entry.")
    return cache_tag, cache_key


def _load_method_cache_mat(method: str, cache_tag: str, cache_key: str) -> Any:
    from c_spikes.inference.types import MethodResult, compute_sampling_rate

    mat_path = Path("results/inference_cache") / method / cache_tag / f"{cache_key}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing cache mat: {mat_path}")
    data = sio.loadmat(mat_path)
    time_arr = np.asarray(data.get("time_stamps")).squeeze()
    prob_arr = np.asarray(data.get("spike_prob")).squeeze()
    fs = float(compute_sampling_rate(np.asarray(time_arr, dtype=np.float64).ravel()))
    return MethodResult(name=method, time_stamps=time_arr, spike_prob=prob_arr, sampling_rate=fs, metadata={})


def _segment_slices(times: np.ndarray, fs_est: float, gap_factor: float = 4.0) -> List[slice]:
    times = np.asarray(times, dtype=np.float64).ravel()
    if times.size == 0:
        return []
    diffs = np.diff(times)
    base_dt = 1.0 / float(fs_est)
    threshold = float(gap_factor) * base_dt
    breaks = np.where((diffs > threshold) | ~np.isfinite(diffs))[0] + 1
    idx = np.concatenate([[0], breaks, [times.size]])
    segs: List[slice] = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            segs.append(slice(int(a), int(b)))
    return segs


def plot_trace_panel(
    *,
    csv_path: Path,
    eval_root: Path,
    data_root: Path,
    dataset: str,
    smoothing: str = "raw",
    corr_sigma_ms: float = 50.0,
    display_sigma_ms: Optional[float] = None,
    edges_path: Optional[Path] = None,
    methods: Optional[Sequence[str]] = None,
    run: Optional[str] = None,
    run_by_method: Optional[Sequence[str]] = None,
    trial: Optional[int] = None,
    duration_s: float = 5.0,
    start_s: Optional[float] = None,
    center: str = "median_spike",
    scalebar_time_s: float = 5.0,
    scalebar_dff: float = 0.5,
    title: str = "Excitatory cell sample",
    figsize: Tuple[float, float] = (3.4, 5.2),
    dpi: int = 250,
    colors: Optional[Mapping[str, str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    ax: Any = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Stacked trace panel; returns (fig, ax, meta).

    `run_by_method` is a list of strings like ["pgas=pgasraw", "ens2=cascadein_nodisc_ens2"].
    """
    if methods is None:
        methods = ["pgas", "cascade", "ens2"]
    if center not in {"median_spike", "trial_mid"}:
        raise ValueError("center must be 'median_spike' or 'trial_mid'")

    # Lazy import for notebook friendliness.
    import matplotlib.pyplot as plt

    from c_spikes.inference.eval import build_ground_truth_series, resample_prediction_to_reference
    from c_spikes.model_eval.model_eval import smooth_prediction
    from c_spikes.utils import load_Janelia_data

    colors_map = dict(DEFAULT_COLORS)
    if colors:
        colors_map.update({str(k): str(v) for k, v in colors.items()})
    labels_map = dict(DEFAULT_LABELS)
    if labels:
        labels_map.update({str(k): str(v) for k, v in labels.items()})

    if display_sigma_ms is None:
        display_sigma_ms = float(corr_sigma_ms)

    dataset_stem = str(dataset).strip()
    methods = list(methods)
    run_map = _parse_run_by_method(run_by_method)
    default_run = str(run).strip() if run else "cascadein_nodisc_ens2"

    rows = read_trialwise_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    corr_by_method_trial: Dict[str, Dict[int, float]] = {m: {} for m in methods}
    for r in rows:
        if r.get("dataset") != dataset_stem:
            continue
        if r.get("smoothing") != smoothing:
            continue
        if not np.isclose(_coerce_float(r, "corr_sigma_ms"), float(corr_sigma_ms), atol=1e-6):
            continue
        method = str(r.get("method", "")).strip()
        if method not in corr_by_method_trial:
            continue
        run_tag = str(r.get("run", "")).strip()
        expected_run = run_map.get(method, default_run)
        if run_tag != expected_run:
            continue
        trial_idx = int(float(r.get("trial", "nan")))
        corr_by_method_trial[method][trial_idx] = _coerce_float(r, "correlation")

    missing = [m for m in methods if not corr_by_method_trial.get(m)]
    if missing:
        raise ValueError(
            "Missing trialwise correlations for method(s): "
            + ", ".join(missing)
            + ". Check run/run_by_method and corr_sigma_ms."
        )

    medians: Dict[str, float] = {}
    for method, trial_map in corr_by_method_trial.items():
        vals = _finite(list(trial_map.values()))
        medians[method] = float(np.median(vals)) if vals.size else float("nan")

    common_trials = set.intersection(*(set(m.keys()) for m in corr_by_method_trial.values()))
    if not common_trials:
        raise ValueError("No trial indices are shared across selected methods/run tags.")

    if trial is not None:
        selected_trial = int(trial)
        if selected_trial not in common_trials:
            raise ValueError(f"Requested trial {selected_trial} not available for all selected methods.")
    else:
        best_trial = None
        best_score = None
        for t in sorted(common_trials):
            score = 0.0
            ok = True
            for method in methods:
                corr = corr_by_method_trial[method].get(t, float("nan"))
                if not np.isfinite(corr) or not np.isfinite(medians[method]):
                    ok = False
                    break
                score += abs(float(corr) - float(medians[method]))
            if not ok:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_trial = t
        if best_trial is None:
            raise ValueError("Failed to select a representative trial (insufficient finite correlations).")
        selected_trial = int(best_trial)

    data_root = data_root.expanduser().resolve()
    dataset_path = data_root / f"{dataset_stem}.mat"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    edges_lookup = _load_edges(edges_path)
    trial_windows, raw_fs = _trial_windows_from_mat(dataset_path, edges_lookup=edges_lookup)
    if selected_trial < 0 or selected_trial >= len(trial_windows):
        raise ValueError(f"Trial index {selected_trial} out of range (n_trials={len(trial_windows)}).")
    trial_start, trial_end = trial_windows[selected_trial]

    duration = float(duration_s)
    if duration <= 0:
        raise ValueError("duration_s must be positive.")
    duration = min(duration, float(trial_end - trial_start))

    time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))
    t_trial = np.asarray(time_stamps[selected_trial], dtype=np.float64).ravel()
    y_trial = np.asarray(dff[selected_trial], dtype=np.float64).ravel()
    m = np.isfinite(t_trial) & np.isfinite(y_trial)
    t_trial = t_trial[m]
    y_trial = y_trial[m]
    if t_trial.size < 2:
        raise ValueError(f"Trial {selected_trial} has insufficient finite samples.")

    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    spike_times = spike_times[np.isfinite(spike_times)]
    spike_times = spike_times[(spike_times >= trial_start) & (spike_times <= trial_end)]

    if start_s is not None:
        win_start = float(np.clip(float(start_s), trial_start, trial_end - duration))
    else:
        if center == "median_spike" and spike_times.size:
            center_val = float(np.median(spike_times))
        else:
            center_val = float(0.5 * (trial_start + trial_end))
        win_start = float(np.clip(center_val - 0.5 * duration, trial_start, trial_end - duration))
    win_end = float(win_start + duration)

    ref_fs = _reference_fs_from_label(smoothing, raw_fs)
    ref_time, ref_gt_for_corr = build_ground_truth_series(
        spike_times, win_start, win_end, reference_fs=ref_fs, sigma_ms=float(corr_sigma_ms)
    )
    _, ref_gt_for_plot = build_ground_truth_series(
        spike_times, win_start, win_end, reference_fs=ref_fs, sigma_ms=float(display_sigma_ms)
    )
    ref_gt_for_plot = _normalize_0_1(ref_gt_for_plot)

    eval_root = eval_root.expanduser().resolve()
    method_results: Dict[str, Any] = {}
    for method in methods:
        run_tag = run_map.get(method, default_run)
        spec = CacheSpec(method=method, run_tag=run_tag, dataset=dataset_stem, smoothing=smoothing)
        entry = _load_comparison_method_entry(eval_root, spec)
        cache_tag, cache_key = _cache_paths_from_entry(entry, dataset_fallback=dataset_stem)
        method_results[method] = _load_method_cache_mat(method, cache_tag, cache_key)

    method_snippets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    method_corrs: Dict[str, float] = {}
    for method, result in method_results.items():
        segs = _segment_slices(result.time_stamps, result.sampling_rate)
        if len(segs) <= selected_trial:
            raise ValueError(f"Method {method} has {len(segs)} segments; cannot select trial {selected_trial}.")
        seg = segs[selected_trial]
        t_seg = np.asarray(result.time_stamps[seg], dtype=np.float64).ravel()
        y_seg = np.asarray(result.spike_prob[seg], dtype=np.float64).ravel()
        mm = np.isfinite(t_seg) & np.isfinite(y_seg)
        t_seg = t_seg[mm]
        y_seg = y_seg[mm]
        mwin = (t_seg >= win_start) & (t_seg <= win_end)
        t_win = t_seg[mwin]
        y_win = y_seg[mwin]
        if t_win.size < 2:
            method_snippets[method] = (np.array([win_start, win_end]), np.array([np.nan, np.nan]))
            method_corrs[method] = float("nan")
            continue

        y_disp_full = smooth_prediction(y_seg, result.sampling_rate, sigma_ms=float(display_sigma_ms))
        y_disp = np.asarray(y_disp_full, dtype=np.float64).ravel()[mwin]
        method_snippets[method] = (t_win, _normalize_0_1(y_disp))

        pred_smoothed = smooth_prediction(y_win, result.sampling_rate, sigma_ms=float(corr_sigma_ms))
        pred_aligned = resample_prediction_to_reference(t_win, pred_smoothed, ref_time, fs_est=result.sampling_rate)
        method_corrs[method] = _pearson(ref_gt_for_corr, pred_aligned)

    mwin_f = (t_trial >= win_start) & (t_trial <= win_end)
    t_f = t_trial[mwin_f]
    y_f = y_trial[mwin_f]
    if t_f.size < 2:
        raise ValueError("Fluorescence snippet has insufficient samples.")
    y_f0 = float(np.nanmedian(y_f))
    dff_scale = float(scalebar_dff) if float(scalebar_dff) > 0 else 0.5
    y_f_scaled = (y_f - y_f0) / dff_scale

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    else:
        fig = ax.figure

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.bottom": False,
            "ytick.left": False,
            "font.size": 11,
        }
    )

    ranked_methods = sorted(
        methods,
        key=lambda m: (
            -(method_corrs.get(m, float("nan")) if np.isfinite(method_corrs.get(m, float("nan"))) else -1e9),
            m,
        ),
    )
    row_gap = 1.0
    n_methods = len(ranked_methods)
    y_method_base = {m: (n_methods - 1 - i) * row_gap for i, m in enumerate(ranked_methods)}
    y_gt_base = n_methods * row_gap
    y_f_base = (n_methods + 1) * row_gap

    for s in spike_times:
        if s < win_start or s > win_end:
            continue
        ax.axvline(float(s), color="#888888", linestyle=":", linewidth=0.6, alpha=0.9, zorder=0)

    ax.plot(t_f, y_f_base + y_f_scaled, color="black", linewidth=1.0)
    ax.text(win_start, y_f_base + 0.8, "Fluorescence", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.plot(ref_time, y_gt_base + ref_gt_for_plot, color="black", linewidth=2.0)
    ax.text(win_start, y_gt_base + 0.8, "Ground truth", ha="left", va="center", fontsize=11, fontweight="bold")

    for method in ranked_methods:
        t_win, y_norm = method_snippets[method]
        base = y_method_base[method]
        color = colors_map.get(method, "#333333")
        label = labels_map.get(method, method)
        ax.plot(t_win, base + y_norm, color=color, linewidth=1.6)
        ax.text(win_start, base + 0.50, label, color=color, ha="left", va="center", fontsize=12, fontweight="bold")
        r = method_corrs.get(method, float("nan"))
        r_txt = "r = nan" if not np.isfinite(r) else f"r = {r:.2f}"
        ax.text(
            win_end + 0.16 * duration,
            base + 0.50,
            r_txt,
            color=color,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    sb_time = float(scalebar_time_s)
    sb_time = min(sb_time, duration)
    sb_x1 = win_end - 0.06 * duration
    sb_x0 = sb_x1 - sb_time
    sb_y0 = y_f_base + 0.9
    sb_y1 = sb_y0 + 1.0
    ax.plot([sb_x0, sb_x1], [sb_y1, sb_y1], color="black", linewidth=1.2)
    ax.plot([sb_x1, sb_x1], [sb_y0, sb_y1], color="black", linewidth=1.2)
    ax.text(0.5 * (sb_x0 + sb_x1), sb_y1 + 0.12, f"{sb_time:g} s", ha="center", va="bottom", fontsize=10)
    ax.text(sb_x1 + 0.02 * duration, 0.5 * (sb_y0 + sb_y1), f"{float(scalebar_dff):g} ΔF/F", ha="left", va="center", fontsize=10)

    ax.set_title(str(title), pad=8)
    ax.set_xlim(win_start, win_end + 0.18 * duration)
    ax.set_ylim(-0.4, y_f_base + 1.6)
    ax.set_axis_off()

    fig.tight_layout()
    meta = {
        "trial": int(selected_trial),
        "window_start_s": float(win_start),
        "window_end_s": float(win_end),
        "method_corrs": dict(method_corrs),
        "run_by_method": {m: run_map.get(m, default_run) for m in methods},
    }
    return fig, ax, meta

