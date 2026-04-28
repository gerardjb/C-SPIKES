from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# In some sandboxed/HPC environments Intel OpenMP shared-memory init can fail
# (e.g. "Can't open SHM2"). Using sequential threading avoids that. This must
# be set before importing numpy/matplotlib in fresh processes.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.io as sio


DEFAULT_COLORS: Dict[str, str] = {
    "pgas": "#009E73",
    "cascade": "#F3AE14",
    "ens2": "#A6780C",
    "biophys_ml": "#007755",
    "mlspike": "#0072B2",
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
    # In some sandboxed/HPC environments Intel OpenMP shared-memory init can fail
    # (e.g. "Can't open SHM2"). Using sequential threading avoids that.
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    candidates: List[Path] = []
    if cache_dir is not None:
        candidates.append(cache_dir)
    else:
        candidates.extend(
            [
                Path.cwd() / "tmp" / "mpl_cache",
                Path.cwd() / "results" / ".mplconfig",
                Path.home() / ".cache" / "c_spikes_mpl_cache",
                Path("/tmp") / "c_spikes_mpl_cache",
            ]
        )

    last_err: Optional[OSError] = None
    for cand in candidates:
        try:
            cand = cand.expanduser().resolve()
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_test"
            with probe.open("w", encoding="utf-8") as fh:
                fh.write("ok\n")
            try:
                probe.unlink()
            except OSError:
                pass
            os.environ.setdefault("MPLCONFIGDIR", str(cand))
            os.environ.setdefault("XDG_CACHE_HOME", str(cand))
            return cand
        except OSError as exc:
            last_err = exc
            continue

    # If everything failed, fall back without setting MPLCONFIGDIR. Matplotlib may still work
    # depending on the environment, but could be slower or error in locked-down contexts.
    if last_err is not None:
        raise last_err
    return Path.cwd()


Pathish = str | Path | os.PathLike[str]


def _to_path(path: Pathish) -> Path:
    return Path(path).expanduser().resolve()


def read_trialwise_csv(path: Pathish) -> List[Dict[str, str]]:
    path = _to_path(path)
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
    transform: Optional[Any] = None,
    pad_px: float = 2.0,
    bbox_expand_y: float = 1.08,
) -> None:
    # Respect input ordering (caller can rank-order labels); apply offsets to avoid overlap.
    ys = [float(y) for y in y_positions]

    for i in range(1, len(ys)):
        if not np.isfinite(ys[i - 1]) or not np.isfinite(ys[i]):
            continue
        if ys[i - 1] - ys[i] < min_sep:
            ys[i] = ys[i - 1] - min_sep

    if transform is None:
        transform = ax.transData

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
                transform=transform,
            )
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad_px = float(pad_px)
    bbox_expand_y = float(bbox_expand_y)
    if not np.isfinite(pad_px) or pad_px < 0:
        pad_px = 0.0
    if not np.isfinite(bbox_expand_y) or bbox_expand_y <= 0:
        bbox_expand_y = 1.0
    prev_bbox = None
    for text in texts:
        bbox = text.get_window_extent(renderer=renderer).expanded(1.0, bbox_expand_y)
        while prev_bbox is not None and bbox.overlaps(prev_bbox):
            shift_px = float(bbox.y1 - prev_bbox.y0 + pad_px)
            x_disp, y_disp = transform.transform(text.get_position())
            new_y_disp = y_disp - shift_px
            _, new_y = transform.inverted().transform((x_disp, new_y_disp))
            text.set_position((x, float(new_y)))
            fig.canvas.draw()
            bbox = text.get_window_extent(renderer=renderer).expanded(1.0, bbox_expand_y)
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
                x_disp, y_disp = transform.transform(t.get_position())
                new_y_disp = y_disp + shift_up
                _, new_y = transform.inverted().transform((x_disp, new_y_disp))
                t.set_position((x, float(new_y)))


def plot_corr_vs_sigma(
    *,
    csv_path: Pathish,
    out_path: Optional[Pathish] = None,
    runs: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    smoothings: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    reduce: str = "dataset",
    title: Optional[str] = None,
    ylabel: str = "Pearson correlation",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (0.6, 1.0),
    figsize: Tuple[float, float] = (7.2, 2.8),
    dpi: int = 200,
    legend: bool = False,
    right_label_x_offset_frac: float = 0.08,
    right_label_xlim_frac: float = 0.22,
    right_label_pad_px: float = 4.0,
    right_label_bbox_expand_y: float = 1.10,
    grid_x_step_ms: float = 20.0,
    grid_y_step_corr: float = 0.1,
    grid_color: str = "#808080",
    grid_alpha: float = 0.3,
    grid_linewidth: float = 1.0,
    axis_linewidth: float = 1.0,
    colors: Optional[Mapping[str, str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    ax: Any = None,
) -> Tuple[Any, Any]:
    """
    Mean ± SEM correlation vs corr_sigma_ms (ms), with right-side labels.

    To compare the same underlying method across different run tags (e.g. published ENS2 vs a
    synthetic-trained ENS2 distilled from PGAS), pass `methods` entries using the series syntax:
      - "ens2@base"
      - "biophys_ml=ens2@ens2_custom_k1_..."

    Returns (fig, ax). If out_path is provided, saves the figure.
    """
    if reduce not in {"dataset", "trial"}:
        raise ValueError("reduce must be 'dataset' or 'trial'")

    # Ensure we can write font/cache metadata on HPC (before importing matplotlib).
    ensure_matplotlib_cache_dir()
    # Lazy import so notebooks can configure matplotlib before importing.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.bottom": True,
            "ytick.left": True,
            "font.size": 11,
        }
    )

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

    # Parse series specs (method/run disambiguation + alias labels/colors).
    if methods is None:
        uniq_methods = sorted({str(r.get("method", "")).strip() for r in rows if r.get("method")})
        series = [SeriesSpec(key=m, method=m, run_tag=None) for m in uniq_methods if m]
    else:
        series = _parse_series_specs(methods)
    if not series:
        raise ValueError("No methods/series selected.")

    # Resolve rows to series keys. If run_tag is given, pin to that run; otherwise allow wildcard by method.
    series_lookup: Dict[Tuple[str, str], str] = {}
    wildcard_by_method: Dict[str, str] = {}
    for spec in series:
        expected_run = str(spec.run_tag).strip() if spec.run_tag else ""
        if expected_run:
            key = (spec.method, expected_run)
            if key in series_lookup:
                raise ValueError(f"Duplicate series method/run pair: {spec.method!r}@{expected_run!r}")
            series_lookup[key] = spec.key
        else:
            if spec.method in wildcard_by_method:
                raise ValueError(
                    f"Duplicate unpinned series for method {spec.method!r}; add @run_tag to disambiguate."
                )
            wildcard_by_method[spec.method] = spec.key

    filtered: List[Dict[str, Any]] = []
    for r in rows:
        run = str(r.get("run", "")).strip()
        dataset = str(r.get("dataset", "")).strip()
        smoothing = str(r.get("smoothing", "")).strip()
        if run_filter and run not in run_filter:
            continue
        if dataset_filter and dataset not in dataset_filter:
            continue
        if smoothing_filter and smoothing not in smoothing_filter:
            continue
        filtered.append(r)
    if not filtered:
        raise ValueError("No rows matched filters.")

    spec_by_key: Dict[str, SeriesSpec] = {spec.key: spec for spec in series}
    samples: Dict[Tuple[str, float], List[float]] = {}
    if reduce == "trial":
        for r in filtered:
            method = str(r.get("method", "")).strip()
            run = str(r.get("run", "")).strip()
            series_key = series_lookup.get((method, run))
            if series_key is None:
                series_key = wildcard_by_method.get(method)
            if series_key is None:
                continue
            sigma = _coerce_float(r, "corr_sigma_ms")
            corr = _coerce_float(r, "correlation")
            if not np.isfinite(sigma):
                continue
            samples.setdefault((series_key, float(sigma)), []).append(float(corr))
    else:
        by_bucket: Dict[Tuple[str, float, str, str, str], List[float]] = {}
        for r in filtered:
            method = str(r.get("method", "")).strip()
            sigma = _coerce_float(r, "corr_sigma_ms")
            dataset = str(r.get("dataset", "")).strip()
            run = str(r.get("run", "")).strip()
            smoothing = str(r.get("smoothing", "")).strip()
            corr = _coerce_float(r, "correlation")
            series_key = series_lookup.get((method, run))
            if series_key is None:
                series_key = wildcard_by_method.get(method)
            if series_key is None:
                continue
            if not dataset or not np.isfinite(sigma):
                continue
            by_bucket.setdefault((series_key, float(sigma), dataset, run, smoothing), []).append(float(corr))
        for (series_key, sigma, _dataset, _run, _smoothing), values in by_bucket.items():
            mean_val = float(np.mean(_finite(values))) if _finite(values).size else float("nan")
            samples.setdefault((series_key, float(sigma)), []).append(mean_val)

    series_keys_present = {m for (m, _sigma) in samples.keys()}
    series_list = [spec.key for spec in series if spec.key in series_keys_present]
    sigma_list = sorted({sigma for (_m, sigma) in samples.keys()})
    if not series_list or not sigma_list:
        raise ValueError("No usable (method, corr_sigma_ms) samples found.")

    sigma_min = float(min(sigma_list))
    sigma_max = float(max(sigma_list))
    if xlim is not None:
        x_left, x_right = float(xlim[0]), float(xlim[1])
        if not np.isfinite(x_left) or not np.isfinite(x_right) or x_right <= x_left:
            raise ValueError(f"Invalid xlim={xlim!r}; expected (lo, hi) with hi > lo.")
    else:
        x_span = max(1e-9, sigma_max - sigma_min)
        x_pad = max(5.0, 0.05 * x_span)
        x_left, x_right = sigma_min, sigma_max + x_pad
    x_anchor = float(min(sigma_max, x_right))

    # Derive a display label per series, appending "(run_tag)" only when the rendered labels would collide.
    label_counts_by_method: Dict[str, Dict[str, int]] = {}
    raw_label_by_key: Dict[str, str] = {}
    run_by_key: Dict[str, str] = {}
    for key in series_list:
        spec = spec_by_key.get(key)
        if spec is None:
            continue
        base_label = labels_map.get(spec.method, spec.method)
        label0 = labels_map.get(key, base_label)
        raw_label_by_key[key] = label0
        expected_run = str(spec.run_tag).strip() if spec.run_tag else ""
        run_by_key[key] = expected_run
        label_counts_by_method.setdefault(spec.method, {})
        label_counts_by_method[spec.method][label0] = label_counts_by_method[spec.method].get(label0, 0) + 1

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    else:
        fig = ax.figure

    plt.rcParams.update({"font.size": 12})
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(float(axis_linewidth))
    ax.spines["bottom"].set_linewidth(float(axis_linewidth))
    ax.tick_params(axis="both", which="both", width=float(axis_linewidth))

    method_to_color: Dict[str, str] = {}
    method_to_label: Dict[str, str] = {}
    method_to_y100: Dict[str, float] = {}

    for series_key in series_list:
        spec = spec_by_key[series_key]
        xs: List[float] = []
        ys: List[float] = []
        sems: List[float] = []
        for sigma in sigma_list:
            vals = samples.get((series_key, sigma), [])
            mean, sem, n = _mean_sem(vals)
            xs.append(float(sigma))
            ys.append(float(mean) if n else float("nan"))
            sems.append(float(sem) if n else float("nan"))

        xs_arr = np.asarray(xs, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.float64)
        sem_arr = np.asarray(sems, dtype=np.float64)

        color = colors_map.get(series_key, colors_map.get(spec.method))
        if color is None:
            color = plt.cm.tab10(hash(series_key) % 10)  # type: ignore[arg-type]

        ax.plot(xs_arr, ys_arr, color=color, linewidth=2.5)
        ax.fill_between(xs_arr, ys_arr - sem_arr, ys_arr + sem_arr, color=color, alpha=0.18, linewidth=0)

        method_to_color[series_key] = str(color)
        label = raw_label_by_key.get(series_key, labels_map.get(spec.method, spec.method))
        expected_run = run_by_key.get(series_key, "")
        if label_counts_by_method.get(spec.method, {}).get(label, 0) > 1 and expected_run:
            label = f"{label} ({expected_run})"
        method_to_label[series_key] = str(label)
        method_to_y100[series_key] = _y_at_x(xs_arr, ys_arr, x_anchor)

    ax.set_xlabel("Filter Width (ms)")
    ax.set_ylabel(str(ylabel))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if title:
        ax.set_title(str(title))

    ax.set_xlim(x_left, x_right)
    right_x = 1.0 + float(right_label_x_offset_frac)

    # Tick/grid conventions: show lines at fixed increments (with labeled ticks).
    ax.xaxis.set_major_locator(MultipleLocator(float(grid_x_step_ms)))
    ax.yaxis.set_major_locator(MultipleLocator(float(grid_y_step_corr)))
    ax.grid(True, which="major", color=str(grid_color), alpha=float(grid_alpha), linewidth=float(grid_linewidth))
    ax.tick_params(which="major", length=6)

    if legend:
        handles = list(ax.get_lines())
        legend_labels = [method_to_label.get(m, m) for m in series_list]
        ax.legend(handles, legend_labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        ranked = sorted(
            ((m, method_to_y100.get(m, float("nan"))) for m in series_list),
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
            x=right_x,
            y_positions=right_y,
            labels=right_labels,
            colors=right_colors,
            min_sep=min_sep,
            transform=ax.get_yaxis_transform(),
            pad_px=float(right_label_pad_px),
            bbox_expand_y=float(right_label_bbox_expand_y),
        )

    rect_right = 1.0 - float(right_label_xlim_frac)
    if not np.isfinite(rect_right) or rect_right <= 0.2:
        rect_right = 1.0
    fig.tight_layout(rect=(0.0, 0.0, rect_right, 1.0))
    if out_path is not None:
        out_path = _to_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig, ax


def _load_edges(path: Optional[Pathish]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    edges_path = _to_path(path)
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
class SeriesSpec:
    key: str
    method: str
    run_tag: Optional[str] = None


def _parse_series_specs(items: Sequence[str]) -> List[SeriesSpec]:
    """
    Parse a list of series specs.

    Accepted forms:
      - "ens2"                         -> key="ens2", method="ens2", run_tag=None
      - "ens2@base"                    -> key="ens2@base", method="ens2", run_tag="base"
      - "Published ENS2=ens2@base"     -> key="Published ENS2", method="ens2", run_tag="base"
    """
    specs: List[SeriesSpec] = []
    seen_keys: set[str] = set()
    for raw in items:
        token = str(raw).strip()
        if not token:
            continue
        if "=" in token:
            key_part, rest = token.split("=", 1)
            key = key_part.strip()
            rest = rest.strip()
        else:
            key = ""
            rest = token
        if "@" in rest:
            method_part, run_part = rest.split("@", 1)
            method = method_part.strip()
            run_tag = run_part.strip()
            if not run_tag:
                raise ValueError(f"Invalid series spec (empty run tag): {token!r}")
        else:
            method = rest.strip()
            run_tag = None
        if not method:
            raise ValueError(f"Invalid series spec (empty method): {token!r}")
        if not key:
            key = f"{method}@{run_tag}" if run_tag else method
        if key in seen_keys:
            raise ValueError(f"Duplicate series key: {key!r}")
        seen_keys.add(key)
        specs.append(SeriesSpec(key=key, method=method, run_tag=run_tag))
    return specs


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


def _select_segment_for_trial(
    segments: Sequence[slice],
    times: np.ndarray,
    trial_idx: int,
    start: float,
    end: float,
    *,
    n_trials: Optional[int] = None,
) -> Optional[slice]:
    if not segments:
        return None
    if n_trials is not None and len(segments) == int(n_trials) and 0 <= trial_idx < len(segments):
        candidate = segments[trial_idx]
        cand_times = np.asarray(times, dtype=np.float64).ravel()[candidate]
        cand_times = cand_times[np.isfinite(cand_times)]
        if cand_times.size and float(cand_times[-1]) >= float(start) and float(cand_times[0]) <= float(end):
            return candidate
    times = np.asarray(times, dtype=np.float64).ravel()
    best: Optional[slice] = None
    best_overlap = -1.0
    for seg in segments:
        seg_times = times[seg]
        seg_times = seg_times[np.isfinite(seg_times)]
        if seg_times.size == 0:
            continue
        seg_start = float(seg_times[0])
        seg_end = float(seg_times[-1])
        overlap = max(0.0, min(seg_end, float(end)) - max(seg_start, float(start)))
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    if best_overlap <= 0.0:
        return None
    return best


def plot_trace_panel(
    *,
    csv_path: Pathish,
    eval_root: Pathish,
    data_root: Pathish,
    dataset: str,
    smoothing: str = "raw",
    corr_sigma_ms: float = 50.0,
    display_sigma_ms: Optional[float] = None,
    edges_path: Optional[Pathish] = None,
    methods: Optional[Sequence[str]] = None,
    run: Optional[str] = None,
    run_by_method: Optional[Sequence[str]] = None,
    trial: Optional[int] = None,
    duration_s: float = 5.0,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    center: str = "median_spike",
    row_pad_frac: float = 0.05,
    gt_method_pad_frac: float = 0.05,
    dff_height: float = 1.25,
    method_label_x_offset_frac: float = 0.0,
    show_snippet_corr: bool = False,
    show_panel_labels: bool = True,
    show_method_labels: bool = True,
    show_scalebar: bool = True,
    scalebar_time_s: float = 0.6,
    scalebar_dff: float = 0.5,
    ymax: Optional[float] = None,
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

    To compare multiple run tags for the *same* method (e.g. two ENS2 models), pass distinct entries in
    `methods` using the `method@run_tag` syntax, e.g.:
      methods=["pgas", "ens2@ens2_published_rerun", "ens2@ens2_custom_...", "cascade"]

    Correlation text:
      - By default, uses the full-trial correlation loaded from `csv_path` (trialwise_correlations.csv).
      - If `show_snippet_corr=True`, also computes a correlation over the displayed snippet window and
        shows both values (CSV value on top, snippet value below).

    Labels/scalebar:
      - Set `show_panel_labels=False` to hide the "Fluorescence" and "Ground truth" labels.
      - Set `show_method_labels=False` to hide method names at left of each trace.
      - Set `show_scalebar=False` to hide the time/ΔF/F scalebar.

    Snippet window:
      - The snippet is clamped to the intersection of all selected methods' per-trial time support.
        This avoids cases where a method (notably PGAS) only has outputs for a sub-window of the epoch
        and would otherwise appear flat (all-NaN after resampling).
      - You can override the snippet window by providing both `start_s` and `end_s` (seconds).

    Axis limits:
      - Use `ymax` to override the upper y-limit (the lower limit is fixed for this panel layout).
    """
    if methods is None:
        methods = ["pgas", "cascade", "ens2"]
    if center not in {"median_spike", "trial_mid"}:
        raise ValueError("center must be 'median_spike' or 'trial_mid'")
    if float(dff_height) <= 0:
        raise ValueError("dff_height must be positive.")
    if float(row_pad_frac) < 0 or float(gt_method_pad_frac) < 0:
        raise ValueError("row_pad_frac and gt_method_pad_frac must be non-negative.")

    # Ensure we can write font/cache metadata on HPC (before importing matplotlib).
    ensure_matplotlib_cache_dir()
    # Lazy import for notebook friendliness.
    import matplotlib.pyplot as plt

    from c_spikes.inference.eval import build_ground_truth_series, resample_prediction_to_reference
    from c_spikes.inference.smoothing import mean_downsample_trace
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
    series = _parse_series_specs(methods)
    if not series:
        raise ValueError("No methods/series selected.")
    run_map = _parse_run_by_method(run_by_method)
    default_run = str(run).strip() if run else "cascadein_nodisc_ens2"

    rows = read_trialwise_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    # Resolve each series to a concrete (method, run_tag) pair.
    series_lookup: Dict[Tuple[str, str], str] = {}
    series_run: Dict[str, str] = {}
    for spec in series:
        run_tag = str(spec.run_tag).strip() if spec.run_tag else run_map.get(spec.method, default_run)
        if not run_tag:
            raise ValueError(f"Series {spec.key!r} has no run_tag (set method@run_tag or run_by_method).")
        key = (spec.method, run_tag)
        if key in series_lookup:
            raise ValueError(f"Duplicate series method/run pair: {spec.method!r}@{run_tag!r}")
        series_lookup[key] = spec.key
        series_run[spec.key] = run_tag

    corr_by_method_trial: Dict[str, Dict[int, float]] = {spec.key: {} for spec in series}
    for r in rows:
        if r.get("dataset") != dataset_stem:
            continue
        if r.get("smoothing") != smoothing:
            continue
        if not np.isclose(_coerce_float(r, "corr_sigma_ms"), float(corr_sigma_ms), atol=1e-6):
            continue
        method = str(r.get("method", "")).strip()
        run_tag = str(r.get("run", "")).strip()
        series_key = series_lookup.get((method, run_tag))
        if series_key is None:
            continue
        trial_idx = int(float(r.get("trial", "nan")))
        corr_by_method_trial[series_key][trial_idx] = _coerce_float(r, "correlation")

    missing = [spec.key for spec in series if not corr_by_method_trial.get(spec.key)]
    if missing:
        raise ValueError(
            "Missing trialwise correlations for series: "
            + ", ".join(missing)
            + ". Check run/run_by_method/method@run_tag and corr_sigma_ms."
        )

    medians: Dict[str, float] = {}
    for method, trial_map in corr_by_method_trial.items():
        vals = _finite(list(trial_map.values()))
        medians[method] = float(np.median(vals)) if vals.size else float("nan")

    common_trials = set.intersection(*(set(m.keys()) for m in corr_by_method_trial.values()))
    if not common_trials:
        raise ValueError("No trial indices are shared across selected series/run tags.")

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
            for method in corr_by_method_trial:
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

    # Correlations to display come from the provided trialwise CSV (computed over the *full trial window*).
    trial_corrs: Dict[str, float] = {
        method: float(corr_by_method_trial[method].get(selected_trial, float("nan"))) for method in corr_by_method_trial
    }

    data_root = _to_path(data_root)
    dataset_path = data_root / f"{dataset_stem}.mat"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    edges_lookup = _load_edges(edges_path)
    trial_windows, raw_fs = _trial_windows_from_mat(dataset_path, edges_lookup=edges_lookup)
    if selected_trial < 0 or selected_trial >= len(trial_windows):
        raise ValueError(f"Trial index {selected_trial} out of range (n_trials={len(trial_windows)}).")

    time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))
    t_trial = np.asarray(time_stamps[selected_trial], dtype=np.float64).ravel()
    y_trial = np.asarray(dff[selected_trial], dtype=np.float64).ravel()
    m = np.isfinite(t_trial) & np.isfinite(y_trial)
    t_trial = t_trial[m]
    y_trial = y_trial[m]
    if t_trial.size < 2:
        raise ValueError(f"Trial {selected_trial} has insufficient finite samples.")

    target_fs = _parse_smoothing_fs(smoothing)
    if target_fs is not None:
        ds = mean_downsample_trace(t_trial, y_trial, float(target_fs))
        t_trial = np.asarray(ds.times, dtype=np.float64).ravel()
        y_trial = np.asarray(ds.values, dtype=np.float64).ravel()

    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    spike_times = spike_times[np.isfinite(spike_times)]

    eval_root = _to_path(eval_root)
    method_results: Dict[str, Any] = {}
    seg_ranges: List[Tuple[float, float]] = []
    seg_slices: Dict[str, slice] = {}
    baseline_start, baseline_end = trial_windows[selected_trial]
    spec_by_key: Dict[str, SeriesSpec] = {spec.key: spec for spec in series}
    for series_key, spec in spec_by_key.items():
        run_tag = series_run[series_key]
        cache_spec = CacheSpec(method=spec.method, run_tag=run_tag, dataset=dataset_stem, smoothing=smoothing)
        entry = _load_comparison_method_entry(eval_root, cache_spec)
        cache_tag, cache_key = _cache_paths_from_entry(entry, dataset_fallback=dataset_stem)
        method_results[series_key] = _load_method_cache_mat(spec.method, cache_tag, cache_key)
        result = method_results[series_key]
        segs = _segment_slices(result.time_stamps, result.sampling_rate)
        seg = _select_segment_for_trial(
            segs,
            result.time_stamps,
            int(selected_trial),
            float(baseline_start),
            float(baseline_end),
            n_trials=len(trial_windows),
        )
        if seg is None:
            raise ValueError(f"Series {series_key} has no segment overlapping trial {selected_trial}.")
        seg_slices[series_key] = seg
        t_seg = np.asarray(result.time_stamps[seg], dtype=np.float64).ravel()
        t_seg = t_seg[np.isfinite(t_seg)]
        if t_seg.size == 0:
            raise ValueError(f"Series {series_key} segment for trial {selected_trial} has no finite time stamps.")
        seg_ranges.append((float(np.min(t_seg)), float(np.max(t_seg))))

    # Pick an evaluation window that lies inside *all* methods' trial segments. This prevents
    # cases where a method (notably PGAS) only has outputs for a sub-window of the epoch, and
    # the chosen snippet lands outside that support (making the trace appear flat after normalization).
    seg_start = max(s for s, _e in seg_ranges)
    seg_end = min(e for _s, e in seg_ranges)
    trial_start = max(float(baseline_start), float(seg_start))
    trial_end = min(float(baseline_end), float(seg_end))
    if trial_end <= trial_start:
        # Fall back to the baseline trial window; downstream resampling may still yield NaNs.
        trial_start = float(baseline_start)
        trial_end = float(baseline_end)

    spike_times_trial = spike_times[(spike_times >= trial_start) & (spike_times <= trial_end)]

    if end_s is not None:
        if start_s is None:
            raise ValueError("end_s requires start_s.")
        win_start_req = float(start_s)
        win_end_req = float(end_s)
        if not np.isfinite(win_start_req) or not np.isfinite(win_end_req):
            raise ValueError("start_s and end_s must be finite.")
        if win_end_req <= win_start_req:
            raise ValueError("end_s must be > start_s.")
        win_start = float(np.clip(win_start_req, trial_start, trial_end))
        win_end = float(np.clip(win_end_req, trial_start, trial_end))
        duration = float(win_end - win_start)
        if duration <= 0:
            raise ValueError("Selected snippet window is empty after clamping to the trial window.")
    else:
        duration = float(duration_s)
        if duration <= 0:
            raise ValueError("duration_s must be positive.")
        duration = min(duration, float(trial_end - trial_start))
        if duration <= 0:
            raise ValueError("Selected trial window is empty after intersecting method segments.")

        if start_s is not None:
            win_start = float(np.clip(float(start_s), trial_start, trial_end - duration))
        else:
            if center == "median_spike" and spike_times_trial.size:
                center_val = float(np.median(spike_times_trial))
            else:
                center_val = float(0.5 * (trial_start + trial_end))
            win_start = float(np.clip(center_val - 0.5 * duration, trial_start, trial_end - duration))
        win_end = float(win_start + duration)
    label_x = float(win_start + float(method_label_x_offset_frac) * duration)

    # Fluorescence snippet (plotted at the same sampling as the selected `smoothing`).
    mwin_f = (t_trial >= win_start) & (t_trial <= win_end)
    t_f = t_trial[mwin_f]
    y_f = y_trial[mwin_f]
    if t_f.size < 2:
        raise ValueError("Fluorescence snippet has insufficient samples.")

    # Reference grid for GT + correlations (keep consistent with trialwise_correlations.py).
    ref_fs = _reference_fs_from_label(smoothing, raw_fs)
    ref_time, ref_gt_for_plot = build_ground_truth_series(
        spike_times_trial, win_start, win_end, reference_fs=ref_fs, sigma_ms=float(display_sigma_ms)
    )
    ref_gt_for_corr: Optional[np.ndarray] = None
    if show_snippet_corr:
        corr_time, corr_trace = build_ground_truth_series(
            spike_times_trial, win_start, win_end, reference_fs=ref_fs, sigma_ms=float(corr_sigma_ms)
        )
        corr_time = np.asarray(corr_time, dtype=np.float64).ravel()
        corr_trace = np.asarray(corr_trace, dtype=np.float64).ravel()
        if corr_time.shape != ref_time.shape or not np.allclose(corr_time, ref_time):
            corr_trace = np.interp(ref_time, corr_time, corr_trace)
        ref_gt_for_corr = corr_trace
    ref_gt_for_plot = _normalize_0_1(ref_gt_for_plot)
    if ref_time.size < 2:
        raise ValueError("Reference grid has insufficient samples for plotting/correlation.")

    method_snippets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    snippet_corrs: Dict[str, float] = {}
    for series_key, result in method_results.items():
        method = spec_by_key[series_key].method
        seg = seg_slices[series_key]
        t_seg = np.asarray(result.time_stamps[seg], dtype=np.float64).ravel()
        y_seg = np.asarray(result.spike_prob[seg], dtype=np.float64).ravel()
        mm = np.isfinite(t_seg) & np.isfinite(y_seg)
        t_seg = t_seg[mm]
        y_seg = y_seg[mm]
        if method == "pgas" and y_seg.size >= 2:
            # PGAS spike counts are reported on the *right edge* of each interval; shift
            # left by one sample so peaks line up with the start-of-bin GT convention.
            y_seg = np.concatenate([y_seg[1:], np.array([0.0], dtype=y_seg.dtype)])

        y_disp_full = smooth_prediction(y_seg, result.sampling_rate, sigma_ms=float(display_sigma_ms))
        y_disp = resample_prediction_to_reference(t_seg, y_disp_full, ref_time, fs_est=result.sampling_rate)
        method_snippets[series_key] = (ref_time, _normalize_0_1(y_disp))
        if show_snippet_corr and ref_gt_for_corr is not None:
            pred_smoothed_full = smooth_prediction(y_seg, result.sampling_rate, sigma_ms=float(corr_sigma_ms))
            pred_aligned = resample_prediction_to_reference(
                t_seg, pred_smoothed_full, ref_time, fs_est=result.sampling_rate
            )
            snippet_corrs[series_key] = _pearson(ref_gt_for_corr, pred_aligned)
    y_f0 = float(np.nanmedian(y_f))
    y_f_p1 = float(np.nanpercentile(y_f, 1))
    y_f_p99 = float(np.nanpercentile(y_f, 99))
    dff_span_raw = float(max(0.0, y_f_p99 - y_f_p1))
    # Scale ΔF/F into plot units so the robust span occupies ~`dff_height` y-units.
    # (GT/method traces are normalized to ~[0,1] in y-units.)
    dff_unit_scale = dff_span_raw / float(dff_height) if dff_span_raw > 0 else float(scalebar_dff or 0.5)
    if not np.isfinite(dff_unit_scale) or dff_unit_scale <= 0:
        dff_unit_scale = float(scalebar_dff or 0.5)
    y_f_scaled = (y_f - y_f0) / float(dff_unit_scale)
    dff_min = float(np.nanpercentile(y_f_scaled, 1)) if np.isfinite(y_f_scaled).any() else 0.0
    dff_max = float(np.nanpercentile(y_f_scaled, 99)) if np.isfinite(y_f_scaled).any() else 0.0
    dff_span = float(max(0.0, dff_max - dff_min))

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
        list(trial_corrs.keys()),
        key=lambda m: (
            -(trial_corrs.get(m, float("nan")) if np.isfinite(trial_corrs.get(m, float("nan"))) else -1e9),
            m,
        ),
    )
    # Each normalized trace occupies ~[0, 1] in y. Add a small pad between rows.
    # Default is 5% of the trace height, which is close to the "shadedErrorBar" style spacing.
    trace_height = 1.0
    row_pad = float(row_pad_frac) * trace_height
    row_step = trace_height + row_pad
    n_methods = len(ranked_methods)
    y_method_base = {m: (n_methods - 1 - i) * row_step for i, m in enumerate(ranked_methods)}
    gt_pad = float(gt_method_pad_frac) * trace_height
    y_gt_base = n_methods * row_step + gt_pad

    # Place fluorescence above GT with a gap that prevents overlap. Because fluorescence can span
    # beyond [0, 1] in this plot (it's scaled by `scalebar_dff`), ensure the *bottom* of the
    # fluorescence trace clears the *top* of GT with padding.
    dff_bottom_clearance = max(0.0, -dff_min)
    dff_extra_gap = float(row_pad_frac) * max(1.0, dff_span)
    y_f_base = y_gt_base + trace_height + row_pad + dff_bottom_clearance + dff_extra_gap

    for s in spike_times_trial:
        if s < win_start or s > win_end:
            continue
        ax.axvline(float(s), color="#888888", linestyle=":", linewidth=0.6, alpha=0.9, zorder=0)

    ax.plot(t_f, y_f_base + y_f_scaled, color="black", linewidth=1.0)
    if show_panel_labels:
        ax.text(label_x, y_f_base + 0.8, "Fluorescence", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.plot(ref_time, y_gt_base + ref_gt_for_plot, color="black", linewidth=2.0)
    if show_panel_labels:
        ax.text(label_x, y_gt_base + 0.8, "Ground truth", ha="left", va="center", fontsize=11, fontweight="bold")

    # Only append "(run_tag)" to series labels when the *rendered* labels would otherwise collide for
    # the same base method. This lets users alias one ENS2 run as e.g. `biophys_ml=ens2@...` without
    # forcing the remaining ENS2 series to show "(base)".
    label_counts_by_method: Dict[str, Dict[str, int]] = {}
    raw_label_by_key: Dict[str, str] = {}
    for spec in series:
        base_label = labels_map.get(spec.method, spec.method)
        label0 = labels_map.get(spec.key, base_label)
        raw_label_by_key[spec.key] = label0
        label_counts_by_method.setdefault(spec.method, {})
        label_counts_by_method[spec.method][label0] = label_counts_by_method[spec.method].get(label0, 0) + 1

    for method in ranked_methods:
        t_win, y_norm = method_snippets[method]
        base = y_method_base[method]
        series_spec = spec_by_key[method]
        run_tag = series_run.get(method, "")
        color = colors_map.get(method, colors_map.get(series_spec.method, "#333333"))
        label = raw_label_by_key.get(method, labels_map.get(series_spec.method, series_spec.method))
        if label_counts_by_method.get(series_spec.method, {}).get(label, 0) > 1 and run_tag:
            label = f"{label} ({run_tag})"
        ax.plot(t_win, base + y_norm, color=color, linewidth=1.6)
        if show_method_labels:
            ax.text(label_x, base + 0.50, label, color=color, ha="left", va="center", fontsize=12, fontweight="bold")
        r_trial = float(trial_corrs.get(method, float("nan")))
        if show_snippet_corr:
            r_snip = float(snippet_corrs.get(method, float("nan")))
            r_txt = (
                f"r = {'nan' if not np.isfinite(r_trial) else f'{r_trial:.2f}'}\n"
                f"r = {'nan' if not np.isfinite(r_snip) else f'{r_snip:.2f}'}"
            )
        else:
            r_txt = f"r = {'nan' if not np.isfinite(r_trial) else f'{r_trial:.2f}'}"
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

    y_min_plot = -0.4
    y_top_auto = y_f_base + max(1.0, dff_max) + 1.1
    y_top = float(y_top_auto) if ymax is None else float(ymax)
    if not np.isfinite(y_top) or y_top <= y_min_plot:
        raise ValueError(f"ymax must be > {y_min_plot}, got {ymax!r}")

    if show_scalebar:
        sb_time = float(scalebar_time_s)
        sb_time = min(sb_time, duration)
        sb_x1 = win_end - 0.06 * duration
        sb_x0 = sb_x1 - sb_time
        sb_height = float(scalebar_dff) / float(dff_unit_scale)
        sb_text_pad = 0.30
        sb_margin = 0.05
        sb_y0_default = y_f_base + 0.9
        sb_y0 = min(sb_y0_default, y_top - sb_height - sb_text_pad - sb_margin)
        sb_y0 = max(sb_y0, y_min_plot + sb_margin)
        # Draw a vertical ΔF/F scalebar corresponding to `scalebar_dff` in raw units.
        sb_y1 = sb_y0 + sb_height
        ax.plot([sb_x0, sb_x1], [sb_y1, sb_y1], color="black", linewidth=1.2)
        ax.plot([sb_x1, sb_x1], [sb_y0, sb_y1], color="black", linewidth=1.2)
        time_label_y = min(sb_y1 + 0.12, y_top - sb_margin)
        ax.text(0.5 * (sb_x0 + sb_x1), time_label_y, f"{sb_time:g} s", ha="center", va="bottom", fontsize=10)
        ax.text(
            sb_x1 + 0.02 * duration,
            0.5 * (sb_y0 + sb_y1),
            f"{float(scalebar_dff):g} ΔF/F",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_title(str(title), pad=8)
    ax.set_xlim(win_start, win_end + 0.18 * duration)
    ax.set_ylim(y_min_plot, y_top)
    ax.set_axis_off()

    fig.tight_layout()
    meta = {
        "trial": int(selected_trial),
        "window_start_s": float(win_start),
        "window_end_s": float(win_end),
        "method_corrs": dict(trial_corrs),
        "snippet_corrs": dict(snippet_corrs) if show_snippet_corr else None,
        # Backwards-compatible-ish: previously this was method->run_tag; now it is series_key->run_tag.
        "run_by_method": {spec.key: series_run.get(spec.key) for spec in series},
        "series": [{"key": spec.key, "method": spec.method, "run_tag": series_run.get(spec.key)} for spec in series],
    }
    return fig, ax, meta


def _parse_smoothing_fs(label: str) -> Optional[float]:
    token = str(label).strip()
    if token.lower() == "raw":
        return None
    if token.lower().endswith("hz"):
        try:
            return float(token[:-2].replace("p", "."))
        except ValueError:
            return None
    return None


def _sort_smoothing_labels(labels: Sequence[str]) -> List[str]:
    def key(lbl: str) -> Tuple[int, float, str]:
        fs = _parse_smoothing_fs(lbl)
        if fs is None:
            # Raw (highest effective rate) first.
            return (0, 1e9, lbl)
        # Higher Hz first.
        return (1, -fs, lbl)

    return sorted({str(l).strip() for l in labels if str(l).strip()}, key=key)


def plot_raincloud_by_downsample(
    *,
    csv_path: Pathish,
    out_path: Optional[Pathish] = None,
    corr_sigma_ms: float = 50.0,
    methods: Optional[Sequence[str]] = None,
    smoothings: Optional[Sequence[str]] = None,
    runs: Optional[Sequence[str]] = None,
    run_by_method: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    reduce: str = "trial",
    title: Optional[str] = None,
    ylim: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[float, float] = (7.2, 4.2),
    dpi: int = 200,
    seed: int = 0,
    group_width: float = 0.80,
    group_spacing: float = 1.25,
    method_label_x_offset_frac: float = 0.05,
    colors: Optional[Mapping[str, str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    ax: Any = None,
) -> Tuple[Any, Any]:
    """
    Raincloud-style distribution of trialwise correlations vs downsample rate.

    Layout:
      - one axis
      - x-axis categories: smoothing labels (raw / 30Hz / 10Hz / ...)
      - within each smoothing category, method distributions are shown side-by-side
      - y-axis: correlation values

    `reduce`:
      - "trial": each trial correlation is a sample (default)
      - "dataset": average across trials within each dataset/run/smoothing first

    `run_by_method`:
      - Optional list of strings like ["pgas=base", "ens2=ens2_custom_k1_..."].
      - When provided, this selects the run tag to use for each method (instead of a global `runs` filter),
        which is useful for mixing e.g. PGAS/CASCADE from `base` with ENS2 from a custom model run tag.
      - Note: if you want to plot the *same method* from multiple runs (e.g. ENS2 published vs ENS2 custom),
        pass distinct entries in `methods` using the `method@run_tag` syntax, e.g.
        methods=["ens2@base", "ens2@ens2_custom_..."].

    `group_spacing`:
      - Multiplies the spacing between adjacent downsample groups on the x-axis (default: 1.25).
        Use values >1.0 to create more whitespace between smoothing categories.

    `method_label_x_offset_frac`:
      - Horizontal offset for the right-hand method labels, expressed as a fraction of the x-span
        of the plotted categories.
    """
    if reduce not in {"trial", "dataset"}:
        raise ValueError("reduce must be 'trial' or 'dataset'")

    ensure_matplotlib_cache_dir()
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

    if methods is None:
        uniq_methods = sorted({str(r.get("method", "")).strip() for r in rows if r.get("method")})
        series = [SeriesSpec(key=m, method=m, run_tag=None) for m in uniq_methods if m]
    else:
        series = _parse_series_specs(methods)
    if not series:
        raise ValueError("No methods selected.")

    # Determine smoothing categories (x-axis) either from user or data.
    if smoothings is None:
        smoothings = _sort_smoothing_labels(list({r.get("smoothing", "") for r in rows if r.get("smoothing")}))  # type: ignore[arg-type]
    else:
        # Keep user-provided order, but drop empties.
        smoothings = [str(s).strip() for s in smoothings if str(s).strip()]
        if not smoothings:
            raise ValueError("No smoothings selected.")

    run_filter = set(_uniq(runs) or [])
    dataset_filter = set(_uniq(datasets) or [])
    run_map = _parse_run_by_method(run_by_method)

    # Build distribution samples per (method, smoothing).
    series_lookup: Dict[Tuple[str, str], str] = {}
    wildcard_by_method: Dict[str, str] = {}
    spec_by_key: Dict[str, SeriesSpec] = {spec.key: spec for spec in series}
    for spec in series:
        expected_run = str(spec.run_tag).strip() if spec.run_tag else run_map.get(spec.method)
        if expected_run:
            key = (spec.method, expected_run)
            if key in series_lookup:
                raise ValueError(f"Duplicate series method/run pair: {spec.method!r}@{expected_run!r}")
            series_lookup[key] = spec.key
        else:
            if spec.method in wildcard_by_method:
                raise ValueError(f"Duplicate unpinned series for method {spec.method!r}; add @run_tag to disambiguate.")
            wildcard_by_method[spec.method] = spec.key

    values_by_ms: Dict[Tuple[str, str], List[float]] = {(spec.key, s): [] for spec in series for s in smoothings}
    if reduce == "trial":
        for r in rows:
            if not np.isclose(_coerce_float(r, "corr_sigma_ms"), float(corr_sigma_ms), atol=1e-6):
                continue
            run = str(r.get("run", "")).strip()
            if run_filter and run not in run_filter:
                continue
            dataset = str(r.get("dataset", "")).strip()
            if dataset_filter and dataset not in dataset_filter:
                continue
            method = str(r.get("method", "")).strip()
            smoothing = str(r.get("smoothing", "")).strip()
            series_key = series_lookup.get((method, run))
            if series_key is None:
                series_key = wildcard_by_method.get(method)
            if series_key is None:
                continue
            if smoothing not in smoothings:
                continue
            values_by_ms[(series_key, smoothing)].append(_coerce_float(r, "correlation"))
    else:
        by_bucket: Dict[Tuple[str, str, str, str], List[float]] = {}
        # (series_key, smoothing, dataset, run) -> trial correlations
        for r in rows:
            if not np.isclose(_coerce_float(r, "corr_sigma_ms"), float(corr_sigma_ms), atol=1e-6):
                continue
            run = str(r.get("run", "")).strip()
            if run_filter and run not in run_filter:
                continue
            method = str(r.get("method", "")).strip()
            smoothing = str(r.get("smoothing", "")).strip()
            dataset = str(r.get("dataset", "")).strip()
            if dataset_filter and dataset not in dataset_filter:
                continue
            series_key = series_lookup.get((method, run))
            if series_key is None:
                series_key = wildcard_by_method.get(method)
            if series_key is None:
                continue
            if smoothing not in smoothings:
                continue
            if not dataset:
                continue
            by_bucket.setdefault((series_key, smoothing, dataset, run), []).append(_coerce_float(r, "correlation"))
        for (series_key, smoothing, _dataset, _run), vals in by_bucket.items():
            mean_val = float(np.mean(_finite(vals))) if _finite(vals).size else float("nan")
            values_by_ms[(series_key, smoothing)].append(mean_val)

    rng = np.random.default_rng(int(seed))

    if float(group_width) <= 0:
        raise ValueError("group_width must be positive.")
    group_width = float(min(0.95, group_width))
    if float(group_spacing) <= 0:
        raise ValueError("group_spacing must be positive.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    else:
        fig = ax.figure
    axes_list = [ax]

    x_centers = np.arange(len(smoothings), dtype=np.float64) * float(group_spacing)
    n_series = len(series)
    if n_series == 0:
        raise ValueError("No methods selected.")

    # Method positions within each downsample group.
    if n_series == 1:
        offsets = np.array([0.0], dtype=np.float64)
    else:
        step = group_width / float(n_series)
        offsets = (np.arange(n_series, dtype=np.float64) - (n_series - 1) / 2.0) * step
    violin_width = min(0.70 * group_width / float(n_series), 0.35)
    box_width = 0.35 * violin_width

    # Plot each (smoothing, method) distribution at its grouped x position.
    for s_idx, smoothing in enumerate(smoothings):
        x0 = float(x_centers[s_idx])
        for m_idx, spec in enumerate(series):
            vals = _finite(values_by_ms.get((spec.key, smoothing), []))
            if vals.size == 0:
                continue
            pos = x0 + float(offsets[m_idx])
            color = colors_map.get(spec.key, colors_map.get(spec.method, "#333333"))

            vio = ax.violinplot([vals], positions=[pos], widths=float(violin_width), showextrema=False)
            for body in vio.get("bodies", []):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.18)
                body.set_linewidth(1.0)

            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=float(box_width),
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                zorder=3,
            )
            for patch in bp.get("boxes", []):
                # Use a high-contrast box outline without filling (so points remain visible).
                patch.set_facecolor("none")
                patch.set_alpha(1.0)
                # Keep the fill method-colored, but draw summary stats in high-contrast black.
                patch.set_edgecolor("#000000")
                patch.set_linewidth(1.2)
                patch.set_zorder(4)
            for key, lw in (("medians", 1.4), ("whiskers", 1.0), ("caps", 1.0)):
                for line in bp.get(key, []):
                    line.set_color("#000000")
                    line.set_linewidth(float(lw))
                    line.set_zorder(5)

            jitter = rng.normal(loc=0.0, scale=0.10 * violin_width, size=vals.size)
            ax.scatter(
                np.full(vals.shape, pos) + jitter,
                vals,
                s=10,
                color=color,
                alpha=0.30,
                linewidths=0.0,
                zorder=2,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(list(smoothings))
    ax.set_xlabel("Downsample rate")
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, axis="y", color="#808080", alpha=0.20, linewidth=1.0)

    if title:
        ax.set_title(str(title))

    # Tight x-limits just around the group extents.
    x_left = float(x_centers[0]) - 0.75
    x_right = float(x_centers[-1]) + 0.75
    x_pad = max(0.8, 0.35 * float(len(series)))
    x_span = max(1e-9, x_right - x_left)
    right_x = x_right + float(method_label_x_offset_frac) * x_span
    ax.set_xlim(x_left, max(x_right, right_x) + x_pad)

    # Color-coded labels at the right-hand side (legend replacement).
    label_counts_by_method: Dict[str, Dict[str, int]] = {}
    raw_label_by_key: Dict[str, str] = {}
    expected_run_by_key: Dict[str, str] = {}
    for spec in series:
        base_label = labels_map.get(spec.method, spec.method)
        label0 = labels_map.get(spec.key, base_label)
        raw_label_by_key[spec.key] = label0
        expected_run = str(spec.run_tag).strip() if spec.run_tag else run_map.get(spec.method, "")
        expected_run_by_key[spec.key] = str(expected_run).strip()
        label_counts_by_method.setdefault(spec.method, {})
        label_counts_by_method[spec.method][label0] = label_counts_by_method[spec.method].get(label0, 0) + 1

    label_rows: List[Tuple[str, str, str, float]] = []
    for spec in series:
        label = raw_label_by_key.get(spec.key, labels_map.get(spec.method, spec.method))
        expected_run = expected_run_by_key.get(spec.key, "")
        if label_counts_by_method.get(spec.method, {}).get(label, 0) > 1 and expected_run:
            label = f"{label} ({expected_run})"
        vals_all: List[float] = []
        for s in smoothings:
            vals_all.extend(values_by_ms.get((spec.key, s), []))
        vals = _finite(vals_all)
        y_pos = float(np.median(vals)) if vals.size else float("nan")
        color = colors_map.get(spec.key, colors_map.get(spec.method, "#333333"))
        label_rows.append((spec.key, str(label), str(color), y_pos))

    # Rank by overall median (ties broken by label), and avoid overlaps with a placement helper.
    label_rows.sort(key=lambda t: (-(t[3] if np.isfinite(t[3]) else -1e9), t[1]))
    y_min, y_max = ax.get_ylim()
    min_sep = 0.09 * (y_max - y_min)
    _place_right_labels(
        ax,
        fig,
        x=right_x,
        y_positions=[t[3] for t in label_rows],
        labels=[t[1] for t in label_rows],
        colors=[t[2] for t in label_rows],
        min_sep=float(min_sep),
        fontsize=14,
    )

    fig.tight_layout()
    if out_path is not None:
        out_path = _to_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig, axes_list
