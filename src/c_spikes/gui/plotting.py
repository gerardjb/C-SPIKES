from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from matplotlib.figure import Figure

from c_spikes.inference.types import MethodResult


METHOD_ORDER = ("pgas", "biophys_ml", "cascade", "ens2")
METHOD_COLORS = {
    "pgas": "#009E73",
    "cascade": "#F3AE14",
    "ens2": "#A6780C",
    "biophys_ml": "#007755",
}
METHOD_LABELS = {
    "pgas": r"Biophys$_{SMC}$",
    "cascade": "CASCADE",
    "ens2": r"ENS$^2$",
    "biophys_ml": r"Biophys$_{ML}$",
}


def plot_epoch(
    fig: Figure,
    *,
    time: np.ndarray,
    dff: np.ndarray,
    methods: Dict[str, MethodResult],
    method_labels: Optional[Dict[str, str]] = None,
    spike_times: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    fig.clear()
    # Use constrained layout when available; tight_layout can warn with some shared-axes setups.
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        pass
    method_keys = sorted(methods.keys(), key=_method_sort_key)
    n_rows = 1 + len(method_keys)
    gs = fig.add_gridspec(n_rows, 1, hspace=0.25)

    ax_dff = fig.add_subplot(gs[0, 0])
    ax_dff.plot(time, dff, color="black", linewidth=1.0)
    _plot_ground_truth_spikes(ax_dff, time, spike_times)
    ax_dff.set_ylabel("dF/F")
    if title:
        ax_dff.set_title(title)
    ax_dff.grid(True, alpha=0.2)

    for idx, method in enumerate(method_keys, start=1):
        ax = fig.add_subplot(gs[idx, 0], sharex=ax_dff)
        result = methods[method]
        base_method = _method_base(method)
        color = METHOD_COLORS.get(base_method, "#444444")
        ax.plot(result.time_stamps, result.spike_prob, color=color, linewidth=1.0)
        if method_labels and method in method_labels:
            ax.set_ylabel(method_labels[method])
        else:
            ax.set_ylabel(_method_label(method))
        ax.grid(True, alpha=0.2)
        _plot_discrete_spikes(ax, result)

    fig.axes[-1].set_xlabel("Time (s)")
    # Fallback spacing for matplotlib builds without constrained layout support.
    if getattr(fig, "get_layout_engine", None) is None or fig.get_layout_engine() is None:
        fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, hspace=0.25)


def _plot_discrete_spikes(ax, result: MethodResult) -> None:
    if result.discrete_spikes is None:
        return
    spikes = np.asarray(result.discrete_spikes)
    if spikes.size == 0:
        return
    mask = np.isfinite(spikes) & (spikes > 0)
    if not np.any(mask):
        return
    times = np.asarray(result.time_stamps)
    times = times[mask]
    if times.size == 0:
        return
    y_vals = np.asarray(result.spike_prob)
    y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size == 0:
        return
    y_max = float(np.nanmax(y_vals))
    y_min = float(np.nanmin(y_vals))
    height = max((y_max - y_min) * 0.1, 1e-3)
    ax.vlines(times, y_max - height, y_max, color=ax.lines[-1].get_color(), linewidth=1.5)


def _plot_ground_truth_spikes(ax, time: np.ndarray, spike_times: Optional[np.ndarray]) -> None:
    if spike_times is None:
        return
    spikes = np.asarray(spike_times, dtype=np.float64).ravel()
    if spikes.size == 0:
        return
    t_min = float(np.nanmin(time))
    t_max = float(np.nanmax(time))
    spikes = spikes[(spikes >= t_min) & (spikes <= t_max)]
    if spikes.size == 0:
        return
    for s in spikes:
        ax.axvline(float(s), color="#888888", linestyle=":", linewidth=0.6, alpha=0.9, zorder=0)


def _method_base(method_key: str) -> str:
    token = str(method_key)
    if "::" in token:
        return token.split("::", 1)[0]
    return token


def _method_variant(method_key: str) -> str:
    token = str(method_key)
    if "::" in token:
        return token.split("::", 1)[1]
    return ""


def _method_sort_key(method_key: str) -> tuple[int, str, str]:
    base = _method_base(method_key)
    try:
        idx = METHOD_ORDER.index(base)
    except ValueError:
        idx = len(METHOD_ORDER)
    return idx, base, _method_variant(method_key)


def _method_label(method_key: str) -> str:
    base = _method_base(method_key)
    variant = _method_variant(method_key)
    label = METHOD_LABELS.get(base, base)
    if not variant:
        return label
    return f"{label} | {variant}"
