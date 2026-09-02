from __future__ import annotations

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from c_spikes.gui.plotting import METHOD_ORDER, plot_epoch
from c_spikes.inference.types import MethodResult


def _result(name: str, *, discrete_spikes=None) -> MethodResult:
    return MethodResult(
        name=name,
        time_stamps=np.asarray([0.0, 0.1, 0.2]),
        spike_prob=np.asarray([0.0, 0.4, 0.1]),
        sampling_rate=10.0,
        discrete_spikes=discrete_spikes,
    )


def test_oasis_plot_order_label_color_and_continuous_only_output() -> None:
    assert METHOD_ORDER == ("pgas", "oasis", "biophys_ml", "cascade", "ens2")

    fig = Figure(figsize=(4.0, 4.0))
    plot_epoch(
        fig,
        time=np.asarray([0.0, 0.1, 0.2]),
        dff=np.asarray([0.1, 0.2, 0.15]),
        methods={
            "unknown": _result("unknown"),
            "ens2": _result("ens2"),
            "oasis::L1": _result("oasis"),
            "pgas": _result("pgas"),
        },
    )

    method_axes = fig.axes[1:]
    assert [ax.get_ylabel() for ax in method_axes] == [
        r"Biophys$_{SMC}$",
        "OASIS | L1",
        r"ENS$^2$",
        "unknown",
    ]

    oasis_ax = method_axes[1]
    assert to_hex(oasis_ax.lines[0].get_color()) == "#cc79a7"
    assert not oasis_ax.collections


def test_oasis_binary_event_support_draws_event_markers() -> None:
    fig = Figure(figsize=(4.0, 3.0))
    plot_epoch(
        fig,
        time=np.asarray([0.0, 0.1, 0.2]),
        dff=np.asarray([0.1, 0.2, 0.15]),
        methods={
            "oasis": _result(
                "oasis", discrete_spikes=np.asarray([0.0, 1.0, 0.0])
            )
        },
    )

    oasis_ax = fig.axes[1]
    assert len(oasis_ax.collections) == 1
    segments = oasis_ax.collections[0].get_segments()
    assert len(segments) == 1
    np.testing.assert_allclose(segments[0][:, 0], np.asarray([0.1, 0.1]))
