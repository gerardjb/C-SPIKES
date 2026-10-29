"""Owned concentration histories from the existing linear-buffer native path."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StateHistory:
    """Concentrations in M; G0–G8 are native zero-based indicator states."""

    time_s: np.ndarray
    state_M: np.ndarray
    state_order: tuple = tuple(f"G{i}" for i in range(9)) + ("BCa", "Ca", "Ca_in")
    buffer_approximation: str = "linear"
    integration_method: str = "integrateOverTime"
    internal_dt_s: float = 1e-4


def _owned_vector(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return np.array(values, dtype=np.float64, order="C", copy=True)


def simulate_state_history(model, time_s, spike_times_s, reset=True):
    """Return owned time and twelve-column concentration histories.

    Times are absolute seconds, strictly increasing with at least two samples.
    Spike times must be sorted; duplicates and out-of-window times are preserved.
    Input validation precedes model mutation. With reset=True, call the model's
    existing init(); otherwise start from its stored state. Both paths overwrite
    native histories and alter existing caches. Native integration does not store
    its final evolved state back in the model, so calls are not continuation.

    The primary native method uses the linear buffer approximation and a fixed
    100 microsecond internal step, independently of the model's evolution mode.
    Existing numerical diagnostics remain active; NaN/Inf results are returned
    unchanged. No fluorescence, normalization, files or random draws are added.
    """
    times = _owned_vector(time_s, "time_s")
    spikes = _owned_vector(spike_times_s, "spike_times_s")
    if times.size < 2 or not np.all(times[1:] > times[:-1]):
        raise ValueError("time_s must contain at least two strictly increasing samples")
    if not np.all(spikes[1:] >= spikes[:-1]):
        raise ValueError("spike_times_s must be sorted")
    if reset:
        model.init()
    model.integrateOverTime(times, spikes)
    states = model.getStates()
    concentrations = np.column_stack(
        (states["G"], states["BCa"], states["Ca"], states["Ca_in"])
    )
    return StateHistory(
        time_s=times.copy(),
        state_M=np.array(concentrations, dtype=np.float64, order="C", copy=True),
    )


@dataclass(frozen=True)
class StateOccupancies:
    """Independent raw history and separate pool fractions, never a 12-state simplex.

    Validity flags describe denominators only, not physical validity of the result.
    Ca and Ca_in remain concentration columns in history.state_M.
    """

    history: StateHistory
    indicator_fraction: np.ndarray
    buffer_bound_fraction: np.ndarray
    pool_totals_M: tuple
    valid_denominators: tuple
    pool_order: tuple = ("indicator", "endogenous_buffer")


def _occupancies_from_history(history, totals):
    import logging

    denominators = tuple(float(x) for x in totals)
    valid = tuple(bool(np.isfinite(x) and x > 0) for x in denominators)
    raw = history.state_M
    indicator = np.full(raw[:, :9].shape, np.nan, dtype=np.float64)
    buffer = np.full(raw[:, 9].shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if valid[0]:
            np.divide(raw[:, :9], denominators[0], out=indicator)
        if valid[1]:
            np.divide(raw[:, 9], denominators[1], out=buffer)
    if not all(valid):
        logging.getLogger(__name__).warning(
            "State occupancy denominator invalid: G_tot_M=%s B_tot_M=%s; "
            "affected fractions are NaN; raw concentrations preserved",
            *denominators,
        )
    return StateOccupancies(history, indicator, buffer, denominators, valid)


def simulate_state_occupancies(model, time_s, spike_times_s, reset=True):
    """Simulate using the existing history API and divide by actual model pool totals.

    G0..G8 use G_tot; BCa uses B_tot. Capture both before this invocation; no
    concurrent model mutation is supported. Invalid totals give explicit metadata,
    NaN fractions for that pool and one warning. No clipping, renormalization,
    filtering or numerical exceptions are added. Reset/history semantics remain
    those of simulate_state_history; this is not chained continuation.
    """
    totals = model.getPoolTotals()
    denominators = (totals["G_tot_M"], totals["B_tot_M"])
    history = simulate_state_history(model, time_s, spike_times_s, reset=reset)
    return _occupancies_from_history(history, denominators)
