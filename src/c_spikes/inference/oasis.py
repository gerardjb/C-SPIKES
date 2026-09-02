from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .cache import load_method_cache, save_method_cache
from .types import MethodResult, TrialSeries, compute_config_signature, ensure_serializable


OASIS_ADAPTER_VERSION = "2"
OASIS_NUMERICAL_REVISION = "cspikes-numerical-v2"
OASIS_DISCRETE_OUTPUT_VERSION = "binary-event-support-v1"
OASIS_SOURCE_VERSION = (
    "oasis_port-0.2.0+e738431502040ad7db8f79a12b2927ae9d2f4e7c."
    f"{OASIS_NUMERICAL_REVISION}"
)


@dataclass
class OasisConfig:
    """Configuration for per-trial OASIS deconvolution.

    ``g`` contains either one AR(1) coefficient or two AR(2) coefficients.
    An all-``None`` tuple requests per-trial estimation. OASIS coefficients are
    per-bin values, so every input trial must have the same near-uniform sample
    rate. This adapter deliberately performs no implicit resampling.

    ``discrete_mode="support"`` derives a binary event-support proxy from
    the continuous OASIS event amplitudes without changing the solver result.
    Noise-scaled thresholds are resolved independently for each trial from its
    final noise estimate and dominant fitted decay.
    """

    dataset_tag: str
    g: Tuple[Optional[float], ...] = (None,)
    sn: Optional[float] = None
    b: Optional[float] = None
    b_nonneg: bool = True
    optimize_g: int = 0
    penalty: int = 1
    decimate: int = 1
    max_iter: Optional[int] = None
    shift: Optional[int] = None
    window: Optional[int] = None
    tol: Optional[float] = None
    discrete_mode: str = "none"
    event_threshold: Optional[float] = None
    threshold_units: str = "absolute"
    downsample_label: str = "raw"
    uniformity_rtol: float = 5e-3
    uniformity_atol: float = 1e-9
    use_cache: bool = True
    cache_root: Optional[Path] = None


def _native_import_error(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "OASIS inference is unavailable because the native "
        "c_spikes.oasis.oasis_methods extension could not be imported. "
        "Rebuild or reinstall C-SPIKES with C_SPIKES_BUILD_OASIS=ON and a "
        "supported NumPy version."
    )


def _load_deconvolve() -> Callable[..., Tuple[Any, Any, Any, Any, Any]]:
    """Import the OASIS facade only when OASIS inference is requested."""

    try:
        from c_spikes.oasis.functions import deconvolve
    except (ImportError, OSError, ValueError) as exc:
        raise _native_import_error(exc) from exc
    return deconvolve


def _load_estimate_parameters() -> Callable[..., Tuple[Any, Any]]:
    """Lazily load the estimator used by the low-level facade."""

    try:
        from c_spikes.oasis.functions import estimate_parameters
    except (ImportError, OSError, ValueError) as exc:
        raise _native_import_error(exc) from exc
    return estimate_parameters


def _format_tag_token(value: str) -> str:
    token = str(value).strip().replace(" ", "_").replace(".", "p")
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", token).strip("_")
    return token or "raw"


def _finite_optional_scalar(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar or None")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar or None") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar or None")
    return result


def _normalize_requested_g(g: Sequence[Optional[float]]) -> Tuple[Optional[float], ...]:
    try:
        values = tuple(g)
    except TypeError as exc:
        raise ValueError("g must contain one AR(1) or two AR(2) coefficients") from exc
    if len(values) not in (1, 2):
        raise ValueError("g must contain one AR(1) or two AR(2) coefficients")
    missing = tuple(value is None for value in values)
    if any(missing):
        if not all(missing):
            raise ValueError("g coefficients must either all be provided or all be None")
        return tuple(None for _ in values)
    return tuple(_finite_optional_scalar(value, "g") for value in values)


def _validate_stable_g(
    g: Sequence[float],
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    coefficients = tuple(float(value) for value in g)
    if len(coefficients) == 1:
        if not 0.0 < coefficients[0] < 1.0:
            raise error_type("AR(1) g must satisfy 0 < g < 1")
        return
    roots = np.roots((1.0, -coefficients[0], -coefficients[1]))
    if (
        np.max(np.abs(roots.imag)) > 1e-12
        or np.any(roots.real <= 0.0)
        or np.any(roots.real >= 1.0)
    ):
        raise error_type("AR(2) g must have two real roots strictly between 0 and 1")


def _validate_config(config: OasisConfig) -> Tuple[Optional[float], ...]:
    if not str(config.dataset_tag).strip():
        raise ValueError("dataset_tag must be non-empty")
    if not str(config.downsample_label).strip():
        raise ValueError("downsample_label must be non-empty")
    requested_g = _normalize_requested_g(config.g)
    if requested_g[0] is not None:
        _validate_stable_g(requested_g)
    sn = _finite_optional_scalar(config.sn, "sn")
    _finite_optional_scalar(config.b, "b")
    if sn is not None and sn < 0:
        raise ValueError("sn must be non-negative")
    if not isinstance(config.b_nonneg, (bool, np.bool_)):
        raise TypeError("b_nonneg must be a boolean")
    if not isinstance(config.use_cache, (bool, np.bool_)):
        raise TypeError("use_cache must be a boolean")
    if (
        isinstance(config.optimize_g, (bool, np.bool_))
        or not isinstance(config.optimize_g, (int, np.integer))
        or config.optimize_g < 0
    ):
        raise ValueError("optimize_g must be a non-negative integer")
    if (
        isinstance(config.penalty, (bool, np.bool_))
        or not isinstance(config.penalty, (int, np.integer))
        or config.penalty not in (0, 1)
    ):
        raise ValueError("penalty must be either 0 (L0) or 1 (L1)")
    if (
        isinstance(config.decimate, (bool, np.bool_))
        or not isinstance(config.decimate, (int, np.integer))
        or config.decimate < 1
    ):
        raise ValueError("decimate must be a positive integer")
    for name, value in (
        ("max_iter", config.max_iter),
        ("shift", config.shift),
        ("window", config.window),
    ):
        if value is None:
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    tol = _finite_optional_scalar(config.tol, "tol")
    if tol is not None and tol <= 0:
        raise ValueError("tol must be positive")
    if (
        not isinstance(config.discrete_mode, str)
        or config.discrete_mode not in {"none", "support"}
    ):
        raise ValueError("discrete_mode must be either 'none' or 'support'")
    if (
        not isinstance(config.threshold_units, str)
        or config.threshold_units not in {"absolute", "noise_scaled"}
    ):
        raise ValueError("threshold_units must be either 'absolute' or 'noise_scaled'")
    event_threshold = _finite_optional_scalar(config.event_threshold, "event_threshold")
    if config.discrete_mode == "none":
        if event_threshold is not None:
            raise ValueError("event_threshold must be None when discrete_mode is 'none'")
        if config.threshold_units != "absolute":
            raise ValueError("threshold_units must be 'absolute' when discrete_mode is 'none'")
    elif event_threshold is None or event_threshold <= 0:
        raise ValueError(
            "event_threshold must be a positive finite scalar when discrete_mode is 'support'"
        )
    for name, value in (
        ("uniformity_rtol", config.uniformity_rtol),
        ("uniformity_atol", config.uniformity_atol),
    ):
        normalized = _finite_optional_scalar(value, name)
        if normalized is None or normalized < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if len(requested_g) == 1 and any(
        value is not None for value in (config.shift, config.window, config.tol)
    ):
        raise ValueError("shift, window, and tol are only supported by the AR(2) backend")
    return requested_g


def _prepare_trials(
    trials: Sequence[TrialSeries],
    config: OasisConfig,
) -> Tuple[list[TrialSeries], list[float], list[float]]:
    if not trials:
        raise ValueError("OASIS inference requires at least one trial")

    prepared: list[TrialSeries] = []
    sampling_rates: list[float] = []
    sample_intervals: list[float] = []
    for trial_index, trial in enumerate(trials):
        try:
            raw_times = np.asarray(trial.times)
            raw_values = np.asarray(trial.values)
            if np.iscomplexobj(raw_times) or np.iscomplexobj(raw_values):
                raise ValueError
            times = np.asarray(raw_times, dtype=np.float64)
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"OASIS trial {trial_index} times and values must be real numeric arrays"
            ) from exc
        if times.ndim != 1 or values.ndim != 1:
            raise ValueError(f"OASIS trial {trial_index} times and values must be one-dimensional")
        if times.size != values.size:
            raise ValueError(f"OASIS trial {trial_index} times and values must have equal lengths")
        if times.size < 3:
            raise ValueError(f"OASIS trial {trial_index} must contain at least three samples")
        if not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError(f"OASIS trial {trial_index} must contain only finite values")

        with np.errstate(over="ignore", invalid="ignore"):
            diffs = np.diff(times)
        if not np.isfinite(diffs).all() or np.any(diffs <= 0):
            raise ValueError(f"OASIS trial {trial_index} timestamps must be strictly increasing")
        dt = float(np.median(diffs))
        if not np.allclose(
            diffs,
            dt,
            rtol=float(config.uniformity_rtol),
            atol=float(config.uniformity_atol),
        ):
            raise ValueError(
                f"OASIS trial {trial_index} is not uniformly sampled; "
                "resample it explicitly before inference"
            )

        prepared.append(
            TrialSeries(times=times.copy(), values=np.asarray(values, dtype=np.float64).copy())
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sampling_rate = float(1.0 / dt)
        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError(f"OASIS trial {trial_index} has an invalid sampling rate")
        sample_intervals.append(dt)
        sampling_rates.append(sampling_rate)

    shortest_trial = min(trial.values.size for trial in prepared)
    if config.decimate > shortest_trial:
        raise ValueError("decimate cannot exceed the shortest trial length")

    reference_dt = sample_intervals[0]
    for trial_index, dt in enumerate(sample_intervals[1:], start=1):
        if not np.isclose(
            dt,
            reference_dt,
            rtol=float(config.uniformity_rtol),
            atol=float(config.uniformity_atol),
        ):
            raise ValueError(
                "OASIS requires a consistent sampling rate across trials; "
                f"trial 0 is {1.0 / reference_dt:g} Hz and trial {trial_index} is {1.0 / dt:g} Hz"
            )
    return prepared, sampling_rates, sample_intervals


def _hash_trials(trials: Sequence[TrialSeries]) -> str:
    """Hash exact inference inputs while preserving trial boundaries."""

    digest = hashlib.sha256()
    digest.update(b"c-spikes-oasis-trials-v1\0")
    digest.update(np.asarray([len(trials)], dtype="<i8").tobytes())
    for trial in trials:
        times = np.ascontiguousarray(trial.times, dtype="<f8")
        values = np.ascontiguousarray(trial.values, dtype="<f8")
        digest.update(np.asarray([times.size], dtype="<i8").tobytes())
        digest.update(times.tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _resolve_trial_parameters(
    values: np.ndarray,
    requested_g: Tuple[Optional[float], ...],
    config: OasisConfig,
    estimate_parameters: Optional[Callable[..., Tuple[Any, Any]]] = None,
) -> Tuple[Tuple[float, ...], float]:
    estimate_g = requested_g[0] is None
    estimate_sn = config.sn is None
    estimated_g: Any = None
    estimated_sn: Any = None
    if estimate_g or estimate_sn:
        if estimate_parameters is None:
            raise RuntimeError("Internal OASIS parameter-estimation function is unavailable")
        working_values = values if config.b is None else values - float(config.b)
        fudge_factor = 0.97 if config.optimize_g and len(requested_g) == 1 else 0.98
        estimated_g, estimated_sn = estimate_parameters(
            working_values,
            p=len(requested_g),
            fudge_factor=fudge_factor,
        )

    if estimate_g:
        final_g = tuple(float(value) for value in np.asarray(estimated_g).reshape(-1))
    else:
        final_g = tuple(float(value) for value in requested_g)
    if len(final_g) != len(requested_g) or not np.isfinite(final_g).all():
        raise ValueError("OASIS produced invalid estimated AR coefficients")
    _validate_stable_g(final_g)

    final_sn = float(estimated_sn) if estimate_sn else float(config.sn)
    if not np.isfinite(final_sn) or final_sn < 0:
        raise ValueError("OASIS produced an invalid noise estimate")
    return final_g, final_sn


def _validate_estimation_trace(
    values: np.ndarray,
    requested_g: Tuple[Optional[float], ...],
    baseline: Optional[float],
) -> None:
    """Mirror the facade's preconditions before importing the estimator."""

    minimum_length = 11 + len(requested_g)
    if values.size < minimum_length:
        raise ValueError(
            f"Automatic parameter estimation for AR({len(requested_g)}) requires at least "
            f"{minimum_length} samples"
        )
    working_values = values if baseline is None else values - float(baseline)
    scale = max(1.0, float(np.max(np.abs(working_values))))
    if np.std(working_values) <= np.finfo(np.float64).eps * scale:
        raise ValueError("Automatic parameter estimation requires a non-constant trace")


def _public_config(
    config: OasisConfig,
    requested_g: Tuple[Optional[float], ...],
) -> Dict[str, Any]:
    public_config: Dict[str, Any] = {
        "dataset_tag": str(config.dataset_tag),
        "g": requested_g,
        "sn": None if config.sn is None else float(config.sn),
        "b": None if config.b is None else float(config.b),
        "b_nonneg": bool(config.b_nonneg),
        "optimize_g": int(config.optimize_g),
        "penalty": int(config.penalty),
        "decimate": int(config.decimate),
        "max_iter": None if config.max_iter is None else int(config.max_iter),
        "shift": None if config.shift is None else int(config.shift),
        "window": None if config.window is None else int(config.window),
        "tol": None if config.tol is None else float(config.tol),
        "downsample_label": str(config.downsample_label),
        "uniformity_rtol": float(config.uniformity_rtol),
        "uniformity_atol": float(config.uniformity_atol),
        "preprocessing": "none",
        "use_cache": bool(config.use_cache),
        "cache_root": None if config.cache_root is None else str(config.cache_root),
    }
    # Preserve the public metadata and cache identity of the legacy/default
    # continuous-only configuration. Discretization keys are inference-affecting
    # only when the opt-in support train is requested.
    if config.discrete_mode == "support":
        public_config.update(
            {
                "discrete_mode": "support",
                "event_threshold": float(config.event_threshold),
                "threshold_units": str(config.threshold_units),
            }
        )
    return public_config


def _solver_kwargs(config: OasisConfig, ar_order: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "b_nonneg": bool(config.b_nonneg),
        "optimize_g": int(config.optimize_g),
        "penalty": int(config.penalty),
        "decimate": int(config.decimate),
    }
    if config.max_iter is not None:
        kwargs["max_iter"] = int(config.max_iter)
    if ar_order == 2:
        if config.shift is not None:
            kwargs["shift"] = int(config.shift)
        if config.window is not None:
            kwargs["window"] = int(config.window)
        if config.tol is not None:
            kwargs["tol"] = float(config.tol)
    return kwargs


def _normalize_fitted_g(g: Any, ar_order: int) -> Tuple[float, ...]:
    values = tuple(float(value) for value in np.asarray(g).reshape(-1))
    if len(values) != ar_order or not np.isfinite(values).all():
        raise RuntimeError("OASIS returned invalid AR coefficients")
    _validate_stable_g(values, error_type=RuntimeError)
    return values


def _dominant_decay(g: Sequence[float]) -> float:
    """Return the slowest stable per-bin decay represented by AR coefficients."""

    coefficients = tuple(float(value) for value in g)
    if len(coefficients) == 1:
        return coefficients[0]
    roots = np.roots((1.0, -coefficients[0], -coefficients[1]))
    return float(np.max(roots.real))


def _resolve_event_threshold(
    config: OasisConfig,
    *,
    final_g: Sequence[float],
    final_sn: float,
) -> float:
    """Resolve an event-amplitude threshold for one fitted trial."""

    requested = float(config.event_threshold)
    if config.threshold_units == "absolute":
        resolved = requested
    else:
        dominant_decay = _dominant_decay(final_g)
        resolved = requested * float(final_sn) * np.sqrt(1.0 - dominant_decay)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            "OASIS event threshold must resolve to a positive finite value; "
            "noise_scaled thresholds require a positive fitted noise estimate, "
            "so use absolute units when sn is zero"
        )
    return float(resolved)


def run_oasis_inference(
    trials: Sequence[TrialSeries],
    config: OasisConfig,
) -> MethodResult:
    """Run OASIS independently on uniformly sampled trials."""

    requested_g = _validate_config(config)
    prepared_trials, sampling_rates, _ = _prepare_trials(list(trials), config)
    trace_hash = _hash_trials(prepared_trials)

    cache_tag = (
        f"{_format_tag_token(config.dataset_tag)}_"
        f"s{_format_tag_token(config.downsample_label)}"
    )
    public_config = _public_config(config, requested_g)
    cache_config: Dict[str, Any] = {
        key: value
        for key, value in public_config.items()
        if key not in {"use_cache", "cache_root"}
    }
    cache_config.update(
        {
            "adapter_version": OASIS_ADAPTER_VERSION,
            "source_version": OASIS_SOURCE_VERSION,
            "ar_order": len(requested_g),
            "g_strategy": (
                "estimate_per_trial_deterministic_v1"
                if requested_g[0] is None
                else "provided_per_bin"
            ),
            "sn_strategy": (
                "estimate_per_trial_psd_v1" if config.sn is None else "provided"
            ),
            "input_signature": trace_hash,
            "trial_lengths": [int(trial.values.size) for trial in prepared_trials],
            "sampling_rates": sampling_rates,
        }
    )
    if config.discrete_mode == "support":
        cache_config["discrete_output_version"] = OASIS_DISCRETE_OUTPUT_VERSION
    cache_key, _ = compute_config_signature(cache_config)

    if config.use_cache:
        try:
            cached = load_method_cache(
                "oasis",
                cache_tag,
                cache_config,
                trace_hash,
                cache_root=config.cache_root,
            )
        except OSError:
            cached = None
        if cached is not None:
            cached.metadata["config"] = ensure_serializable(public_config)
            cached.metadata.setdefault("cache_tag", cache_tag)
            cached.metadata.setdefault("cache_key", cache_key)
            cached.metadata["cache_hit"] = True
            return cached

    requires_estimation = requested_g[0] is None or config.sn is None
    if requires_estimation:
        for trial in prepared_trials:
            _validate_estimation_trace(trial.values, requested_g, config.b)
    estimate_parameters = _load_estimate_parameters() if requires_estimation else None
    resolved_parameters = [
        _resolve_trial_parameters(
            trial.values,
            requested_g,
            config,
            estimate_parameters=estimate_parameters,
        )
        for trial in prepared_trials
    ]
    deconvolve = _load_deconvolve()
    time_segments: list[np.ndarray] = []
    spike_segments: list[np.ndarray] = []
    reconstruction_segments: list[np.ndarray] = []
    discrete_segments: list[np.ndarray] = []
    trial_metadata: list[Dict[str, Any]] = []

    for trial_index, (trial, sampling_rate, parameters) in enumerate(
        zip(prepared_trials, sampling_rates, resolved_parameters)
    ):
        initial_g, final_sn = parameters
        c, s, fitted_b, fitted_g, lam = deconvolve(
            trial.values,
            g=initial_g,
            sn=final_sn,
            b=config.b,
            **_solver_kwargs(config, len(initial_g)),
        )
        calcium = np.asarray(c, dtype=np.float64)
        spikes = np.asarray(s, dtype=np.float64)
        if calcium.ndim != 1 or spikes.ndim != 1:
            raise RuntimeError(f"OASIS returned non-vector output for trial {trial_index}")
        if calcium.size != trial.values.size or spikes.size != trial.values.size:
            raise RuntimeError(f"OASIS returned misaligned output for trial {trial_index}")
        if not np.isfinite(calcium).all() or not np.isfinite(spikes).all():
            raise RuntimeError(f"OASIS returned non-finite output for trial {trial_index}")
        spike_scale = max(1.0, float(np.max(np.abs(spikes))))
        negative_tolerance = 64.0 * np.finfo(np.float64).eps * spike_scale
        if np.any(spikes < -negative_tolerance):
            raise RuntimeError(f"OASIS returned negative event amplitudes for trial {trial_index}")
        if np.any(spikes < 0.0):
            spikes = np.maximum(spikes, 0.0)

        fitted_b = float(fitted_b)
        lam = float(lam)
        if not np.isfinite(fitted_b) or not np.isfinite(lam):
            raise RuntimeError(f"OASIS returned invalid scalar output for trial {trial_index}")
        final_g = _normalize_fitted_g(fitted_g, len(initial_g))
        backend = (
            "compiled_constrained_oasisAR1"
            if len(initial_g) == 1
            else "oasis_python_constrained_onnlsAR2"
        )

        discretization: Optional[Dict[str, Any]] = None
        if config.discrete_mode == "support":
            resolved_threshold = _resolve_event_threshold(
                config,
                final_g=final_g,
                final_sn=final_sn,
            )
            support = np.asarray(spikes >= resolved_threshold, dtype=np.uint8)
            discrete_segments.append(support)
            discretization = {
                "semantics": "binary_event_support",
                "requested_threshold": float(config.event_threshold),
                "threshold_units": str(config.threshold_units),
                "resolved_threshold": resolved_threshold,
                "comparison": "s >= resolved_threshold",
                "event_count": int(np.count_nonzero(support)),
                "max_events_per_bin": 1,
            }

        time_segments.append(trial.times)
        spike_segments.append(spikes)
        reconstruction_segments.append(calcium + fitted_b)
        trial_entry: Dict[str, Any] = {
            "index": trial_index,
            "length": int(trial.values.size),
            "b": fitted_b,
            "g": final_g,
            "sn": final_sn,
            "lam": lam,
            "sampling_rate": sampling_rate,
            "solver": "deconvolve",
            "backend": backend,
        }
        if discretization is not None:
            trial_entry["discretization"] = discretization
        trial_metadata.append(trial_entry)

    times = np.concatenate(time_segments)
    spikes = np.concatenate(spike_segments)
    reconstruction = np.concatenate(reconstruction_segments)
    discrete = np.concatenate(discrete_segments) if discrete_segments else None
    order = np.argsort(times, kind="stable")
    metadata: Dict[str, Any] = {
        "config": ensure_serializable(public_config),
        "source_version": OASIS_SOURCE_VERSION,
        "adapter_version": OASIS_ADAPTER_VERSION,
        "cache_tag": cache_tag,
        "cache_key": cache_key,
        "cache_hit": False,
        "trials": ensure_serializable(trial_metadata),
    }
    if discrete is not None:
        metadata["discretization"] = {
            "mode": "support",
            "semantics": "binary_event_support",
            "version": OASIS_DISCRETE_OUTPUT_VERSION,
            "requested_threshold": float(config.event_threshold),
            "threshold_units": str(config.threshold_units),
            "comparison": "s >= resolved_threshold",
            "event_count": int(np.count_nonzero(discrete)),
            "max_events_per_bin": 1,
        }
    result = MethodResult(
        name="oasis",
        time_stamps=times[order],
        spike_prob=spikes[order],
        sampling_rate=float(np.median(sampling_rates)),
        metadata=metadata,
        reconstruction=reconstruction[order],
        discrete_spikes=None if discrete is None else discrete[order],
    )

    if config.use_cache:
        try:
            save_method_cache(
                "oasis",
                cache_tag,
                result,
                cache_config,
                trace_hash,
                cache_root=config.cache_root,
            )
        except OSError:
            # Inference remains useful in read-only/shared environments.
            pass
    return result
