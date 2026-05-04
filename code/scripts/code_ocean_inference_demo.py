#!/usr/bin/env python3
"""Run the Code Ocean manuscript inference parity demonstration.

This stage intentionally reuses the repository's existing batch runner and
plotting tools. The defaults are small enough for a capsule smoke demo while
remaining easy to expand by increasing trial counts or adding stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


JG8F_NOTEBOOK_DATASET = "jGCaMP8f_ANM471994_cell05"
JG8F_BIOPHYS_ML_SOURCE_DATASET = "jGCaMP8f_ANM478349_cell04"
JG8M_NOTEBOOK_DATASET = "jGCaMP8m_ANM472179_cell02"

RUN_JG8F_BASE = "code_ocean_jg8f_base"
RUN_JG8F_PARAMS = "code_ocean_jg8f_params"
RUN_JG8F_BIOPHYS_ML_CACHE = "code_ocean_jg8f_biophys_ml_cache"
RUN_JG8M_BASE = "code_ocean_jg8m_base"
RUN_JG8M_BIOPHYS_ML_CACHE = "code_ocean_jg8m_biophys_ml_cache"

DEFAULT_JG8F_BM_SIGMA = "0.03"
DEFAULT_JG8M_BM_SIGMA = "0.05"


@dataclass(frozen=True)
class ParityMap:
    context: str
    manuscript_figure: str
    generated_csv: str
    reference_csv: str
    generated_run: str
    generated_method: str
    reference_run: str
    reference_method: str
    display_method: str


PARITY_MAPS: tuple[ParityMap, ...] = (
    ParityMap(
        "cell2_jg8f_full_dataset",
        "Figure 4B,D; Supplementary Figure 12A,B",
        "trialwise_correlations_jG8f_repro.csv",
        "trialwise_correlations_jG8f.csv",
        RUN_JG8F_BASE,
        "pgas",
        "refbuild_jg8f_gold",
        "pgas",
        "BiophysSMC",
    ),
    ParityMap(
        "cell2_jg8f_full_dataset",
        "Figure 4B,D; Supplementary Figure 12A,B",
        "trialwise_correlations_jG8f_repro.csv",
        "trialwise_correlations_jG8f.csv",
        RUN_JG8F_BASE,
        "ens2",
        "refbuild_jg8f_gold",
        "ens2",
        "Published ENS2",
    ),
    ParityMap(
        "cell2_jg8f_full_dataset",
        "Figure 4B,D; Supplementary Figure 12A,B",
        "trialwise_correlations_jG8f_repro.csv",
        "trialwise_correlations_jG8f.csv",
        RUN_JG8F_BASE,
        "cascade",
        "refbuild_jg8f_gold",
        "cascade",
        "Published CASCADE",
    ),
    ParityMap(
        "cell2_jg8f_full_dataset",
        "Figure 4B,D; Supplementary Figure 12A,B",
        "trialwise_correlations_jG8f_repro.csv",
        "trialwise_correlations_jG8f.csv",
        RUN_JG8F_BASE,
        "biophys_ml",
        "refbuild_jg8f_biophysml",
        "ens2",
        "BiophysML",
    ),
    ParityMap(
        "cell3_jg8f_parameterization",
        "Supplementary Figure 13C,D",
        "trialwise_correlations_jG8f_repro.csv",
        "trialwise_correlations_jG8f.csv",
        RUN_JG8F_BASE,
        "pgas",
        "refbuild_jg8f_gold",
        "pgas",
        "BiophysSMC default parameters",
    ),
    ParityMap(
        "cell3_jg8f_parameterization",
        "Supplementary Figure 13C,D",
        "trialwise_correlations_jG8f_repro.csv",
        "trialwise_correlations_jG8f.csv",
        RUN_JG8F_PARAMS,
        "pgas",
        "refbuild_jg8f_janelia_params",
        "pgas",
        "BiophysSMC jGCaMP8f parameters",
    ),
    ParityMap(
        "cell4_jg8m_full_dataset",
        "Supplementary Figure 14F,G",
        "trialwise_correlations_jG8m_repro.csv",
        "trialwise_correlations_jG8m.csv",
        RUN_JG8M_BASE,
        "pgas",
        "refbuild_jg8m_bm0p05_biophysml_full",
        "pgas",
        "BiophysSMC",
    ),
    ParityMap(
        "cell4_jg8m_full_dataset",
        "Supplementary Figure 14F,G",
        "trialwise_correlations_jG8m_repro.csv",
        "trialwise_correlations_jG8m.csv",
        RUN_JG8M_BASE,
        "ens2",
        "refbuild_jg8m_bm0p05_biophysml_full",
        "ens2",
        "Published ENS2",
    ),
    ParityMap(
        "cell4_jg8m_full_dataset",
        "Supplementary Figure 14F,G",
        "trialwise_correlations_jG8m_repro.csv",
        "trialwise_correlations_jG8m.csv",
        RUN_JG8M_BASE,
        "cascade",
        "refbuild_jg8m_bm0p05_biophysml_full",
        "cascade",
        "Published CASCADE",
    ),
    ParityMap(
        "cell4_jg8m_full_dataset",
        "Supplementary Figure 14F,G",
        "trialwise_correlations_jG8m_repro.csv",
        "trialwise_correlations_jG8m.csv",
        RUN_JG8M_BASE,
        "biophys_ml",
        "refbuild_jg8m_bm0p05_biophysml_full",
        "biophys_ml",
        "BiophysML",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--corr-sigma-ms",
        type=float,
        default=float(os.environ.get("C_SPIKES_INFERENCE_CORR_SIGMA_MS", "30.0")),
        help="Correlation smoothing sigma used for parity CSVs and plots.",
    )
    parser.add_argument(
        "--extra-random-epochs",
        type=int,
        default=int(os.environ.get("C_SPIKES_INFERENCE_EXTRA_RANDOM_EPOCHS", "2")),
        help=(
            "Number of deterministic extra epochs to include from the BiophysML source "
            "dataset in addition to the source epoch."
        ),
    )
    parser.add_argument(
        "--dataset-percent",
        type=float,
        default=(
            float(os.environ["C_SPIKES_DATASET_PERCENT"])
            if os.environ.get("C_SPIKES_DATASET_PERCENT")
            else None
        ),
        help=(
            "Optional percent of the full eligible excitatory epoch list to run. "
            "If omitted, only curated default epochs are processed. 100 means all eligible epochs."
        ),
    )
    parser.add_argument(
        "--jg8f-bm-sigma",
        default=os.environ.get("C_SPIKES_JG8F_BM_SIGMA", DEFAULT_JG8F_BM_SIGMA),
        help="Fixed PGAS bm_sigma for jGCaMP8f runs, or 'auto'.",
    )
    parser.add_argument(
        "--jg8m-bm-sigma",
        default=os.environ.get("C_SPIKES_JG8M_BM_SIGMA", DEFAULT_JG8M_BM_SIGMA),
        help="Fixed PGAS bm_sigma for jGCaMP8m runs, or 'auto'.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=int(os.environ.get("C_SPIKES_INFERENCE_RANDOM_SEED", "20260428")),
    )
    parser.add_argument(
        "--notebook-trial",
        type=int,
        default=int(os.environ.get("C_SPIKES_INFERENCE_NOTEBOOK_TRIAL", "1")),
        help="0-based trial used for notebook trace-panel parity windows.",
    )
    parser.add_argument(
        "--include-all-notebook-trials",
        action="store_true",
        default=os.environ.get("C_SPIKES_INFERENCE_ALL_NOTEBOOK_TRIALS", "0") == "1",
        help="Run all edged notebook trials rather than the single plotted notebook trial.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        default=os.environ.get("C_SPIKES_INFERENCE_PLAN_ONLY", "0") == "1",
        help="Write selection/plan files without launching inference.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=os.environ.get("C_SPIKES_INFERENCE_SKIP_PLOTS", "0") == "1",
    )
    parser.add_argument(
        "--skip-pgas",
        action="store_true",
        default=os.environ.get("C_SPIKES_INFERENCE_SKIP_PGAS", "0") == "1",
        help="Skip PGAS-heavy runs for debugging the non-PGAS wiring.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _capsule_root() -> Path:
    return _repo_root()


def _load_edges(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True).item()
    return {str(k): np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def _bounded_trial(dataset: str, requested: int, edges: Mapping[str, np.ndarray]) -> int:
    n = int(np.asarray(edges[dataset]).shape[0])
    if n <= 0:
        raise ValueError(f"No edged trials available for {dataset}")
    return min(max(int(requested), 0), n - 1)


def _selection_for_dataset(
    dataset: str,
    *,
    edges: Mapping[str, np.ndarray],
    preferred: Iterable[int],
) -> list[int]:
    n = int(np.asarray(edges[dataset]).shape[0])
    out: list[int] = []
    for raw_idx in preferred:
        idx = int(raw_idx)
        if 0 <= idx < n and idx not in out:
            out.append(idx)
    if not out:
        out = [0]
    return out


def _valid_trial_indices(dataset: str, edges: Mapping[str, np.ndarray]) -> list[int]:
    arr = np.asarray(edges[dataset], dtype=np.float64)
    out: list[int] = []
    for idx, row in enumerate(arr):
        if row.shape[0] >= 2 and np.isfinite(row[0]) and np.isfinite(row[1]) and float(row[1]) > float(row[0]):
            out.append(int(idx))
    return out


def _eligible_epoch_pairs(data_root: Path, edges: Mapping[str, np.ndarray]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    available = {path.stem for path in sorted(data_root.glob("*.mat"))}
    for dataset in sorted(set(edges).intersection(available)):
        for trial_idx in _valid_trial_indices(dataset, edges):
            pairs.append((dataset, int(trial_idx)))
    return pairs


def _selection_from_pairs(pairs: Sequence[tuple[str, int]]) -> dict[str, list[int]]:
    selection: dict[str, list[int]] = {}
    for dataset, trial_idx in pairs:
        bucket = selection.setdefault(str(dataset), [])
        if int(trial_idx) not in bucket:
            bucket.append(int(trial_idx))
    return {dataset: trials for dataset, trials in selection.items() if trials}


def _selection_for_percent(
    *,
    data_root: Path,
    edges: Mapping[str, np.ndarray],
    percent: float,
) -> dict[str, list[int]]:
    if not np.isfinite(float(percent)) or float(percent) <= 0.0 or float(percent) > 100.0:
        raise ValueError("--dataset-percent must be > 0 and <= 100.")
    pairs = _eligible_epoch_pairs(data_root, edges)
    if not pairs:
        raise ValueError(f"No eligible edged epochs found under {data_root}")
    n = int(math.ceil(len(pairs) * float(percent) / 100.0))
    n = max(1, min(len(pairs), n))
    return _selection_from_pairs(pairs[:n])


def _random_extra_trials(
    dataset: str,
    *,
    edges: Mapping[str, np.ndarray],
    required: int,
    count: int,
    seed: int,
) -> list[int]:
    n = int(np.asarray(edges[dataset]).shape[0])
    candidates = [idx for idx in range(n) if idx != int(required)]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: max(0, int(count))]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str], dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[inference-demo] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(cwd), env=dict(env), check=True)


def _base_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    if env.get("C_SPIKES_USE_SOURCE_TREE") == "1":
        src = str(repo_root / "code" / "src")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("C_SPIKES_TF_SUPPRESS_RUNTIME_STDERR", "1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    return env


def _append_nvidia_libs_to_env(env: dict[str, str]) -> None:
    code = (
        "import glob, site; "
        "print(':'.join(sorted(p for root in site.getsitepackages() "
        "for p in glob.glob(root + '/nvidia/*/lib'))))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
    except Exception:
        return
    libs = proc.stdout.strip()
    if libs:
        env["LD_LIBRARY_PATH"] = f"{libs}{os.pathsep}{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else libs


def _run_batch(
    *,
    cwd: Path,
    env: Mapping[str, str],
    dry_run: bool,
    data_root: Path,
    datasets: Sequence[str],
    selection_json: Path,
    output_root: Path,
    cache_root: Path,
    edges_path: Path,
    run_tag: str,
    methods: Sequence[str],
    smoothing_levels: Sequence[str],
    corr_sigma_ms: float,
    pgas_constants: Path,
    pgas_gparam: Path,
    pgas_output_root: Path,
    pgas_bm_sigma: str,
    ens2_pretrained_root: Path,
    cascade_model_root: Path,
    cascade_model_name: str = "universal_p_cascade_exc_30",
    eval_only: bool = False,
) -> None:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "c_spikes.cli.run",
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--cache-root",
        str(cache_root),
        "--edges-path",
        str(edges_path),
        "--trial-selection-path",
        str(selection_json),
        "--corr-sigma-ms",
        str(float(corr_sigma_ms)),
        "--trialwise-correlations",
        "--run-tag",
        run_tag,
        "--cascade-no-discrete",
        "--cascade-model-root",
        str(cascade_model_root),
        "--cascade-model-name",
        cascade_model_name,
        "--ens2-pretrained-root",
        str(ens2_pretrained_root),
        "--pgas-constants",
        str(pgas_constants),
        "--pgas-gparam",
        str(pgas_gparam),
        "--pgas-output-root",
        str(pgas_output_root),
        "--pgas-bm-sigma",
        str(pgas_bm_sigma),
    ]
    if eval_only:
        cmd.append("--eval-only")
    for method in methods:
        cmd.extend(["--method", method])
    for smoothing in smoothing_levels:
        cmd.extend(["--smoothing-level", smoothing])
    for dataset in datasets:
        cmd.extend(["--dataset", dataset])
    _run(cmd, cwd=cwd, env=env, dry_run=dry_run)


def _import_ens2_cache_as_biophys_ml(
    *,
    cwd: Path,
    env: Mapping[str, str],
    dry_run: bool,
    eval_root: Path,
    cache_root: Path,
    data_root: Path,
    source_run: str,
    target_run: str,
    model_name: str,
) -> None:
    code = r"""
import json
import os
from pathlib import Path

from c_spikes.inference.import_external import import_external_method
from c_spikes.inference.types import compute_config_signature

eval_root = Path(os.environ["C_SPIKES_IMPORT_EVAL_ROOT"])
cache_root = Path(os.environ["C_SPIKES_IMPORT_CACHE_ROOT"])
data_root = Path(os.environ["C_SPIKES_IMPORT_DATA_ROOT"])
source_run = os.environ["C_SPIKES_IMPORT_SOURCE_RUN"]
target_run = os.environ["C_SPIKES_IMPORT_TARGET_RUN"]
model_name = os.environ["C_SPIKES_IMPORT_MODEL_NAME"]

count = 0
for comparison_path in sorted((eval_root / source_run).glob("*/*/comparison.json")):
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    dataset = str(comparison["dataset"])
    smoothing = str(comparison["smoothing"])
    ens2_entries = [entry for entry in comparison.get("methods", []) if entry.get("method") == "ens2"]
    if len(ens2_entries) != 1:
        raise RuntimeError(f"Expected exactly one ENS2 entry in {comparison_path}, got {len(ens2_entries)}")
    entry = ens2_entries[0]
    cache_tag = str(entry.get("cache_tag") or dataset).strip()
    config = dict(entry.get("config") or {})
    cache_key_raw = entry.get("cache_key")
    cache_key = str(cache_key_raw).strip() if cache_key_raw is not None else ""
    if not cache_key:
        cache_key, _ = compute_config_signature(config)
    pred_path = cache_root / "ens2" / cache_tag / f"{cache_key}.mat"
    config["source_method"] = "ens2"
    config["source_run_tag"] = source_run
    config["source_cache_tag"] = cache_tag
    config["source_cache_key"] = cache_key
    config["method_alias"] = "biophys_ml"
    import_external_method(
        pred_path=pred_path,
        method="biophys_ml",
        dataset=dataset,
        smoothing=smoothing,
        run_tag=target_run,
        data_root=data_root,
        eval_root=eval_root,
        cache_root=cache_root,
        cache_tag=f"{model_name}/{dataset}",
        label="biophys_ml",
        config=config,
        corr_sigma_ms=50.0,
    )
    count += 1
print(f"[inference-demo] imported {count} ENS2 cache files as biophys_ml into {target_run}")
"""
    import_env = dict(env)
    import_env.update(
        {
            "C_SPIKES_IMPORT_EVAL_ROOT": str(eval_root),
            "C_SPIKES_IMPORT_CACHE_ROOT": str(cache_root),
            "C_SPIKES_IMPORT_DATA_ROOT": str(data_root),
            "C_SPIKES_IMPORT_SOURCE_RUN": str(source_run),
            "C_SPIKES_IMPORT_TARGET_RUN": str(target_run),
            "C_SPIKES_IMPORT_MODEL_NAME": str(model_name),
        }
    )
    _run([sys.executable, "-c", code], cwd=cwd, env=import_env, dry_run=dry_run)


def _trialwise_csv(
    *,
    cwd: Path,
    env: Mapping[str, str],
    dry_run: bool,
    eval_root: Path,
    data_root: Path,
    edges_path: Path,
    out_csv: Path,
    runs: Sequence[str],
    datasets: Sequence[str],
    corr_sigma_ms: float,
) -> None:
    cmd: list[str] = [
        sys.executable,
        "code/scripts/trialwise_correlations.py",
        "--eval-root",
        str(eval_root),
        "--data-root",
        str(data_root),
        "--edges-path",
        str(edges_path),
        "--out-csv",
        str(out_csv),
        "--corr-sigma-ms",
        str(float(corr_sigma_ms)),
    ]
    for run in runs:
        cmd.extend(["--run", run])
    for dataset in datasets:
        cmd.extend(["--dataset", dataset])
    _run(cmd, cwd=cwd, env=env, dry_run=dry_run)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _filter_finite_correlation_csv(path: Path) -> Path:
    rows = _read_csv(path)
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    filtered = [row for row in rows if _is_finite_number(row.get("correlation"))]
    _write_rows_csv(path, filtered, fieldnames)
    print(f"[inference-demo] filtered {path}: kept {len(filtered)} of {len(rows)} finite rows")
    return path


def _csv_key(row: Mapping[str, str], *, run: str, method: str) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("smoothing", "")),
        str(row.get("trial", "")),
        run,
        method,
    )


def _write_parity_tables(
    *,
    results_dir: Path,
    reference_dir: Path,
    corr_sigma_ms: float,
) -> Path:
    rows: list[dict[str, Any]] = []
    generated_cache: dict[str, list[dict[str, str]]] = {}
    reference_cache: dict[str, list[dict[str, str]]] = {}

    for mapping in PARITY_MAPS:
        gen_rows = generated_cache.setdefault(
            mapping.generated_csv,
            _read_csv(results_dir / mapping.generated_csv),
        )
        ref_rows = reference_cache.setdefault(
            mapping.reference_csv,
            _read_csv(reference_dir / mapping.reference_csv),
        )

        gen_by_key = {
            _csv_key(row, run=row.get("run", ""), method=row.get("method", "")): row
            for row in gen_rows
            if row.get("run") == mapping.generated_run
            and row.get("method") == mapping.generated_method
            and float(row.get("corr_sigma_ms", "nan")) == float(corr_sigma_ms)
        }
        ref_by_key = {
            _csv_key(row, run=row.get("run", ""), method=row.get("method", "")): row
            for row in ref_rows
            if row.get("run") == mapping.reference_run
            and row.get("method") == mapping.reference_method
            and float(row.get("corr_sigma_ms", "nan")) == float(corr_sigma_ms)
        }

        for gen_key, gen in sorted(gen_by_key.items()):
            dataset, smoothing, trial, _run, _method = gen_key
            ref_key = (dataset, smoothing, trial, mapping.reference_run, mapping.reference_method)
            ref = ref_by_key.get(ref_key)
            gen_corr = float(gen.get("correlation", "nan"))
            ref_corr = float(ref.get("correlation", "nan")) if ref else float("nan")
            if not np.isfinite(gen_corr) or not np.isfinite(ref_corr):
                continue
            rows.append(
                {
                    "context": mapping.context,
                    "manuscript_figure": mapping.manuscript_figure,
                    "dataset": dataset,
                    "smoothing": smoothing,
                    "trial": trial,
                    "display_method": mapping.display_method,
                    "generated_run": mapping.generated_run,
                    "generated_method": mapping.generated_method,
                    "reference_run": mapping.reference_run,
                    "reference_method": mapping.reference_method,
                    "corr_sigma_ms": float(corr_sigma_ms),
                    "generated_correlation": gen_corr,
                    "reference_correlation": ref_corr,
                    "delta": gen_corr - ref_corr if np.isfinite(gen_corr) and np.isfinite(ref_corr) else float("nan"),
                    "matched_reference": bool(ref),
                }
            )

    out = results_dir / "correlation_parity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context",
        "manuscript_figure",
        "dataset",
        "smoothing",
        "trial",
        "display_method",
        "generated_run",
        "generated_method",
        "reference_run",
        "reference_method",
        "corr_sigma_ms",
        "generated_correlation",
        "reference_correlation",
        "delta",
        "matched_reference",
    ]
    _write_rows_csv(out, rows, fieldnames)
    return out


def _write_combined_csv(results_dir: Path) -> Path:
    out = results_dir / "trialwise_correlations_reproducible.csv"
    paths = [
        results_dir / "trialwise_correlations_jG8f_repro.csv",
        results_dir / "trialwise_correlations_jG8m_repro.csv",
    ]
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        rows = _read_csv(path)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
            all_rows.append(row)
    if not fieldnames:
        return out
    _write_rows_csv(out, all_rows, fieldnames)
    return out


def _plot_trace_panel(
    *,
    cwd: Path,
    env: Mapping[str, str],
    dry_run: bool,
    csv_path: Path,
    eval_root: Path,
    data_root: Path,
    edges_path: Path,
    dataset: str,
    out: Path,
    methods: Sequence[str],
    smoothing: str,
    corr_sigma_ms: float,
    trial: int,
    start_s: float,
    duration_s: float,
    title: str,
    series_colors: Mapping[str, str] | None = None,
    series_labels: Mapping[str, str] | None = None,
) -> None:
    cmd: list[str] = [
        sys.executable,
        "code/scripts/plot_trialwise_trace_panel.py",
        "--csv",
        str(csv_path),
        "--eval-root",
        str(eval_root),
        "--data-root",
        str(data_root),
        "--edges-path",
        str(edges_path),
        "--dataset",
        dataset,
        "--smoothing",
        smoothing,
        "--corr-sigma-ms",
        str(float(corr_sigma_ms)),
        "--trial",
        str(int(trial)),
        "--start-s",
        str(float(start_s)),
        "--duration-s",
        str(float(duration_s)),
        "--title",
        title,
        "--out",
        str(out),
    ]
    for method in methods:
        cmd.extend(["--method", method])
    for key, color in (series_colors or {}).items():
        cmd.extend(["--series-color", f"{key}={color}"])
    for key, label in (series_labels or {}).items():
        cmd.extend(["--series-label", f"{key}={label}"])
    _run(cmd, cwd=cwd, env=env, dry_run=dry_run)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = _repo_root()
    capsule_root = _capsule_root()
    data_dir = _resolve(args.data_dir)
    results_dir = _resolve(args.results_dir)
    scratch_dir = _resolve(args.scratch_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    env = _base_env(repo_root)
    _append_nvidia_libs_to_env(env)

    edge_8f = data_dir / "reference_inputs" / "edges" / "excitatory_time_stamp_edges.npy"
    edge_8m = data_dir / "reference_inputs" / "edges" / "excitatory_jG8m_edges_2000pts.npy"
    edges_8f = _load_edges(edge_8f)
    edges_8m = _load_edges(edge_8m)
    jg8f_data_root = data_dir / "sample_data" / "janelia_8f" / "excitatory"
    jg8m_data_root = data_dir / "sample_data" / "janelia_8m" / "excitatory"

    notebook_trial_8f = _bounded_trial(JG8F_NOTEBOOK_DATASET, args.notebook_trial, edges_8f)
    notebook_trial_8m = _bounded_trial(JG8M_NOTEBOOK_DATASET, args.notebook_trial, edges_8m)
    source_trial = _bounded_trial(JG8F_BIOPHYS_ML_SOURCE_DATASET, 1, edges_8f)
    source_extra = _random_extra_trials(
        JG8F_BIOPHYS_ML_SOURCE_DATASET,
        edges=edges_8f,
        required=source_trial,
        count=args.extra_random_epochs,
        seed=args.random_seed,
    )

    if args.dataset_percent is not None:
        selection_8f = _selection_for_percent(
            data_root=jg8f_data_root,
            edges=edges_8f,
            percent=float(args.dataset_percent),
        )
        selection_8m = _selection_for_percent(
            data_root=jg8m_data_root,
            edges=edges_8m,
            percent=float(args.dataset_percent),
        )
    elif args.include_all_notebook_trials:
        jg8f_main_trials = list(range(np.asarray(edges_8f[JG8F_NOTEBOOK_DATASET]).shape[0]))
        jg8m_main_trials = list(range(np.asarray(edges_8m[JG8M_NOTEBOOK_DATASET]).shape[0]))
        selection_8f = {
            JG8F_NOTEBOOK_DATASET: _selection_for_dataset(
                JG8F_NOTEBOOK_DATASET,
                edges=edges_8f,
                preferred=jg8f_main_trials,
            ),
            JG8F_BIOPHYS_ML_SOURCE_DATASET: _selection_for_dataset(
                JG8F_BIOPHYS_ML_SOURCE_DATASET,
                edges=edges_8f,
                preferred=[source_trial, *source_extra],
            ),
        }
        selection_8m = {
            JG8M_NOTEBOOK_DATASET: _selection_for_dataset(
                JG8M_NOTEBOOK_DATASET,
                edges=edges_8m,
                preferred=jg8m_main_trials,
            )
        }
    else:
        jg8f_main_trials = [notebook_trial_8f]
        jg8m_main_trials = [notebook_trial_8m]
        selection_8f = {
            JG8F_NOTEBOOK_DATASET: _selection_for_dataset(
                JG8F_NOTEBOOK_DATASET,
                edges=edges_8f,
                preferred=jg8f_main_trials,
            ),
            JG8F_BIOPHYS_ML_SOURCE_DATASET: _selection_for_dataset(
                JG8F_BIOPHYS_ML_SOURCE_DATASET,
                edges=edges_8f,
                preferred=[source_trial, *source_extra],
            ),
        }
        selection_8m = {
            JG8M_NOTEBOOK_DATASET: _selection_for_dataset(
                JG8M_NOTEBOOK_DATASET,
                edges=edges_8m,
                preferred=jg8m_main_trials,
            )
        }

    selection_8f_path = scratch_dir / "code_ocean_selection_jG8f.json"
    selection_8m_path = scratch_dir / "code_ocean_selection_jG8m.json"
    _write_json(selection_8f_path, selection_8f)
    _write_json(selection_8m_path, selection_8m)

    datasets_8f = sorted(selection_8f)
    datasets_8m = sorted(selection_8m)
    jg8f_smoothing_levels = ["raw", "30Hz", "10Hz"]
    jg8m_smoothing_levels = ["raw"]
    ens2_root = data_dir / "Pretrained_models" / "ENS2" / "ens2_published"
    biophys_ml_root = data_dir / "Pretrained_models" / "BiophysML" / "refbuild_biophysml_jg8f"
    biophys_ml_jg8m_root = data_dir / "Pretrained_models" / "BiophysML" / "refbuild_biophysml_jg8m_bm0p05"
    cascade_root = data_dir / "Pretrained_models" / "CASCADE"

    plan = {
        "corr_sigma_ms": float(args.corr_sigma_ms),
        "dataset_percent": args.dataset_percent,
        "jg8f_bm_sigma": str(args.jg8f_bm_sigma),
        "jg8m_bm_sigma": str(args.jg8m_bm_sigma),
        "jG8f_selection": selection_8f,
        "jG8m_selection": selection_8m,
        "jG8f_epoch_count": int(sum(len(v) for v in selection_8f.values())),
        "jG8m_epoch_count": int(sum(len(v) for v in selection_8m.values())),
        "method_matrix": {
            "jG8f_excitatory": {
                "datasets": datasets_8f,
                "smoothing_levels": jg8f_smoothing_levels,
                "methods": ["pgas_gold", "pgas_janelia_params_raw_only", "ens2", "cascade", "biophys_ml"],
            },
            "jG8m_excitatory": {
                "datasets": datasets_8m,
                "smoothing_levels": jg8m_smoothing_levels,
                "methods": ["pgas", "ens2", "cascade", "biophys_ml"],
            },
        },
        "model_paths": {
            "ens2_published": str(ens2_root),
            "biophys_ml_jG8f_reference": str(biophys_ml_root),
            "biophys_ml_jG8m_reference": str(biophys_ml_jg8m_root),
            "cascade_jG8f": str(cascade_root / "universal_p_cascade_exc_30"),
            "cascade_jG8m": str(cascade_root / "Cascade_Universal_30Hz"),
        },
        "pgas_parameter_files": {
            "jG8f_gold": str(data_dir / "pgas_parameters" / "20230525_gold.dat"),
            "jG8f_janelia": str(data_dir / "pgas_parameters" / "20251210_Janelia_8f_params.dat"),
            "jG8m": str(data_dir / "pgas_parameters" / "20251207_jG8m_params.dat"),
        },
        "expected_outputs": {
            "jG8f_trialwise_csv": str(results_dir / "trialwise_correlations_jG8f_repro.csv"),
            "jG8m_trialwise_csv": str(results_dir / "trialwise_correlations_jG8m_repro.csv"),
            "combined_trialwise_csv": str(results_dir / "trialwise_correlations_reproducible.csv"),
            "parity_csv": str(results_dir / "correlation_parity.csv"),
        },
        "reference_note": (
            "Trial indices are 0-based. By default this runs curated epochs only. "
            "Use --dataset-percent or C_SPIKES_DATASET_PERCENT to expand deterministically."
            " jGCaMP8f inhibitory data are staged for optional exploration but are not part "
            "of the default reviewer workflow."
        ),
    }
    _write_json(results_dir / "inference_plan.json", plan)
    print(f"[inference-demo] wrote {results_dir / 'inference_plan.json'}")

    if args.plan_only:
        print("[inference-demo] plan-only requested; stopping before inference.")
        return

    eval_root = results_dir / "full_evaluation"
    cache_root = capsule_root / "results" / "inference_cache"
    pgas_output_root = results_dir / "pgas_output"

    base_8f_methods = ["ens2", "cascade"] if args.skip_pgas else ["pgas", "ens2", "cascade"]
    base_8m_methods = ["ens2", "cascade"] if args.skip_pgas else ["pgas", "ens2", "cascade"]
    param_methods = [] if args.skip_pgas else ["pgas"]

    _run_batch(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        data_root=jg8f_data_root,
        datasets=datasets_8f,
        selection_json=selection_8f_path,
        output_root=eval_root,
        cache_root=cache_root,
        edges_path=edge_8f,
        run_tag=RUN_JG8F_BASE,
        methods=base_8f_methods,
        smoothing_levels=jg8f_smoothing_levels,
        corr_sigma_ms=args.corr_sigma_ms,
        pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8_soma.json",
        pgas_gparam=data_dir / "pgas_parameters" / "20230525_gold.dat",
        pgas_output_root=pgas_output_root / RUN_JG8F_BASE,
        pgas_bm_sigma=str(args.jg8f_bm_sigma),
        ens2_pretrained_root=ens2_root,
        cascade_model_root=cascade_root,
        cascade_model_name="universal_p_cascade_exc_30",
    )
    if param_methods:
        _run_batch(
            cwd=capsule_root,
            env=env,
            dry_run=False,
            data_root=jg8f_data_root,
            datasets=datasets_8f,
            selection_json=selection_8f_path,
            output_root=eval_root,
            cache_root=cache_root,
            edges_path=edge_8f,
            run_tag=RUN_JG8F_PARAMS,
            methods=param_methods,
            smoothing_levels=["raw"],
            corr_sigma_ms=args.corr_sigma_ms,
            pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8_soma.json",
            pgas_gparam=data_dir / "pgas_parameters" / "20251210_Janelia_8f_params.dat",
            pgas_output_root=pgas_output_root / RUN_JG8F_PARAMS,
            pgas_bm_sigma=str(args.jg8f_bm_sigma),
            ens2_pretrained_root=ens2_root,
            cascade_model_root=cascade_root,
            cascade_model_name="universal_p_cascade_exc_30",
        )
    _run_batch(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        data_root=jg8f_data_root,
        datasets=datasets_8f,
        selection_json=selection_8f_path,
        output_root=eval_root,
        cache_root=cache_root,
        edges_path=edge_8f,
        run_tag=RUN_JG8F_BIOPHYS_ML_CACHE,
        methods=["ens2"],
        smoothing_levels=jg8f_smoothing_levels,
        corr_sigma_ms=args.corr_sigma_ms,
        pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8_soma.json",
        pgas_gparam=data_dir / "pgas_parameters" / "20230525_gold.dat",
        pgas_output_root=pgas_output_root / RUN_JG8F_BIOPHYS_ML_CACHE,
        pgas_bm_sigma=str(args.jg8f_bm_sigma),
        ens2_pretrained_root=biophys_ml_root,
        cascade_model_root=cascade_root,
        cascade_model_name="universal_p_cascade_exc_30",
    )
    _import_ens2_cache_as_biophys_ml(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        eval_root=eval_root,
        cache_root=cache_root,
        data_root=jg8f_data_root,
        source_run=RUN_JG8F_BIOPHYS_ML_CACHE,
        target_run=RUN_JG8F_BASE,
        model_name=biophys_ml_root.name,
    )
    _run_batch(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        eval_only=True,
        data_root=jg8f_data_root,
        datasets=datasets_8f,
        selection_json=selection_8f_path,
        output_root=eval_root,
        cache_root=cache_root,
        edges_path=edge_8f,
        run_tag=RUN_JG8F_BASE,
        methods=[*base_8f_methods, "biophys_ml"],
        smoothing_levels=jg8f_smoothing_levels,
        corr_sigma_ms=args.corr_sigma_ms,
        pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8_soma.json",
        pgas_gparam=data_dir / "pgas_parameters" / "20230525_gold.dat",
        pgas_output_root=pgas_output_root / RUN_JG8F_BASE,
        pgas_bm_sigma=str(args.jg8f_bm_sigma),
        ens2_pretrained_root=ens2_root,
        cascade_model_root=cascade_root,
        cascade_model_name="universal_p_cascade_exc_30",
    )
    _run_batch(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        data_root=jg8m_data_root,
        datasets=datasets_8m,
        selection_json=selection_8m_path,
        output_root=eval_root,
        cache_root=cache_root,
        edges_path=edge_8m,
        run_tag=RUN_JG8M_BASE,
        methods=base_8m_methods,
        smoothing_levels=jg8m_smoothing_levels,
        corr_sigma_ms=args.corr_sigma_ms,
        pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8m_soma.json",
        pgas_gparam=data_dir / "pgas_parameters" / "20251207_jG8m_params.dat",
        pgas_output_root=pgas_output_root / RUN_JG8M_BASE,
        pgas_bm_sigma=str(args.jg8m_bm_sigma),
        ens2_pretrained_root=ens2_root,
        cascade_model_root=cascade_root,
        cascade_model_name="Cascade_Universal_30Hz",
    )
    _run_batch(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        data_root=jg8m_data_root,
        datasets=datasets_8m,
        selection_json=selection_8m_path,
        output_root=eval_root,
        cache_root=cache_root,
        edges_path=edge_8m,
        run_tag=RUN_JG8M_BIOPHYS_ML_CACHE,
        methods=["ens2"],
        smoothing_levels=jg8m_smoothing_levels,
        corr_sigma_ms=args.corr_sigma_ms,
        pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8m_soma.json",
        pgas_gparam=data_dir / "pgas_parameters" / "20251207_jG8m_params.dat",
        pgas_output_root=pgas_output_root / RUN_JG8M_BIOPHYS_ML_CACHE,
        pgas_bm_sigma=str(args.jg8m_bm_sigma),
        ens2_pretrained_root=biophys_ml_jg8m_root,
        cascade_model_root=cascade_root,
        cascade_model_name="Cascade_Universal_30Hz",
    )
    _import_ens2_cache_as_biophys_ml(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        eval_root=eval_root,
        cache_root=cache_root,
        data_root=jg8m_data_root,
        source_run=RUN_JG8M_BIOPHYS_ML_CACHE,
        target_run=RUN_JG8M_BASE,
        model_name=biophys_ml_jg8m_root.name,
    )
    _run_batch(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        eval_only=True,
        data_root=jg8m_data_root,
        datasets=datasets_8m,
        selection_json=selection_8m_path,
        output_root=eval_root,
        cache_root=cache_root,
        edges_path=edge_8m,
        run_tag=RUN_JG8M_BASE,
        methods=[*base_8m_methods, "biophys_ml"],
        smoothing_levels=jg8m_smoothing_levels,
        corr_sigma_ms=args.corr_sigma_ms,
        pgas_constants=data_dir / "parameter_files" / "constants_GCaMP8m_soma.json",
        pgas_gparam=data_dir / "pgas_parameters" / "20251207_jG8m_params.dat",
        pgas_output_root=pgas_output_root / RUN_JG8M_BASE,
        pgas_bm_sigma=str(args.jg8m_bm_sigma),
        ens2_pretrained_root=ens2_root,
        cascade_model_root=cascade_root,
        cascade_model_name="Cascade_Universal_30Hz",
    )

    _trialwise_csv(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        eval_root=eval_root,
        data_root=jg8f_data_root,
        edges_path=edge_8f,
        out_csv=results_dir / "trialwise_correlations_jG8f_repro.csv",
        runs=[RUN_JG8F_BASE, RUN_JG8F_PARAMS],
        datasets=datasets_8f,
        corr_sigma_ms=args.corr_sigma_ms,
    )
    _filter_finite_correlation_csv(results_dir / "trialwise_correlations_jG8f_repro.csv")
    _trialwise_csv(
        cwd=capsule_root,
        env=env,
        dry_run=False,
        eval_root=eval_root,
        data_root=jg8m_data_root,
        edges_path=edge_8m,
        out_csv=results_dir / "trialwise_correlations_jG8m_repro.csv",
        runs=[RUN_JG8M_BASE],
        datasets=datasets_8m,
        corr_sigma_ms=args.corr_sigma_ms,
    )
    _filter_finite_correlation_csv(results_dir / "trialwise_correlations_jG8m_repro.csv")
    combined_csv = _write_combined_csv(results_dir)
    print(f"[inference-demo] wrote {combined_csv}")
    parity_csv = _write_parity_tables(
        results_dir=results_dir,
        reference_dir=data_dir / "reference_outputs" / "paper_summaries",
        corr_sigma_ms=args.corr_sigma_ms,
    )
    print(f"[inference-demo] wrote {parity_csv}")

    if args.skip_pgas and not args.skip_plots:
        print("[inference-demo] skipping plots because --skip-pgas omits PGAS traces.")
    elif not args.skip_plots:
        plots_dir = results_dir / "plots"
        _plot_trace_panel(
            cwd=capsule_root,
            env=env,
            dry_run=False,
            csv_path=results_dir / "trialwise_correlations_jG8f_repro.csv",
            eval_root=eval_root,
            data_root=jg8f_data_root,
            edges_path=edge_8f,
            dataset=JG8F_NOTEBOOK_DATASET,
            out=plots_dir / "cell2_jG8f_trace_panel.png",
            methods=[
                f"pgas@{RUN_JG8F_BASE}",
                f"biophys_ml@{RUN_JG8F_BASE}",
                f"ens2=ens2@{RUN_JG8F_BASE}",
                f"cascade@{RUN_JG8F_BASE}",
            ],
            smoothing="raw",
            corr_sigma_ms=args.corr_sigma_ms,
            trial=notebook_trial_8f,
            start_s=465.0,
            duration_s=9.0,
            title="jGCaMP8f inference parity",
        )
        _plot_trace_panel(
            cwd=capsule_root,
            env=env,
            dry_run=False,
            csv_path=results_dir / "trialwise_correlations_jG8f_repro.csv",
            eval_root=eval_root,
            data_root=jg8f_data_root,
            edges_path=edge_8f,
            dataset=JG8F_NOTEBOOK_DATASET,
            out=plots_dir / "cell3_jG8f_parameter_trace_panel.png",
            methods=[
                f"biophys_smc=pgas@{RUN_JG8F_BASE}",
                f"biophys_smc_J=pgas@{RUN_JG8F_PARAMS}",
            ],
            series_colors={"biophys_smc_J": "#000000"},
            series_labels={
                "biophys_smc": "biophys_smc",
                "biophys_smc_J": "biophys_smc_J",
            },
            smoothing="raw",
            corr_sigma_ms=args.corr_sigma_ms,
            trial=notebook_trial_8f,
            start_s=465.0,
            duration_s=9.0,
            title="jGCaMP8f PGAS parameter parity",
        )
        _plot_trace_panel(
            cwd=capsule_root,
            env=env,
            dry_run=False,
            csv_path=results_dir / "trialwise_correlations_jG8m_repro.csv",
            eval_root=eval_root,
            data_root=jg8m_data_root,
            edges_path=edge_8m,
            dataset=JG8M_NOTEBOOK_DATASET,
            out=plots_dir / "cell4_jG8m_trace_panel.png",
            methods=[
                f"pgas@{RUN_JG8M_BASE}",
                f"biophys_ml@{RUN_JG8M_BASE}",
                f"ens2=ens2@{RUN_JG8M_BASE}",
                f"cascade@{RUN_JG8M_BASE}",
            ],
            smoothing="raw",
            corr_sigma_ms=args.corr_sigma_ms,
            trial=notebook_trial_8m,
            start_s=219.0,
            duration_s=9.0,
            title="jGCaMP8m inference parity",
        )


if __name__ == "__main__":
    main()
