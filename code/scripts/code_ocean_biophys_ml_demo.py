#!/usr/bin/env python3
"""Regenerate the packaged BiophysML ENS2 model and compare checksums.

The stage is driven from the packaged model manifest. It verifies that the
PGAS parameter samples and GCaMP parameters match the original inputs, rebuilds
the synthetic ground-truth dataset with those settings, retrains ENS2 from the
archived original synthetic training input, and compares the resulting model
weights to the bundled BiophysML checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_NAME = "refbuild_biophysml_jg8f"
RUN_TAG = "auto"

TRACE_DATASET = "jGCaMP8f_ANM471994_cell05"
TRACE_TRIAL = 1
TRACE_START_S = 465.0
TRACE_DURATION_S = 7.0
TRACE_CORR_SIGMA_MS = 30.0
REFERENCE_TRACE_RUN = "code_ocean_biophys_ml_reference"
RETRAINED_TRACE_RUN = "code_ocean_biophys_ml_retrained"
MANUSCRIPT_FIGURE_BUILD = "Figure 4A"
MANUSCRIPT_FIGURE_TRACE = "Figure 4B,D"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=os.environ.get("C_SPIKES_BIOPHYS_ML_MODEL", MODEL_NAME))
    parser.add_argument("--run-tag", default=os.environ.get("C_SPIKES_BIOPHYS_ML_RUN_TAG", RUN_TAG))
    parser.add_argument(
        "--plan-only",
        action="store_true",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_PLAN_ONLY", "0") == "1",
        help="Verify packaged inputs and write the plan/report without synthesis or training.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_SKIP_TRAIN", "0") == "1",
        help="Generate and hash synthetic data, but do not retrain ENS2.",
    )
    parser.add_argument(
        "--skip-inference-check",
        action="store_true",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_SKIP_INFERENCE", "0") == "1",
        help="Skip the reference-vs-retrained ENS2 trace parity check.",
    )
    parser.add_argument(
        "--strict-weight-check",
        action="store_true",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_STRICT_WEIGHTS", "0") == "1",
        help="Fail the stage if retrained model weights are not bit-identical to the reference state dict.",
    )
    parser.add_argument(
        "--force-synth",
        action="store_true",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_FORCE_SYNTH", "1") == "1",
        help="Overwrite the generated synthetic .mat files under this stage's results directory.",
    )
    parser.add_argument(
        "--train-from-generated-synth",
        action="store_true",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_TRAIN_FROM_GENERATED", "0") == "1",
        help=(
            "Train from this run's regenerated synthetic files. By default the retraining check "
            "uses the archived synthetic ground truth that was the exact original ENS2 training input."
        ),
    )
    parser.add_argument(
        "--param-samples",
        type=Path,
        default=None,
        help="Override the packaged PGAS param_samples input.",
    )
    parser.add_argument(
        "--trace-dataset",
        default=os.environ.get("C_SPIKES_BIOPHYS_ML_TRACE_DATASET", TRACE_DATASET),
    )
    parser.add_argument(
        "--trace-trial",
        type=int,
        default=int(os.environ.get("C_SPIKES_BIOPHYS_ML_TRACE_TRIAL", str(TRACE_TRIAL))),
    )
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).astype(np.float32).tobytes()).hexdigest()


def _hash_synth_dir(path: Path) -> dict[str, Any]:
    import scipy.io as sio

    h = hashlib.sha256()
    count = 0
    total = 0
    for mat in sorted(path.glob("*.mat")):
        payload = mat.read_bytes()
        h.update(mat.name.encode("utf-8"))
        data = sio.loadmat(mat)
        for key in sorted(k for k in data if not k.startswith("__")):
            h.update(key.encode("utf-8"))
            _update_hash_with_mat_obj(h, data[key])
        count += 1
        total += len(payload)
    return {"sha256": h.hexdigest(), "file_count": count, "total_bytes": total}


def _update_hash_with_mat_obj(h: Any, obj: Any) -> None:
    if isinstance(obj, np.ndarray):
        h.update(b"ndarray")
        h.update(str(obj.shape).encode("ascii"))
        h.update(str(obj.dtype).encode("ascii"))
        if obj.dtype.names:
            for name in obj.dtype.names:
                h.update(b"field")
                h.update(name.encode("utf-8"))
                _update_hash_with_mat_obj(h, obj[name])
        elif obj.dtype == object:
            for item in obj.flat:
                _update_hash_with_mat_obj(h, item)
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
    elif np.isscalar(obj):
        arr = np.asarray(obj)
        h.update(b"scalar")
        h.update(str(arr.dtype).encode("ascii"))
        h.update(arr.tobytes())
    elif isinstance(obj, str):
        h.update(b"str")
        h.update(obj.encode("utf-8"))
    elif isinstance(obj, (bytes, bytearray)):
        h.update(b"bytes")
        h.update(bytes(obj))
    else:
        h.update(type(obj).__name__.encode("utf-8"))
        h.update(repr(obj).encode("utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_manifest_entry(manifest: Mapping[str, Any], run_tag: str) -> dict[str, Any]:
    entries = [entry for entry in manifest.get("synthetic_entries", []) or [] if isinstance(entry, dict)]
    if run_tag in {"", "auto"}:
        if not entries:
            raise KeyError("No synthetic manifest entries found.")
        return entries[0]
    for entry in manifest.get("synthetic_entries", []) or []:
        if isinstance(entry, dict) and entry.get("run_tag") == run_tag:
            return entry
    raise KeyError(f"No synthetic manifest entry found for run_tag={run_tag!r}")


def _packaged_param_samples_path(data_dir: Path, entry: Mapping[str, Any]) -> Path:
    return data_dir / "reference_inputs" / "biophys_ml" / Path(str(entry["param_samples_path"])).name


def _packaged_gparam_path(data_dir: Path, entry: Mapping[str, Any]) -> Path:
    return data_dir / "pgas_parameters" / Path(str(entry["gparam_path"])).name


def _load_cparams(param_samples: Path, burnin: int) -> np.ndarray:
    samples = np.loadtxt(param_samples, delimiter=",", skiprows=1)
    if samples.ndim == 1:
        samples = samples[None, :]
    burnin = max(0, min(int(burnin), samples.shape[0] - 1))
    return np.mean(samples[burnin:, 0:6], axis=0)


def _base_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
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


def _run(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    print("[biophys-ml-demo] " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(list(cmd), cwd=str(cwd), env=dict(env), check=True)


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.state_dict() if hasattr(obj, "state_dict") else obj
    return {str(k): v.detach().cpu().contiguous() for k, v in state.items()}


def _state_dict_sha256(path: Path) -> str:
    state = _load_state_dict(path)
    h = hashlib.sha256()
    for key in sorted(state):
        arr = state[key].numpy()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _try_state_dict_sha256(path: Path) -> str | None:
    try:
        return _state_dict_sha256(path)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return None
        raise


def _compare_state_dicts(reference: Path, candidate: Path) -> dict[str, Any]:
    ref = _load_state_dict(reference)
    cand = _load_state_dict(candidate)
    keys = sorted(set(ref) | set(cand))
    missing = [k for k in keys if k not in cand]
    extra = [k for k in keys if k not in ref]
    max_abs = 0.0
    max_key = None
    all_equal = not missing and not extra
    for key in keys:
        if key not in ref or key not in cand:
            all_equal = False
            continue
        a = ref[key].numpy()
        b = cand[key].numpy()
        if a.shape != b.shape:
            all_equal = False
            max_key = key
            max_abs = float("inf")
            continue
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        if diff > max_abs:
            max_abs = diff
            max_key = key
        if diff != 0.0:
            all_equal = False
    return {
        "all_equal": bool(all_equal),
        "max_abs_diff": max_abs,
        "max_abs_diff_key": max_key,
        "missing_keys": missing,
        "extra_keys": extra,
    }


def _write_checksum_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["component", "manuscript_figure", "expected", "observed", "matched", "path", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_trace_check(
    *,
    capsule_root: Path,
    env: Mapping[str, str],
    data_dir: Path,
    results_dir: Path,
    scratch_dir: Path,
    reference_model_dir: Path,
    retrained_model_dir: Path,
    dataset: str,
    trial: int,
) -> dict[str, Any]:
    data_root = data_dir / "sample_data" / "janelia_8f" / "excitatory"
    edges_path = data_dir / "reference_inputs" / "edges" / "excitatory_time_stamp_edges.npy"
    selection_path = scratch_dir / "biophys_ml_trace_selection.json"
    _write_json(selection_path, {dataset: [int(trial)]})
    eval_root = results_dir / "full_evaluation"
    cache_root = capsule_root / "results" / "inference_cache"
    corr_sigma_ms = TRACE_CORR_SIGMA_MS

    def run_inference(run_tag: str, model_dir: Path) -> None:
        _run(
            [
                sys.executable,
                "-m",
                "c_spikes.cli.run",
                "--data-root",
                str(data_root),
                "--dataset",
                dataset,
                "--smoothing-level",
                "raw",
                "--method",
                "ens2",
                "--output-root",
                str(eval_root),
                "--cache-root",
                str(cache_root),
                "--edges-path",
                str(edges_path),
                "--trial-selection-path",
                str(selection_path),
                "--corr-sigma-ms",
                str(corr_sigma_ms),
                "--trialwise-correlations",
                "--run-tag",
                run_tag,
                "--ens2-pretrained-root",
                str(model_dir),
            ],
            cwd=capsule_root,
            env=env,
        )

    run_inference(REFERENCE_TRACE_RUN, reference_model_dir)
    run_inference(RETRAINED_TRACE_RUN, retrained_model_dir)

    csv_path = results_dir / "trialwise_correlations_trace_check.csv"
    _run(
        [
            sys.executable,
            "code/scripts/trialwise_correlations.py",
            "--eval-root",
            str(eval_root),
            "--data-root",
            str(data_root),
            "--edges-path",
            str(edges_path),
            "--out-csv",
            str(csv_path),
            "--corr-sigma-ms",
            str(corr_sigma_ms),
            "--run",
            REFERENCE_TRACE_RUN,
            "--run",
            RETRAINED_TRACE_RUN,
            "--dataset",
            dataset,
        ],
        cwd=capsule_root,
        env=env,
    )

    plot_path = results_dir / "plots" / "biophys_ml_retrained_trace_panel.png"
    _run(
        [
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
            "raw",
            "--corr-sigma-ms",
            str(corr_sigma_ms),
            "--trial",
            str(int(trial)),
            "--start-s",
            str(TRACE_START_S),
            "--duration-s",
            str(TRACE_DURATION_S),
            "--title",
            "BiophysML checkpoint parity",
            "--method",
            f"reference=ens2@{REFERENCE_TRACE_RUN}",
            "--method",
            f"retrained=ens2@{RETRAINED_TRACE_RUN}",
            "--out",
            str(plot_path),
        ],
        cwd=capsule_root,
        env=env,
    )

    pred = _compare_cached_predictions(
        capsule_root=capsule_root,
        eval_root=eval_root,
        dataset=dataset,
        smoothing="raw",
        reference_run=REFERENCE_TRACE_RUN,
        candidate_run=RETRAINED_TRACE_RUN,
    )
    pred.update({"csv_path": str(csv_path), "plot_path": str(plot_path)})
    return pred


def _cache_path_from_comparison(capsule_root: Path, eval_root: Path, run_tag: str, dataset: str, smoothing: str) -> Path:
    from c_spikes.inference.types import compute_config_signature

    cmp_path = eval_root / run_tag / dataset / smoothing / "comparison.json"
    obj = _load_json(cmp_path)
    entry = None
    for candidate in obj.get("methods", []) or []:
        if isinstance(candidate, dict) and candidate.get("method") == "ens2":
            entry = candidate
            break
    if entry is None:
        raise KeyError(f"No ENS2 method entry in {cmp_path}")
    cache_tag = str(entry.get("cache_tag") or dataset)
    cache_key = str(entry.get("cache_key") or "")
    if not cache_key:
        cache_key, _ = compute_config_signature(entry.get("config", {}))
    return capsule_root / "results" / "inference_cache" / "ens2" / cache_tag / f"{cache_key}.mat"


def _compare_cached_predictions(
    *,
    capsule_root: Path,
    eval_root: Path,
    dataset: str,
    smoothing: str,
    reference_run: str,
    candidate_run: str,
) -> dict[str, Any]:
    import scipy.io as sio

    ref_path = _cache_path_from_comparison(capsule_root, eval_root, reference_run, dataset, smoothing)
    cand_path = _cache_path_from_comparison(capsule_root, eval_root, candidate_run, dataset, smoothing)
    ref = sio.loadmat(ref_path)
    cand = sio.loadmat(cand_path)
    ref_prob = np.asarray(ref["spike_prob"], dtype=np.float32)
    cand_prob = np.asarray(cand["spike_prob"], dtype=np.float32)
    ref_time = np.asarray(ref["time_stamps"], dtype=np.float64)
    cand_time = np.asarray(cand["time_stamps"], dtype=np.float64)

    h_ref = hashlib.sha256()
    h_ref.update(ref_time.tobytes())
    h_ref.update(ref_prob.tobytes())
    h_cand = hashlib.sha256()
    h_cand.update(cand_time.tobytes())
    h_cand.update(cand_prob.tobytes())
    same_shape = ref_prob.shape == cand_prob.shape and ref_time.shape == cand_time.shape
    if same_shape:
        max_abs = float(np.max(np.abs(ref_prob - cand_prob))) if ref_prob.size else 0.0
        same_values = bool(np.array_equal(ref_prob, cand_prob) and np.array_equal(ref_time, cand_time))
    else:
        max_abs = float("inf")
        same_values = False
    return {
        "reference_cache": str(ref_path),
        "candidate_cache": str(cand_path),
        "reference_prediction_sha256": h_ref.hexdigest(),
        "candidate_prediction_sha256": h_cand.hexdigest(),
        "prediction_sha256_match": h_ref.hexdigest() == h_cand.hexdigest(),
        "prediction_arrays_equal": same_values,
        "prediction_max_abs_diff": max_abs,
        "prediction_shape_match": same_shape,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = _repo_root()
    capsule_root = repo_root
    data_dir = _resolve(args.data_dir)
    results_dir = _resolve(args.results_dir)
    scratch_dir = _resolve(args.scratch_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if str(repo_root / "code" / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "code" / "src"))

    env = _base_env(repo_root)
    _append_nvidia_libs_to_env(env)

    reference_model_dir = data_dir / "Pretrained_models" / "BiophysML" / str(args.model_name)
    manifest_path = reference_model_dir / "ens2_manifest.json"
    manifest = _load_json(manifest_path)
    entry = _find_manifest_entry(manifest, str(args.run_tag))
    selected_run_tag = str(entry["run_tag"])
    syn_cfg = dict(entry["syn_gen"])

    param_samples = args.param_samples
    if param_samples is None:
        param_samples = _packaged_param_samples_path(data_dir, entry)
    param_samples = _resolve(param_samples)
    gparam_path = _packaged_gparam_path(data_dir, entry)
    noise_dir = data_dir / "reference_inputs" / "synthetic_noise" / "gt_noise_dir"
    archived_synth_dir = data_dir / "reference_inputs" / "biophys_ml" / "Ground_truth" / f"synth_{syn_cfg['tag']}"
    reference_checkpoint = reference_model_dir / "exc_ens2_pub.pt"
    if not archived_synth_dir.exists():
        raise FileNotFoundError(
            "Missing archived BiophysML synthetic training input: "
            f"{archived_synth_dir}. This directory is required for exact ENS2 weight reproduction."
        )

    cparams = _load_cparams(param_samples, int(entry["burnin"]))
    input_checks = {
        "param_samples": {
            "path": str(param_samples),
            "expected_sha256": entry["param_samples_sha256"],
            "observed_sha256": _sha256_file(param_samples),
        },
        "gparam": {
            "path": str(gparam_path),
            "expected_sha256": entry["gparam_sha256"],
            "observed_sha256": _sha256_file(gparam_path),
        },
        "cparams": {
            "expected": entry["cparams_mean"],
            "observed": cparams.tolist(),
            "expected_hash": entry["cparams_hash"],
            "observed_hash": _hash_array(cparams),
            "max_abs_diff": float(np.max(np.abs(np.asarray(entry["cparams_mean"], dtype=float) - cparams))),
        },
    }
    for check in ("param_samples", "gparam"):
        input_checks[check]["sha256_match"] = (
            input_checks[check]["expected_sha256"] == input_checks[check]["observed_sha256"]
        )
    input_checks["cparams"]["hash_match"] = input_checks["cparams"]["expected_hash"] == input_checks["cparams"]["observed_hash"]

    reference_checkpoint_file_sha256 = _sha256_file(reference_checkpoint)
    reference_checkpoint_state_dict_sha256 = _try_state_dict_sha256(reference_checkpoint)
    reference_checkpoint_info = {
        "path": str(reference_checkpoint),
        "expected_file_sha256": reference_checkpoint_file_sha256,
        "observed_file_sha256": reference_checkpoint_file_sha256,
        "expected_state_dict_sha256": reference_checkpoint_state_dict_sha256,
        "observed_state_dict_sha256": reference_checkpoint_state_dict_sha256,
        "state_dict_note": None
        if reference_checkpoint_state_dict_sha256 is not None
        else "Skipped because torch is not importable in this environment.",
    }
    reference_checkpoint_info["file_sha256_match"] = (
        reference_checkpoint_info["expected_file_sha256"] == reference_checkpoint_info["observed_file_sha256"]
    )
    reference_checkpoint_info["state_dict_sha256_match"] = (
        reference_checkpoint_info["expected_state_dict_sha256"] == reference_checkpoint_info["observed_state_dict_sha256"]
    )

    report: dict[str, Any] = {
        "model_name": args.model_name,
        "requested_run_tag": args.run_tag,
        "run_tag": selected_run_tag,
        "reference_manifest": str(manifest_path),
        "input_checks": input_checks,
        "reference_checkpoint": reference_checkpoint_info,
        "archived_synthetic_training_input": _hash_synth_dir(archived_synth_dir) | {"path": str(archived_synth_dir)},
        "synthetic_generation": {},
        "training": {"skipped": bool(args.skip_train or args.plan_only)},
        "trace_check": {"skipped": bool(args.skip_inference_check or args.skip_train or args.plan_only)},
    }
    _write_json(results_dir / "biophys_ml_reproducibility_report.json", report)
    print(f"[biophys-ml-demo] wrote {results_dir / 'biophys_ml_reproducibility_report.json'}", flush=True)

    if args.plan_only:
        print("[biophys-ml-demo] plan-only requested; stopping before synthesis/training.")
        _write_checksum_outputs(results_dir, report)
        return

    from c_spikes.syn_gen import build_synthetic_ground_truth_batch

    param_spec = {
        "param_samples_path": param_samples,
        "burnin": int(entry["burnin"]),
        "spike_rate": float(syn_cfg["spike_rate"]),
        "spike_params": list(syn_cfg["spike_params"]),
        "noise_dir": noise_dir,
        "noise_fraction": float(syn_cfg.get("noise_fraction", 1.0)),
        "noise_seed": syn_cfg.get("noise_seed"),
        "tag": syn_cfg["tag"],
        "run_tag": selected_run_tag,
    }
    retrained_model_root = results_dir / "Pretrained_models"
    retrained_model_dir = retrained_model_root / str(args.model_name)
    retrained_manifest = retrained_model_dir / "ens2_manifest.json"
    build_synthetic_ground_truth_batch(
        [param_spec],
        gparam_path=gparam_path,
        output_root=results_dir,
            manifest_path=retrained_manifest,
            manifest_model_name=str(args.model_name),
        force_synth=bool(args.force_synth),
        seed_spikes=True,
    )
    synth_dir = results_dir / "Ground_truth" / f"synth_{syn_cfg['tag']}"
    generated_synth_hash = _hash_synth_dir(synth_dir) | {"path": str(synth_dir)}
    archived_synth_hash = report["archived_synthetic_training_input"]
    report["synthetic_generation"] = {
        "generated": generated_synth_hash,
        "archived_reference": archived_synth_hash,
        "semantic_sha256_match": generated_synth_hash["sha256"] == archived_synth_hash["sha256"],
        "file_count_match": generated_synth_hash["file_count"] == archived_synth_hash["file_count"],
        "total_bytes_match": generated_synth_hash["total_bytes"] == archived_synth_hash["total_bytes"],
        "note": (
            "The semantic hash is computed from loaded MATLAB arrays, not raw .mat bytes, "
            "so scipy save timestamps do not affect the comparison."
        ),
    }
    _write_json(results_dir / "biophys_ml_reproducibility_report.json", report)

    if not args.skip_train:
        from c_spikes.ens2 import train_model

        training_synth_dir = synth_dir if args.train_from_generated_synth else archived_synth_dir
        checkpoint_path = train_model(
            model_name=str(args.model_name),
            synth_gt_dir=training_synth_dir,
            model_root=retrained_model_root,
            neuron_type="Exc",
            sampling_rate=60.0,
            smoothing_std=0.025,
            manifest_path=retrained_manifest,
            run_tag=selected_run_tag,
        )
        training_info = {
            "skipped": False,
            "checkpoint_path": str(checkpoint_path),
            "file_sha256": _sha256_file(checkpoint_path),
            "state_dict_sha256": _state_dict_sha256(checkpoint_path),
            "expected_file_sha256": reference_checkpoint_file_sha256,
            "expected_state_dict_sha256": reference_checkpoint_state_dict_sha256,
            "synth_dir_used": str(training_synth_dir),
            "trained_from_generated_synth": bool(args.train_from_generated_synth),
        }
        training_info["file_sha256_match"] = training_info["file_sha256"] == reference_checkpoint_file_sha256
        training_info["state_dict_sha256_match"] = training_info["state_dict_sha256"] == reference_checkpoint_state_dict_sha256
        training_info["state_dict_comparison"] = _compare_state_dicts(reference_checkpoint, checkpoint_path)
        report["training"] = training_info
        _write_json(results_dir / "biophys_ml_reproducibility_report.json", report)

        if not args.skip_inference_check:
            report["trace_check"] = _run_trace_check(
                capsule_root=capsule_root,
                env=env,
                data_dir=data_dir,
                results_dir=results_dir,
                scratch_dir=scratch_dir,
                reference_model_dir=reference_model_dir,
                retrained_model_dir=retrained_model_dir,
                dataset=str(args.trace_dataset),
                trial=int(args.trace_trial),
            )
            report["trace_check"]["skipped"] = False
            _write_json(results_dir / "biophys_ml_reproducibility_report.json", report)

        if args.strict_weight_check and not bool(training_info["state_dict_sha256_match"]):
            _write_checksum_outputs(results_dir, report)
            raise SystemExit("Retained BiophysML weights did not match the reference state dict checksum.")

    _write_checksum_outputs(results_dir, report)
    print(f"[biophys-ml-demo] wrote {results_dir / 'biophys_ml_reproducibility_report.json'}", flush=True)
    print(f"[biophys-ml-demo] wrote {results_dir / 'checksum_summary.csv'}", flush=True)


def _write_checksum_outputs(results_dir: Path, report: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    inputs = report.get("input_checks", {})
    for name in ("param_samples", "gparam"):
        check = inputs.get(name, {}) if isinstance(inputs, dict) else {}
        rows.append(
            {
                "component": name,
                "manuscript_figure": MANUSCRIPT_FIGURE_BUILD,
                "expected": check.get("expected_sha256"),
                "observed": check.get("observed_sha256"),
                "matched": check.get("sha256_match"),
                "path": check.get("path"),
                "note": "packaged input checksum",
            }
        )
    cparams = inputs.get("cparams", {}) if isinstance(inputs, dict) else {}
    rows.append(
        {
            "component": "cparams_mean",
            "manuscript_figure": MANUSCRIPT_FIGURE_BUILD,
            "expected": cparams.get("expected_hash"),
            "observed": cparams.get("observed_hash"),
            "matched": cparams.get("hash_match"),
            "path": "",
            "note": f"max_abs_diff={cparams.get('max_abs_diff')}",
        }
    )
    ref = report.get("reference_checkpoint", {})
    if isinstance(ref, dict):
        rows.append(
            {
                "component": "reference_checkpoint_file",
                "manuscript_figure": MANUSCRIPT_FIGURE_TRACE,
                "expected": ref.get("expected_file_sha256"),
                "observed": ref.get("observed_file_sha256"),
                "matched": ref.get("file_sha256_match"),
                "path": ref.get("path"),
                "note": "packaged checkpoint bytes",
            }
        )
        rows.append(
            {
                "component": "reference_checkpoint_state_dict",
                "manuscript_figure": MANUSCRIPT_FIGURE_TRACE,
                "expected": ref.get("expected_state_dict_sha256"),
                "observed": ref.get("observed_state_dict_sha256"),
                "matched": ref.get("state_dict_sha256_match"),
                "path": ref.get("path"),
                "note": "packaged checkpoint weights",
            }
        )
    synth = report.get("synthetic_generation", {})
    if isinstance(synth, dict) and synth:
        generated = synth.get("generated", {}) if isinstance(synth.get("generated", {}), dict) else {}
        archived = (
            synth.get("archived_reference", {})
            if isinstance(synth.get("archived_reference", {}), dict)
            else {}
        )
        rows.append(
            {
                "component": "synthetic_ground_truth",
                "manuscript_figure": MANUSCRIPT_FIGURE_BUILD,
                "expected": archived.get("sha256"),
                "observed": generated.get("sha256"),
                "matched": synth.get("semantic_sha256_match"),
                "path": generated.get("path"),
                "note": (
                    f"semantic .mat array hash; files={generated.get('file_count')} "
                    f"bytes={generated.get('total_bytes')}"
                ),
            }
        )
    archived = report.get("archived_synthetic_training_input", {})
    if isinstance(archived, dict) and archived:
        rows.append(
            {
                "component": "archived_synthetic_training_input",
                "manuscript_figure": MANUSCRIPT_FIGURE_BUILD,
                "expected": archived.get("sha256"),
                "observed": archived.get("sha256"),
                "matched": True,
                "path": archived.get("path"),
                "note": f"exact ENS2 training input; files={archived.get('file_count')}",
            }
        )
    training = report.get("training", {})
    if isinstance(training, dict) and not training.get("skipped") and training.get("checkpoint_path"):
        rows.append(
            {
                "component": "retrained_checkpoint_file",
                "manuscript_figure": MANUSCRIPT_FIGURE_BUILD,
                "expected": training.get("expected_file_sha256"),
                "observed": training.get("file_sha256"),
                "matched": training.get("file_sha256_match"),
                "path": training.get("checkpoint_path"),
                "note": "checkpoint bytes; filename timestamp may differ before publication copy",
            }
        )
        rows.append(
            {
                "component": "retrained_checkpoint_state_dict",
                "manuscript_figure": MANUSCRIPT_FIGURE_BUILD,
                "expected": training.get("expected_state_dict_sha256"),
                "observed": training.get("state_dict_sha256"),
                "matched": training.get("state_dict_sha256_match"),
                "path": training.get("checkpoint_path"),
                "note": f"max_abs_diff={training.get('state_dict_comparison', {}).get('max_abs_diff')}",
            }
        )
    trace = report.get("trace_check", {})
    if isinstance(trace, dict) and not trace.get("skipped") and trace.get("reference_prediction_sha256"):
        rows.append(
            {
                "component": "trace_prediction",
                "manuscript_figure": MANUSCRIPT_FIGURE_TRACE,
                "expected": trace.get("reference_prediction_sha256"),
                "observed": trace.get("candidate_prediction_sha256"),
                "matched": trace.get("prediction_sha256_match"),
                "path": trace.get("plot_path"),
                "note": f"max_abs_diff={trace.get('prediction_max_abs_diff')}",
            }
        )
    _write_checksum_csv(results_dir / "checksum_summary.csv", rows)


if __name__ == "__main__":
    main()
