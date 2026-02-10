from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from c_spikes.syn_gen import build_synthetic_ground_truth_batch


def default_synthetic_config() -> Dict[str, object]:
    return {
        "burnin": 100,
        "spike_rate_values": [6.0, 9.0, 12.0],
        "smooth_values": [1.3, 2.0],
        "duty_values": [0.35, 0.45],
        "noise_fraction": None,
        "noise_seed_start": 0,
        "noise_seed_stride": 1000,
        "noise_dir": None,
        "gparam_path": "src/c_spikes/pgas/20230525_gold.dat",
        "force_synth": False,
        "seed_spikes": True,
        "synth_tag_prefix": "bio_ml",
    }


def default_ens2_train_config() -> Dict[str, object]:
    return {
        "neuron_type": "Exc",
        "sampling_rate": 60.0,
        "smoothing_std": 0.025,
        "model_prefix": "bio_ml_ens2",
    }


def default_cascade_train_config() -> Dict[str, object]:
    return {
        "template_model_dir": "Pretrained_models/CASCADE/Cascade_Universal_30Hz",
        "model_prefix": "bio_ml_cascade",
    }


def _sanitize_token(value: object) -> str:
    text = str(value).strip()
    return text.replace(" ", "_").replace(".", "p").replace("/", "_")


def _float_list(values: object, name: str) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[float] = []
        for v in values:
            out.append(float(v))
        return out
    raise ValueError(f"{name} must be a list of numeric values.")


def _build_param_specs_for_combo(
    *,
    param_samples_paths: Sequence[Path],
    burnin: int,
    spike_rate: float,
    smooth_value: float,
    duty_value: float,
    noise_dir: Optional[Path],
    noise_fraction: float,
    noise_seed_base: int,
    tag_prefix: str,
    combo_id: str,
) -> List[dict]:
    specs: List[dict] = []
    for idx, param_path in enumerate(param_samples_paths):
        base = param_path.stem.replace("param_samples_", "")
        tag = f"{tag_prefix}_{base}__{combo_id}"
        specs.append(
            {
                "param_samples_path": param_path,
                "burnin": int(burnin),
                "spike_rate": float(spike_rate),
                "spike_params": (float(smooth_value), float(duty_value)),
                "noise_dir": noise_dir,
                "noise_fraction": float(noise_fraction),
                "noise_seed": int(noise_seed_base + idx),
                "tag": tag,
                "run_tag": combo_id,
            }
        )
    return specs


def _ensure_biophys_dir(run_root: Path) -> Path:
    out = Path(run_root) / "biophys_ml"
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_synthetic_bundles(
    *,
    param_samples_paths: Sequence[Path],
    run_root: Path,
    run_tag: str,
    synthetic_config: Dict[str, object],
) -> List[Dict[str, object]]:
    if not param_samples_paths:
        raise ValueError("No param_samples paths were provided.")

    run_root = Path(run_root)
    synth_root = run_root / "Synthetic_Datasets"
    synth_root.mkdir(parents=True, exist_ok=True)

    rates = _float_list(synthetic_config.get("spike_rate_values"), "spike_rate_values")
    smooths = _float_list(synthetic_config.get("smooth_values"), "smooth_values")
    duties = _float_list(synthetic_config.get("duty_values"), "duty_values")
    if not rates or not smooths or not duties:
        raise ValueError("Synthetic sweep lists cannot be empty.")

    burnin = int(synthetic_config.get("burnin", 100))
    force_synth = bool(synthetic_config.get("force_synth", False))
    seed_spikes = bool(synthetic_config.get("seed_spikes", True))
    tag_prefix = _sanitize_token(synthetic_config.get("synth_tag_prefix", "bio_ml"))
    seed_start = int(synthetic_config.get("noise_seed_start", 0))
    seed_stride = int(synthetic_config.get("noise_seed_stride", 1000))
    gparam_path = Path(str(synthetic_config.get("gparam_path", "src/c_spikes/pgas/20230525_gold.dat")))
    noise_dir_val = synthetic_config.get("noise_dir")
    noise_dir = None if noise_dir_val in (None, "", "null") else Path(str(noise_dir_val))

    noise_fraction_cfg = synthetic_config.get("noise_fraction")
    if noise_fraction_cfg is None:
        noise_fraction = 1.0 / max(len(param_samples_paths), 1)
    else:
        noise_fraction = float(noise_fraction_cfg)

    bundles: List[Dict[str, object]] = []
    combo_index = 0
    for rate in rates:
        for smooth in smooths:
            for duty in duties:
                seed_base = seed_start + combo_index * seed_stride
                combo_id = (
                    f"{_sanitize_token(run_tag)}_r{_sanitize_token(f'{rate:g}')}"
                    f"_s{_sanitize_token(f'{smooth:g}')}"
                    f"_d{_sanitize_token(f'{duty:g}')}"
                    f"_sb{seed_base}"
                )
                specs = _build_param_specs_for_combo(
                    param_samples_paths=[Path(p) for p in param_samples_paths],
                    burnin=burnin,
                    spike_rate=rate,
                    smooth_value=smooth,
                    duty_value=duty,
                    noise_dir=noise_dir,
                    noise_fraction=noise_fraction,
                    noise_seed_base=seed_base,
                    tag_prefix=tag_prefix,
                    combo_id=combo_id,
                )
                cparams_map = build_synthetic_ground_truth_batch(
                    specs,
                    gparam_path=gparam_path,
                    output_root=synth_root,
                    manifest_path=_ensure_biophys_dir(run_root) / "synthetic_manifest.json",
                    manifest_model_name=f"{tag_prefix}_{combo_id}",
                    force_synth=force_synth,
                    seed_spikes=seed_spikes,
                )
                synth_dirs = [
                    synth_root / "Ground_truth" / f"synth_{spec['tag']}"
                    for spec in specs
                ]
                bundles.append(
                    {
                        "bundle_id": combo_id,
                        "spike_rate": rate,
                        "spike_params": [smooth, duty],
                        "noise_seed_base": seed_base,
                        "param_samples_paths": [str(Path(p)) for p in param_samples_paths],
                        "synth_dirs": [str(p) for p in synth_dirs],
                        "cparams_map": {k: v.tolist() for k, v in cparams_map.items()},
                    }
                )
                combo_index += 1

    bundle_path = _ensure_biophys_dir(run_root) / "synthetic_bundles.json"
    bundle_path.write_text(json.dumps(bundles, indent=2) + "\n", encoding="utf-8")
    return bundles


def _prepare_cascade_model_dir(
    *,
    model_dir: Path,
    template_model_dir: Path,
    training_datasets: Sequence[str],
) -> None:
    from c_spikes.cascade2p import config as cascade_config

    model_dir.mkdir(parents=True, exist_ok=True)
    template_cfg = Path(template_model_dir) / "config.yaml"
    if not template_cfg.exists():
        raise FileNotFoundError(f"Cascade template config not found: {template_cfg}")
    target_cfg = model_dir / "config.yaml"
    if not target_cfg.exists():
        shutil.copy2(template_cfg, target_cfg)
    cfg = cascade_config.read_config(str(target_cfg))
    cfg["model_name"] = model_dir.name
    cfg["training_datasets"] = list(training_datasets)
    cfg["training_finished"] = "No"
    cascade_config.write_config(cfg, str(target_cfg))


def train_models_for_bundles(
    *,
    bundles: Sequence[Dict[str, object]],
    run_root: Path,
    model_family: str,
    model_root: Path,
    ens2_train_config: Optional[Dict[str, object]] = None,
    cascade_train_config: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if not bundles:
        raise ValueError("No synthetic bundles to train.")

    run_root = Path(run_root)
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    family = str(model_family).strip().lower()
    if family not in {"ens2", "cascade"}:
        raise ValueError("model_family must be 'ens2' or 'cascade'.")

    if family == "ens2":
        from c_spikes.ens2 import train_model as ens2_train_model

        cfg = dict(default_ens2_train_config())
        if ens2_train_config:
            cfg.update(ens2_train_config)
        prefix = _sanitize_token(cfg.get("model_prefix", "bio_ml_ens2"))
        neuron_type = str(cfg.get("neuron_type", "Exc"))
        sampling_rate = float(cfg.get("sampling_rate", 60.0))
        smoothing_std = float(cfg.get("smoothing_std", 0.025))

        for bundle in bundles:
            bundle_id = _sanitize_token(bundle["bundle_id"])
            model_name = f"{prefix}_{bundle_id}"
            synth_dirs = [Path(p) for p in bundle.get("synth_dirs", [])]
            checkpoint = ens2_train_model(
                model_name=model_name,
                synth_gt_dir=synth_dirs,
                model_root=model_root,
                neuron_type=neuron_type,
                sampling_rate=sampling_rate,
                smoothing_std=smoothing_std,
                manifest_path=model_root / model_name / "ens2_manifest.json",
                run_tag=bundle_id,
            )
            records.append(
                {
                    "bundle_id": bundle_id,
                    "model_family": "ens2",
                    "model_name": model_name,
                    "model_dir": str(model_root / model_name),
                    "checkpoint_path": str(checkpoint),
                }
            )
    else:
        from c_spikes.cascade2p import cascade

        cfg = dict(default_cascade_train_config())
        if cascade_train_config:
            cfg.update(cascade_train_config)
        prefix = _sanitize_token(cfg.get("model_prefix", "bio_ml_cascade"))
        template_model_dir = Path(str(cfg.get("template_model_dir", "Pretrained_models/CASCADE/Cascade_Universal_30Hz")))
        synthetic_ground_truth_root = run_root / "Synthetic_Datasets" / "Ground_truth"

        for bundle in bundles:
            bundle_id = _sanitize_token(bundle["bundle_id"])
            model_name = f"{prefix}_{bundle_id}"
            model_dir = model_root / model_name
            synth_dirs = [Path(p) for p in bundle.get("synth_dirs", [])]
            training_datasets = [p.name for p in synth_dirs]
            _prepare_cascade_model_dir(
                model_dir=model_dir,
                template_model_dir=template_model_dir,
                training_datasets=training_datasets,
            )
            cascade.train_model(
                model_name=model_name,
                model_folder=str(model_root),
                ground_truth_folder=str(synthetic_ground_truth_root),
            )
            records.append(
                {
                    "bundle_id": bundle_id,
                    "model_family": "cascade",
                    "model_name": model_name,
                    "model_dir": str(model_dir),
                }
            )

    record_path = _ensure_biophys_dir(run_root) / f"trained_{family}_models.json"
    record_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records

