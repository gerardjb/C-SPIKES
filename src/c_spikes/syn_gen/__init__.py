from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

from .synth_gen import synth_gen


def build_synthetic_ground_truth_from_pgas(
    param_samples_path: Path | str,
    *,
    gparam_path: Path | str = Path("src/c_spikes/pgas/20230525_gold.dat"),
    burnin: int = 100,
    spike_rate: float = 2.0,
    spike_params: Sequence[float] = (5.0, 0.5),
    noise_dir: Optional[Path | str] = None,
    noise_fraction: float = 1.0,
    noise_seed: Optional[Union[int, Sequence[int]]] = None,
    tag: Optional[str] = None,
    output_root: Path | str = Path("results"),
    manifest_path: Optional[Path | str] = None,
    manifest_model_name: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> np.ndarray:
    """
    Convenience helper: take a PGAS param_samples*.dat file and generate a
    CASCADE/ENS2-compatible synthetic ground-truth dataset.

    Returns:
        np.ndarray: Mean cell parameters (Cparams) used for synthesis.
    """
    param_samples_path = Path(param_samples_path)
    gparam_path = Path(gparam_path)
    output_root = Path(output_root)

    if not param_samples_path.exists():
        raise FileNotFoundError(param_samples_path)
    if not gparam_path.exists():
        raise FileNotFoundError(gparam_path)

    # Load parameter samples: CSV with header line
    samples = np.loadtxt(param_samples_path, delimiter=",", skiprows=1)
    if samples.ndim == 1:
        samples = samples[None, :]

    if burnin < 0:
        burnin = 0
    if burnin > samples.shape[0]:
        burnin = 0
    samples = samples[burnin:, :]

    # First 6 columns correspond to the biophysical cell parameters
    cparams = np.mean(samples[:, 0:6], axis=0)

    # Build GCaMP model with these parameters
    import c_spikes.pgas.pgas_bound as pgas

    gparams = np.loadtxt(gparam_path)
    gcamp = pgas.GCaMP(gparams, cparams)

    # Default noise directory colocated with syn_gen module
    if noise_dir is None:
        noise_dir = Path(__file__).resolve().parent / "gt_noise_dir"
    else:
        noise_dir = Path(noise_dir)

    if tag is None:
        # Strip leading "param_samples_" if present to keep dataset stem
        stem = param_samples_path.stem
        tag = stem.replace("param_samples_", "")

    # normalize seeds to list for synth_gen
    seeds = None
    if noise_seed is None:
        seeds = None
    elif isinstance(noise_seed, (list, tuple)):
        seeds = list(noise_seed)
    else:
        seeds = noise_seed

    synth = synth_gen(
        spike_rate=spike_rate,
        spike_params=list(spike_params),
        cell_params=cparams,
        noise_dir=str(noise_dir),
        GCaMP_model=gcamp,
        tag=tag,
        plot_on=False,
        use_noise=True,
        noise_fraction=float(noise_fraction),
        noise_seed=seeds,
    )
    synth.generate(output_folder=str(output_root))

    if manifest_path is not None:
        from c_spikes.ens2.manifest import add_synthetic_entry

        manifest_path = Path(manifest_path)
        synth_dir = output_root / "Ground_truth" / f"synth_{tag}"
        if isinstance(noise_seed, (list, tuple)):
            noise_seed_serializable = list(noise_seed)
        else:
            noise_seed_serializable = noise_seed
        syn_params = {
            "tag": tag,
            "spike_rate": float(spike_rate),
            "spike_params": list(spike_params),
            "noise_dir": str(noise_dir),
            "noise_fraction": float(noise_fraction),
            "noise_seed": noise_seed_serializable,
            "output_dir": str(synth_dir),
        }
        add_synthetic_entry(
            manifest_path,
            model_name=manifest_model_name,
            param_samples_path=param_samples_path,
            gparam_path=gparam_path,
            burnin=burnin,
            cparams=cparams,
            syn_gen_params=syn_params,
            output_dir=synth_dir,
            run_tag=run_tag,
        )

    return cparams


__all__ = ["synth_gen", "build_synthetic_ground_truth_from_pgas", "build_synthetic_ground_truth_batch"]


def build_synthetic_ground_truth_batch(
    param_specs: Sequence[Dict[str, object]],
    *,
    gparam_path: Path | str = Path("src/c_spikes/pgas/20230525_gold.dat"),
    output_root: Path | str = Path("results"),
    manifest_path: Optional[Path | str] = None,
    manifest_model_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Batch helper to generate multiple synthetic datasets from multiple PGAS param_samples inputs.

    Each element of param_specs can contain:
        - param_samples_path (required)
        - burnin (int)
        - spike_rate (float)
        - spike_params (Sequence[float])
        - noise_dir (Path/str)
        - tag (str)
        - run_tag (str)

    Returns:
        Dict[tag, cparams] for each generated synthetic dataset.
    """
    cparams_map: Dict[str, np.ndarray] = {}
    for spec in param_specs:
        ppath = spec.get("param_samples_path")
        if ppath is None:
            raise ValueError("Each param_spec must include 'param_samples_path'.")
        burnin = int(spec.get("burnin", 100))
        spike_rate = float(spec.get("spike_rate", 2.0))
        spike_params = spec.get("spike_params", (5.0, 0.5))
        noise_dir = spec.get("noise_dir", None)
        noise_fraction = float(spec.get("noise_fraction", 1.0))
        noise_seed = spec.get("noise_seed", None)
        tag = spec.get("tag", None)
        run_tag = spec.get("run_tag", None)

        cparams = build_synthetic_ground_truth_from_pgas(
            ppath,
            gparam_path=gparam_path,
            burnin=burnin,
            spike_rate=spike_rate,
            spike_params=spike_params,  # type: ignore[arg-type]
            noise_dir=noise_dir,  # type: ignore[arg-type]
            noise_fraction=noise_fraction,
            noise_seed=noise_seed,  # type: ignore[arg-type]
            tag=tag,
            output_root=output_root,
            manifest_path=manifest_path,
            manifest_model_name=manifest_model_name,
            run_tag=run_tag,
        )
        final_tag = tag
        if final_tag is None:
            stem = Path(str(ppath)).stem
            final_tag = stem.replace("param_samples_", "")
        cparams_map[str(final_tag)] = cparams
    return cparams_map
