"""
Lightweight wrapper around the ENS2 training/inference code to mirror the
CASCADE-style API (train_model / predict) and make it easy to point ENS2 at
synthetic ground-truth datasets produced by syn_gen.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _stage_ground_truth(synth_dirs: Sequence[Path], stage_root: Path) -> Sequence[Path]:
    """
    ENS2's training code expects relative paths ./ground_truth/DS*.
    ENS2 also assumes DS numbering starts at 1 and skips DS1 for training.

    We therefore create an empty DS1 placeholder and place each synthetic
    directory under a separate DS index starting at 2 to align with the
    ENS2 Exc/Inh index conventions.
    """
    stage_root = stage_root.resolve()
    gt_root = stage_root / "ground_truth"
    gt_root.mkdir(parents=True, exist_ok=True)
    # Placeholder to keep ENS2 dataset indices aligned (DS1 is ignored in training).
    (gt_root / "DS1").mkdir(parents=True, exist_ok=True)

    staged_dirs = []
    for idx, synth_dir in enumerate(synth_dirs, start=2):
        ds_dir = gt_root / f"DS{idx}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        synth_dir = synth_dir.resolve()
        for mat in synth_dir.glob("*.mat"):
            target = ds_dir / mat.name
            if not target.exists():
                try:
                    target.symlink_to(mat)
                except FileExistsError:
                    pass
        staged_dirs.append(ds_dir)
    return staged_dirs


def _select_checkpoint(saved_model_dir: Path) -> Optional[Path]:
    candidates = sorted(saved_model_dir.glob("C_*Epoch*.pt"))
    return candidates[-1] if candidates else None


def _load_ens2_module():
    import importlib

    return importlib.import_module("c_spikes.ens2.ENS2")


def train_model(
    model_name: str,
    synth_gt_dir: str | Path | Sequence[str | Path],
    *,
    model_root: str | Path = Path("results/Pretrained_models"),
    neuron_type: str = "Exc",
    sampling_rate: float = 60.0,
    smoothing_std: float = 0.025,
    verbose: int = 1,
    manifest_path: str | Path | None = None,
    run_tag: str | None = None,
) -> Path:
    """
    Train ENS2 on a synthetic ground-truth folder (output of syn_gen).

    Args:
        model_name: Destination subfolder under model_root.
        synth_gt_dir: Path (or list of paths) to synthetic ground-truth mat files (e.g., results/Ground_truth/synth_<tag>).
        model_root: Root under which to place the training artifacts/checkpoints.
        neuron_type: "Exc" or "Inh".
        sampling_rate: Target sampling rate for ENS2 (Hz).
        smoothing_std: Smoothing std (seconds) for ENS2 labels.
        verbose: Verbosity flag passed through to ENS2 (0=silent).
        manifest_path: Where to write the manifest for this custom model (defaults to model_dir/ens2_manifest.json).
        run_tag: Optional tag to carry through into the manifest.

    Returns:
        Path to the published-format checkpoint (exc_ens2_pub.pt or inh_ens2_pub.pt).
    """
    ens2_mod = _load_ens2_module()
    opt = ens2_mod.opt
    opt.sampling_rate = float(sampling_rate)
    opt.smoothing_std = float(smoothing_std)
    opt.smoothing = opt.smoothing_std * opt.sampling_rate

    model_root = Path(model_root).resolve()
    model_dir = model_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path) if manifest_path is not None else model_dir / "ens2_manifest.json"

    # Stage ground truth as ground_truth/DS2+ inside the model directory (DS1 placeholder created automatically).
    synth_dirs: Sequence[Path]
    if isinstance(synth_gt_dir, (str, Path)):
        synth_dirs = [Path(synth_gt_dir)]
    else:
        synth_dirs = [Path(p) for p in synth_gt_dir]
    staged_dirs = _stage_ground_truth(synth_dirs, model_dir)

    checkpoint_name = "exc_ens2_pub.pt" if neuron_type.lower().startswith("exc") else "inh_ens2_pub.pt"

    # Run training with CWD set to model_dir so ENS2 finds ./ground_truth/DS*.
    with _pushd(model_dir):
        ens = ens2_mod.ENS2()
        ens.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # ENS2.train signature is largely string-based; convert to match.
        ens.train(
            neuron=neuron_type,
            Fs=str(opt.sampling_rate),
            smoothing_std=str(opt.smoothing_std),
            verbose=verbose,
        )
        saved_model_dir = Path("saved_model")
        latest = _select_checkpoint(saved_model_dir)
        if latest is None:
            raise RuntimeError("ENS2 training did not produce a checkpoint in ./saved_model")
        target = Path(checkpoint_name)
        shutil.copy2(latest, target)
        checkpoint_path = model_dir / target

        # Record training metadata
        try:
            from c_spikes.ens2.manifest import add_training_entry

            add_training_entry(
                manifest_path,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                neuron_type=neuron_type,
                sampling_rate=opt.sampling_rate,
                smoothing_std=opt.smoothing_std,
                run_tag=run_tag,
                ground_truth_dir=[str(p) for p in staged_dirs],
                device=ens.DEVICE,
            )
        except Exception:
            # Keep training success even if manifest write fails
            pass

        return checkpoint_path


def predict(
    model_dir: str | Path,
    traces: np.ndarray,
    *,
    neuron_type: str = "Exc",
    sampling_rate: float = 60.0,
    smoothing_std: float = 0.025,
    trial_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[float]]:
    """
    Run ENS2 inference on provided traces using a checkpoint stored in model_dir.

    Args:
        model_dir: Directory containing exc_ens2_pub.pt or inh_ens2_pub.pt.
        traces: Array of shape (trials, timepoints) with dFF traces.
        neuron_type: "Exc" or "Inh" (selects checkpoint name).
        sampling_rate: Trace sampling rate in Hz.
        smoothing_std: ENS2 smoothing std in seconds (sets opt).
        trial_time: Optional duration override in seconds; if None, inferred from len(trace)/sampling_rate.

    Returns:
        calcium (center sample), pd_rate, pd_spike, pd_event from ENS2.predict.
    """
    ens2_mod = _load_ens2_module()
    opt = ens2_mod.opt
    opt.sampling_rate = float(sampling_rate)
    opt.smoothing_std = float(smoothing_std)
    opt.smoothing = opt.smoothing_std * opt.sampling_rate

    model_dir = Path(model_dir).resolve()
    checkpoint_name = "exc_ens2_pub.pt" if neuron_type.lower().startswith("exc") else "inh_ens2_pub.pt"
    checkpoint_path = model_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if trial_time is None:
        # assume uniform sampling
        trial_time = traces.shape[-1] / float(sampling_rate)

    test_data = ens2_mod.compile_test_data(traces, trial_time)

    load_kwargs = {"map_location": torch.device(device)}
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    ens = ens2_mod.ENS2()
    ens.DEVICE = device
    return ens.predict(test_data, state_dict=state_dict)


__all__ = ["train_model", "predict"]
