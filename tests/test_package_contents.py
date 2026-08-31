from pathlib import Path

import c_spikes.syn_gen as syn_gen
from c_spikes.biophys_ml import (
    default_cascade_train_config,
    default_ens2_train_config,
    default_synthetic_config,
    generate_synthetic_bundles,
    train_models_for_bundles,
)


def test_biophys_ml_runtime_package_is_importable():
    assert callable(default_cascade_train_config)
    assert callable(default_ens2_train_config)
    assert callable(default_synthetic_config)
    assert callable(generate_synthetic_bundles)
    assert callable(train_models_for_bundles)


def test_gui_application_imports_with_packaged_dependencies(monkeypatch):
    monkeypatch.setenv("C_SPIKES_TF_QUIET_IMPORT", "0")
    from c_spikes.gui import app

    assert callable(app.main)


def test_default_synthetic_noise_data_is_packaged():
    noise_dir = Path(syn_gen.__file__).resolve().parent / "gt_noise_dir"

    assert noise_dir.is_dir()
    assert list(noise_dir.glob("*.mat"))
