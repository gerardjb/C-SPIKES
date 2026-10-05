import pytest

from c_spikes.inference.pgas import resampling_provenance


@pytest.fixture(autouse=True)
def clear_options(monkeypatch):
    monkeypatch.delenv("C_SPIKES_PGAS_RESAMPLING", raising=False)
    monkeypatch.delenv("C_SPIKES_PGAS_ANCESTOR_SEED", raising=False)


def test_default_and_explicit_host_match(monkeypatch):
    default = resampling_provenance()
    monkeypatch.setenv("C_SPIKES_PGAS_RESAMPLING", "host")
    assert default == resampling_provenance()
    assert default["ancestor_rng"] == "gsl-mt19937-alias"


def test_device_identity_and_seed(monkeypatch):
    monkeypatch.setenv("C_SPIKES_PGAS_RESAMPLING", "device")
    inferred = resampling_provenance()
    monkeypatch.setenv("C_SPIKES_PGAS_ANCESTOR_SEED", str(2**64 - 1))
    explicit = resampling_provenance()
    assert inferred != explicit
    assert explicit["ancestor_seed"] == str(2**64 - 1)
    assert explicit["ancestor_rng"] == "philox4x32-10-v1"


@pytest.mark.parametrize("seed", ["", "-1", "+1", "1.0", " 1", str(2**64), "１２"])
def test_invalid_seed(monkeypatch, seed):
    monkeypatch.setenv("C_SPIKES_PGAS_RESAMPLING", "device")
    monkeypatch.setenv("C_SPIKES_PGAS_ANCESTOR_SEED", seed)
    with pytest.raises(ValueError):
        resampling_provenance()


def test_unused_seed_and_unknown_mode_rejected(monkeypatch):
    monkeypatch.setenv("C_SPIKES_PGAS_ANCESTOR_SEED", "1")
    with pytest.raises(ValueError):
        resampling_provenance()

    monkeypatch.setenv("C_SPIKES_PGAS_RESAMPLING", "automatic")
    with pytest.raises(ValueError):
        resampling_provenance()


def test_legacy_host_lookup_rejects_device_cache(tmp_path):
    import numpy as np
    from c_spikes.inference.cache import save_method_cache, load_method_cache_legacy_compatible
    from c_spikes.inference.types import MethodResult
    result = MethodResult("pgas", np.arange(3.), np.zeros(3), 1.)
    device = {"niter": 200, "resampling": {"mode": "device"}}
    save_method_cache("pgas", "fixture", result, device, "trace", cache_root=tmp_path)
    assert load_method_cache_legacy_compatible("pgas", ["fixture"], {"niter": 200},
        "trace", stable_config_keys=["niter"], cache_root=tmp_path) is None
    assert load_method_cache_legacy_compatible("pgas", ["fixture"], device,
        "trace", stable_config_keys=["niter"], cache_root=tmp_path) is not None
