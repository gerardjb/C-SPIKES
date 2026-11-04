"""Exercise default resources from an unrelated cwd in source and wheel tests."""
import json
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import pytest

from c_spikes.biophys_ml import pipeline as biophys
from c_spikes.cli.run import parse_args
from c_spikes.gui.paths import resolve_pgas_parameter_dirs
from c_spikes.pipeline import RunConfig
import c_spikes.syn_gen as syn_gen
from c_spikes import tensorflow_env


def test_cascade_configuration_uses_declared_yaml_dependency(tmp_path):
    # uv-created wheel-test environments do not include pip. Loading model
    # configuration must need only the declared ruamel.yaml dependency.
    from c_spikes.cascade2p.config import read_config, write_config

    path = tmp_path / "config.yaml"
    write_config({"model_name": "test", "sampling_rate": 60.}, path)
    values = read_config(path)
    assert values["model_name"] == "test" and values["sampling_rate"] == 60.


def test_gui_module_launches_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = types.ModuleType("c_spikes.gui.app")
    launched = []
    app.main = lambda: launched.append(True)
    monkeypatch.setitem(sys.modules, "c_spikes.gui.app", app)
    runpy.run_module("c_spikes.gui", run_name="__main__")
    assert launched == [True]


def test_optional_tensorflow_preload_skips_missing_package(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setenv("C_SPIKES_TF_QUIET_IMPORT", "1")
    monkeypatch.setattr(tensorflow_env.importlib.util, "find_spec", lambda name: None)

    def unexpected_import(name):
        raise AssertionError("A missing optional dependency must not be imported")

    monkeypatch.setattr(tensorflow_env.importlib, "import_module", unexpected_import)
    tensorflow_env.preload_tensorflow_quietly()


def test_optional_tensorflow_preload_reports_broken_installation(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setenv("C_SPIKES_TF_QUIET_IMPORT", "1")
    monkeypatch.setattr(tensorflow_env.importlib.util, "find_spec", lambda name: object())

    def broken_import(name):
        raise ImportError("broken TensorFlow installation")

    monkeypatch.setattr(tensorflow_env.importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="broken TensorFlow"):
        tensorflow_env.preload_tensorflow_quietly()


def test_default_inputs_are_readable_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = RunConfig()
    args = parse_args([])
    constants_dir, gparam_dir = resolve_pgas_parameter_dirs(tmp_path)

    for constants in (config.pgas_constants, args.pgas_constants,
                      constants_dir / "constants_GCaMP8_soma.json",
                      constants_dir / "constants_GCaMP8m_soma.json"):
        assert constants.is_absolute()
        values = json.loads(constants.read_text())
        assert values["MCMC"]["nparticles"] > 0
        assert values["data"]["sampling_frequency"] > 0
    for gparam in (config.pgas_gparam, args.pgas_gparam,
                   Path(biophys.default_synthetic_config()["gparam_path"]),
                   gparam_dir / "20230525_gold.dat"):
        assert gparam.is_absolute()
        assert np.isfinite(np.loadtxt(gparam)).all()


def test_explicit_parameter_paths_are_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    constants, gparam = Path("custom.json"), Path("custom.dat")
    args = parse_args(["--pgas-constants", str(constants), "--pgas-gparam", str(gparam)])
    config = RunConfig(pgas_constants=constants, pgas_gparam=gparam)
    assert args.pgas_constants == config.pgas_constants == constants
    assert args.pgas_gparam == config.pgas_gparam == gparam


def test_gui_project_parameters_override_bundled_defaults(tmp_path):
    constants_dir = tmp_path / "parameter_files"
    constants_dir.mkdir()
    # A project can override just one of the two resource directories.
    constants, gparams = resolve_pgas_parameter_dirs(tmp_path)
    assert constants == constants_dir
    assert (gparams / "20230525_gold.dat").is_file()
    custom_gparams = tmp_path / "src" / "c_spikes" / "pgas"
    custom_gparams.mkdir(parents=True)
    assert resolve_pgas_parameter_dirs(tmp_path) == (constants_dir, custom_gparams)


@pytest.mark.parametrize("entrypoint", ["single", "batch", "biophys", "biophys-implicit"])
def test_synthetic_defaults_initialize_native_model(tmp_path, monkeypatch, entrypoint):
    monkeypatch.chdir(tmp_path)
    samples = tmp_path / "param_samples_example.dat"
    expected = np.array([30e-6, 1000., 1e-5, 5., 30., 10.])
    np.savetxt(samples, expected[None, :], delimiter=",", header="G_tot,gamma,DCaT,Rf,gam_in,gam_out")
    calls = []

    def tiny_generate(self, output_folder):
        # Use the real native GCaMP model and packaged sensor/noise inputs;
        # replace only the expensive full-dataset generation loop.
        assert list(Path(self.noise_dir).glob("*.mat"))
        self.gcamp.init()
        self.gcamp.integrateOverTime(np.array([0., .001, .002]), np.array([0.]))
        assert np.isfinite(self.gcamp.getDFFValues()).all()
        calls.append(output_folder)

    monkeypatch.setattr(syn_gen.synth_gen, "generate", tiny_generate)
    if entrypoint == "single":
        values = syn_gen.build_synthetic_ground_truth_from_pgas(samples)
        np.testing.assert_allclose(values, expected)
    elif entrypoint == "batch":
        values = syn_gen.build_synthetic_ground_truth_batch([{"param_samples_path": samples}])
        np.testing.assert_allclose(values["example"], expected)
    else:
        config = biophys.default_synthetic_config()
        config.update(spike_rate_values=[2.], smooth_values=[1.3], duty_values=[.35])
        if entrypoint == "biophys-implicit":
            config.pop("gparam_path")
        bundles = biophys.generate_synthetic_bundles(
            param_samples_paths=[samples], run_root=tmp_path / "output",
            run_tag="test", synthetic_config=config)
        assert len(bundles) == 1
        np.testing.assert_allclose(next(iter(bundles[0]["cparams_map"].values())), expected)
    assert len(calls) == 1
