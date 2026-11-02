"""Installed-wheel checks for lazy loading, independent runtimes and selection."""
import argparse
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--first", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.no_gpu:
        # Windows' C runtime treats an empty environment value as deletion.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.pop("C_SPIKES_PGAS_BACKEND", None)
    from c_spikes.pgas import get_backend, pgas_bound
    assert not any(f"c_spikes.pgas.pgas_bound_{name}" in sys.modules for name in ("cpu", "gpu"))
    assert pgas_bound.__backend__ == "cpu"
    import numpy as np
    fixture = Path(__file__).resolve().parents[1] / "fixtures/native_core"

    with tempfile.TemporaryDirectory() as temporary:
        models = []
        order = [args.first, "cuda" if args.first == "cpu" else "cpu"] if args.gpu else ["cpu"]
        for backend in order:
            module = get_backend(backend)
            assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
            models.append(module.Analyzer(
                time=np.array([0., .001, .002]), data=np.zeros(3),
                constants_file=str(fixture / "analyzer.json"), output_folder=temporary,
                column=0, tag=backend, niter=2, trainedPriorFile="", append=False,
                trim=1, verbose=False, gtSpikes=np.array([0.]), has_trained_priors=False,
                has_gtspikes=False, maxlen=0, Gparam_file=str(fixture / "gparams.dat"), seed=17))
        if args.no_gpu:
            try:
                get_backend("cuda")
            except ImportError as exc:
                assert "CUDA backend unavailable" in str(exc)
            else:
                raise AssertionError("CUDA unexpectedly available")
        # Keep both objects alive, and run the first again after the second.
        runs = {}
        for model in [*models, models[0]]:
            model.run()
            assert model.get_run_status() == "completed", model.get_run_message()
            samples = model.get_parameter_samples()
            runs[id(model)] = runs.get(id(model), 0) + 1
            # Analyzer retains parameter records across repeated run() calls.
            assert samples.shape == (2 * runs[id(model)], 6) and np.isfinite(samples).all()
        assert get_backend() is get_backend("cpu")
        assert pgas_bound.Analyzer is get_backend("cpu").Analyzer
        os.environ["C_SPIKES_PGAS_BACKEND"] = "cpu"
        assert get_backend().Analyzer is pgas_bound.Analyzer
    print(f"PASS: combined wheel, order={order}, unavailable-GPU check={args.no_gpu}")


if __name__ == "__main__":
    main()
