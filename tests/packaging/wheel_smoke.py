"""Fresh-process installed-wheel checks, including explicit CUDA execution."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--import-order", choices=("numpy-first", "native-first"), required=True)
    parser.add_argument("--prefix", type=Path, default=Path(sys.prefix))
    args = parser.parse_args()
    if args.import_order == "numpy-first":
        import numpy
        import scipy.linalg
    pgas = importlib.import_module(f"c_spikes.pgas.pgas_bound_{args.backend}")
    import numpy as np
    import scipy.linalg
    from c_spikes.oasis import oasis_methods

    for module in (pgas, oasis_methods):
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(args.prefix.resolve()):
            raise RuntimeError(f"Loaded {module.__name__} outside installed prefix: {origin}")
        print(f"Imported {module.__name__} from {origin}", flush=True)
    # Nontrivial NumPy/SciPy calls also force their numerical providers to load.
    np.testing.assert_allclose(scipy.linalg.svdvals(np.eye(9)), np.ones(9))
    fixture = Path(__file__).resolve().parents[1] / "fixtures/native_core"
    with tempfile.TemporaryDirectory() as temporary:
        model = pgas.Analyzer(time=np.array([0., .001, .002]), data=np.zeros(3),
                              constants_file=str(fixture / "analyzer.json"),
                              output_folder=temporary, column=0, tag="wheel", niter=2,
                              trainedPriorFile="", append=False, trim=1, verbose=False,
                              gtSpikes=np.array([0.]), has_trained_priors=False,
                              has_gtspikes=False, maxlen=0,
                              Gparam_file=str(fixture / "gparams.dat"), seed=17)
        model.run()
        if model.get_run_status() != "completed":
            raise RuntimeError(model.get_run_message())
        samples = model.get_parameter_samples()
        if samples.shape != (2, 6) or not np.isfinite(samples).all():
            raise RuntimeError("Invalid native Analyzer output")
        trajectory = np.loadtxt(Path(temporary) / "traj_samples_wheel.dat", delimiter=",", skiprows=1)
        if trajectory.shape != (6, 6) or not np.isfinite(trajectory).all():
            raise RuntimeError("Invalid native trajectory output")
    trace = np.zeros(32)
    trace[8:] = .95 ** np.arange(24)
    calcium, events = oasis_methods.oasisAR1(trace, .95)
    if calcium.shape != trace.shape or events.shape != trace.shape or not np.isfinite(events).all():
        raise RuntimeError("Invalid OASIS output")
    print(f"PASS: {args.backend}, {args.import_order}", flush=True)


if __name__ == "__main__":
    main()
