# C-SPIKES Capsule Environment

`Dockerfile` is the canonical Code Ocean environment. The current image is based on:

- `registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04`
- Kokkos `4.3.01`
- CUDA architecture `75` / Kokkos architecture `TURING75` by default
- Python packages pinned in `environment/requirements.txt`

The Docker build installs the native PGAS dependencies from apt:

- OpenBLAS
- Armadillo
- Boost
- GSL
- JsonCpp

## HPC Build

On HPC systems without apt access, use the helper scripts in this directory. The bootstrap script stages the native dependencies with vcpkg and clones the same Kokkos version used by the Dockerfile:

```bash
environment/hpc_bootstrap_deps.sh
```

Then build the package in the active/target conda environment:

```bash
C_SPIKES_CUDA_MODULE=cudatoolkit/12.4 environment/hpc_build.sh
```

If CUDA `12.4` is unavailable on the cluster, set `C_SPIKES_CUDA_MODULE` to the closest compatible CUDA `12.x` module. Without an explicit module, `hpc_build.sh` tries `12.4`, then newer `12.x` modules.

Useful overrides:

- `C_SPIKES_CONDA_ENV`: conda environment to activate, default `c_spikes_co`
- `C_SPIKES_KOKKOS_ARCH`: Kokkos GPU architecture, for example `TURING75`, `AMPERE80`, `AMPERE86`, `ADA89`, or `HOPPER90`
- `C_SPIKES_CUDA_ARCH`: CUDA architecture number, for example `75`, `80`, `86`, `89`, or `90`
- `C_SPIKES_DEPS_ROOT`: dependency cache root, default `$SCRATCH/c_spikes_deps`
- `C_SPIKES_INSTALL_REQUIREMENTS=1`: install `environment/requirements.txt` before building
- `C_SPIKES_RUN_TESTS=1`: run import tests after building

Both scripts support `--help` for the full option summary.
