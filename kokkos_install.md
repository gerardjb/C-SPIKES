# HPC PGAS/Kokkos Install

The maintained HPC build path is now in `environment/`.

The Code Ocean Dockerfile is the canonical environment definition. It currently uses CUDA `12.4`, Kokkos `4.3.01`, Python `3.12.4`, and the pinned Python packages in `environment/requirements.txt`.

On HPC systems without apt access, use:

```bash
environment/hpc_bootstrap_deps.sh
C_SPIKES_CUDA_MODULE=cudatoolkit/12.4 environment/hpc_build.sh
```

`hpc_bootstrap_deps.sh` stages the native dependencies with vcpkg:

- `openblas[dynamic-arch]`
- `armadillo`
- `boost-circular-buffer`
- `gsl`
- `jsoncpp`

It also clones Kokkos at the same version used by the Dockerfile.

If CUDA `12.4` is unavailable on the cluster, set `C_SPIKES_CUDA_MODULE` to the closest compatible CUDA `12.x` module and set architecture overrides when needed:

```bash
C_SPIKES_CUDA_MODULE=cudatoolkit/12.6 \
C_SPIKES_KOKKOS_ARCH=AMPERE80 \
C_SPIKES_CUDA_ARCH=80 \
environment/hpc_build.sh
```

Both scripts support `--help`.
