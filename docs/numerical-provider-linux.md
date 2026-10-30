# Linux numerical-provider boundary

This page describes the historical, hash-qualified profile. Select it explicitly
with `-DCSP_NUMERICAL_PROFILE=linux-x86_64-lp64-serial`. New wheel builds default to
the separate `portable-lp64` profile described in [build and release](build-and-release.md).
Portable build/test results do not transfer the historical numerical acceptance.

The `linux-x86_64-lp64-serial` CMake profile pins serial LP64 OpenBLAS, reference
LAPACK and the existing Fortran runtime by SHA-256. Set
`CSP_NUMERICAL_PROVIDER_ROOT` to the matching installation. Unsupported profiles
fail configuration; there is no automatic fallback or dependency installation.
The native module no longer links libarmadillo.

`cmake/ExtractPinnedHandler.py` selects the original OpenBLAS error-handler object.
The archive has two members named `xerbla.c.o`; only the second defines weak
`xerbla_` and strong `__xerbla`. Archive hash, member position/name/size and object
hash identify it. Its bytes are linked unchanged before archive scanning, without
a forced-undefined linker flag. Different archives require new provenance and
numerical checks; matching symbol names or sizes is insufficient.

Default DGEMV/DGESDD behavior is preserved. GSL links to its intended GSL CBLAS,
not OpenBLAS CBLAS. Checked GSL remains opt-in. Link-map and symbol/relocation
checks establish ownership; numerical tests separately check resulting values.

The ELF extension export list exposes only `PyInit_...`. Numerical-provider
functions and GSL state are module-private, preventing a Python host's global
BLAS/GSL symbols from replacing them. Native source interfaces remain available
through recompilation; accidental C++/BLAS/GSL exports are not a supported ABI.
Exchanging foreign GSL objects or assuming concurrent access to shared models is
unsupported. NumPy import-order and collision-sentinel checks are needed alongside
numerical tests when admitting a different host environment.

The profile is deliberately specific to Linux x86-64 and its pinned runtime,
including the existing Fortran dependency. ELF controls do not establish macOS or
Windows support. Wheel bundling, relocation, CUDA execution, other architectures
and arbitrary embedded hosts each require checks of their actual build/runtime.
A successful configuration alone is not qualification of those environments.
