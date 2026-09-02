# OASIS provenance

The files in this directory were imported from the local `oasis_port` comparator repository at
commit `e738431502040ad7db8f79a12b2927ae9d2f4e7c`.

Imported source hashes before C-SPIKES-specific changes:

| File | SHA-256 |
|---|---|
| `__init__.py` | `f520d902108c742cf849e30ee006f1a53c31b5ce8b03d0d8254ffbbc6b85cab7` |
| `functions.py` | `233881be138dfed146d3fe8c59ff9737c8b99bd6992d968d2cd55b6ba79bd648` |
| `oasis_methods.pyx` | `980d3dede6eab56f459d90875bd377425c1a8d88507a2d2e00128e06520c9b4f` |
| `oasis_methods.cpp` | `f067303448559561888b33349f5fc5f50173d5f0e01292c054e1545dcde8e52e` |

The implementation is attributed to Johannes Friedrich. The algorithm references retained in the
source include:

- Friedrich J. and Paninski L., NIPS 2016.
- Friedrich J., Zhou P., and Paninski L., PLOS Computational Biology 2017.

Both the source comparator and C-SPIKES are distributed under GPL-3. See the repository-level
`LICENSE` file.

## Local modifications

Local changes must be recorded here when they are made. The initial integration:

- adds a conditional compatibility preamble for the six dtype and six multi-iterator accessors
  emitted by Cython's NumPy 2 declarations, so one generated C++ source also compiles against
  NumPy 1.x headers;
- makes three binary-search midpoint divisions explicitly integral (`//`) and records the original
  global C-division behavior as a Cython source directive;
- integrates the generated extension with the C-SPIKES CMake/scikit-build package; and
- keeps Cython regeneration as a maintainer-only operation rather than a normal build step.

The source hashes after the initial build integration were:

| File | SHA-256 |
|---|---|
| `oasis_methods.pyx` | `2a64a8fdff931ae7b6e8391383ed1738b0db9210cff601152bb758f71804ee62` |
| `oasis_methods.cpp` | `632c51ae0e59f7f12b3ef451e964dbb2705a7c07c628ad7f2f1cabcb3430b688` |

## Regeneration and NumPy compatibility

`oasis_methods.cpp` was regenerated with Cython 3.2.4 while NumPy 2.1.3 supplied the Cython NumPy
declarations. From the repository root, the equivalent maintainer command is:

```bash
python -c 'from setuptools import Extension; from Cython.Build import cythonize; cythonize([Extension("c_spikes.oasis.oasis_methods", ["src/c_spikes/oasis/oasis_methods.pyx"], language="c++")], force=True)'
```

Normal builds compile the checked-in C++ and do not install or invoke Cython. Isolated builds use
NumPy 2.x headers on Python 3.9 and newer so their extension can run with both NumPy 1.x and 2.x.
Python 3.8, which cannot install NumPy 2, uses NumPy 1.24.4 at build time.

The integrated C++ was compiled and imported successfully with NumPy 1.26.4, 2.1.3, and 2.4.2
headers. A NumPy-2-header build was also imported under a NumPy 1.26.4 runtime. The Python 3.8 /
NumPy 1.24.4 source-build fallback was compile-tested separately.

## Numerical hardening

The second integration slice deliberately changed the numerical facade while retaining golden
parity for fixed, valid AR(1) and AR(2) calls. It:

- validates trace shape, dtype, finiteness, AR coefficients, scalar parameters, penalty, optimization,
  and decimation choices before entering a solver;
- implements supplied baselines by subtracting them before inference and returning them unchanged;
- prevents AR(2) optimization from mutating caller-owned coefficients;
- replaces random repair of unstable estimated roots with deterministic values;
- estimates decimated native parameters on the largest complete prefix while solving the full trace;
  and
- deliberately retains Python ONNLS as the facade's AR(2) backend and the low-level facade's legacy
  `penalty=0` default. The C-SPIKES adapter selects `penalty=1` explicitly.

The C++ was regenerated again with the command and Cython/NumPy versions above. Current hashes are:

| File | SHA-256 |
|---|---|
| `functions.py` | `a78b378b2691723b6e4baeecac9079a06c5f11e4680b6c5e967e48a6269b3314` |
| `oasis_methods.pyx` | `f80dd88de95d14ff5e71a1838e50db14507d8a945ab01e06ffcd6953026913cf` |
| `oasis_methods.cpp` | `8c96ddfaebda46d625368d2a410ddeec254c20385060ce0f117fe5e71e396d0d` |
