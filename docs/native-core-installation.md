# Linux native-core installation

`CSP_NATIVE_CORE_INSTALL=ON` enables the opt-in `CSPNativeCore` install component.
It installs the PGAS CPU and OASIS native extensions, their minimal Python packages,
and license/provenance notices. It does not install synthetic generation, GUI,
training or the full package. This is the historical installation profile; use
[build and release](build-and-release.md) for portable CPU/CUDA wheels.

Add these options to an explicitly configured CPU/provider build:

```text
-DCSP_NATIVE_CORE_INSTALL=ON
-DCSP_NUMERICAL_PROFILE=linux-x86_64-lp64-serial
-DCSP_NATIVE_CORE_NOTICES_DIR=<reviewed-notice-directory>
```

The notice directory must contain `MANIFEST.json` and the notices for the actual
dependencies used. CMake checks that the manifest exists; the packager must verify
its content and hashes. The test fixture notice set describes one recorded build,
not an assurance that it covers every future dependency combination.

After building, install this component:

```text
cmake --install <build> --prefix <prefix> --component CSPNativeCore
```

This profile rejects non-Linux, non-x86_64 and GPU configurations. PGAS/OASIS
installed modules have empty install RPATH and do not inherit build/link paths.
Numerical runtimes remain external; the recorded profile used libgfortran.so.3,
libgomp, libstdc++, libquadmath and the system C runtime. Do not treat that set as
a universal Linux ABI contract or a minimum glibc claim. See
[numerical provider requirements](numerical-provider-linux.md).

Default and checked PGAS are alternative builds in separate prefixes and fresh
interpreters. Checked GSL remains opt-in. If the checked build omits OASIS, assemble
that prefix explicitly with the identical newly built OASIS extension from the
matching ordinary build. No missing-extension fallback is performed.

## Verification and limits

The recorded Linux CPU installation checks exercised installed imports, NumPy
import order, numerical-symbol collisions, OASIS golden outputs, ownership,
notices and relocation to a second prefix with the first removed. They used
CPython 3.10.16, NumPy 2.1.3, SciPy 1.15.3 and a pinned LP64 serial provider profile.
Those versions describe the tested environment, not universal dependency minima.

The repository does not currently supply a complete installation/relocation test
controller. The notice inputs are in `tests/packaging/native_install/notices`.
For installation validation, import origins must resolve to the installed prefix
from an empty working directory. Preserve installed bytes across relocation;
build-tree and installed hashes may differ when CMake removes RPATH. Verify that
transformation rather than accepting arbitrary differences.

Changes to the source, diagnostic mode, dependency set or test adapter require
affected validation. This profile is not a wheel builder and does not establish
Python 3.11/3.12, arbitrary Linux distribution, GPU installation, optional ML/GUI,
or full-inference support. Native source/rebuild support and a working prefix do
not by themselves provide a redistributable wheel.
