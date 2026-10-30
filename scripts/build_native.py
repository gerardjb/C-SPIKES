"""Build CPU/CUDA wheels with the same dependency setup locally and in CI.

Prepare Python tools with ``uv sync --locked --no-install-project`` first.
Run with ``uv run --no-sync python scripts/build_native.py --help``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VCPKG_REVISION = "d186940b2f0c4e8e5b4ff804af57d2a26fbfa091"
ARCHITECTURES = {"TURING75": "75", "AMPERE80": "80", "AMPERE86": "86",
                 "ADA89": "89", "HOPPER90": "90"}


def run(command, *, env, cwd=ROOT):
    command = [str(x) for x in command]
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def msvc_environment(env):
    if shutil.which("cl", path=env["PATH"]):
        return env
    vswhere = Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        text=True, env=env).strip()
    if not installation:
        raise RuntimeError("Install Visual Studio C++ build tools, including an x64 Windows SDK")
    setup = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    # cmd.exe parses its command string differently from a C argv array; using
    # a list here causes Python to backslash-escape the batch file's quotes.
    output = subprocess.check_output(f'cmd /d /c call "{setup}" >nul && set',
                                     text=True, env=env)
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key and not key.startswith("="):
            # Windows keys are case-insensitive, but Python dictionaries are not.
            old_key = next((k for k in env if k.upper() == key.upper()), key)
            env[old_key] = value
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-arch", choices=ARCHITECTURES)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--vcpkg-root", type=Path, default=ROOT / "build/tools/vcpkg")
    parser.add_argument("--dependency-root", type=Path, default=ROOT / "build/vcpkg-installed")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--dependencies-only", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run CTest and installed-wheel tests; CUDA requires a GPU")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.backend == "cuda" and not args.cuda_arch:
        parser.error("CUDA builds require an explicit --cuda-arch")
    if args.backend == "cpu" and (args.cuda_arch or args.cuda_root):
        parser.error("CUDA options require --backend cuda")
    system = platform.system().lower()
    if system not in {"linux", "windows"} or platform.machine().lower() not in {"x86_64", "amd64"}:
        parser.error("This build profile currently supports Linux and Windows x86-64")

    env = os.environ.copy()
    if system == "linux" and "microsoft" in platform.release().lower():
        # Avoid CMake probing the inherited Windows PATH through WSL's slow
        # filesystem bridge (and accidentally finding Windows toolchains).
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if not p.startswith("/mnt/"))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    if system == "windows":
        env = msvc_environment(env)
    env.update(CMAKE_GENERATOR="Ninja", CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), VCPKG_MAX_CONCURRENCY=str(args.jobs))
    if args.cuda_root:
        env["PATH"] = str(args.cuda_root.resolve() / "bin") + os.pathsep + env["PATH"]
        if system == "linux":
            cuda = args.cuda_root.resolve()
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [str(cuda / "lib64"), str(cuda / "lib"), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    triplet = f"x64-{system}-csp"
    vcpkg_root = args.vcpkg_root.resolve()
    dependency_root = args.dependency_root.resolve()
    build_name = f"{system}-{args.backend}-py{sys.version_info.major}{sys.version_info.minor}"
    if args.cuda_arch:
        build_name += "-" + args.cuda_arch.lower()
    if args.diagnostics:
        build_name += "-diagnostics"
    build_dir = (args.build_dir or ROOT / "build" / build_name).resolve()
    output_dir = (args.output_dir or ROOT / "dist" / build_name).resolve()
    vcpkg = vcpkg_root / ("vcpkg.exe" if system == "windows" else "vcpkg")

    if not args.skip_dependencies:
        if not vcpkg_root.exists():
            run(["git", "clone", "https://github.com/microsoft/vcpkg.git", vcpkg_root], env=env)
            run(["git", "checkout", "--detach", VCPKG_REVISION], env=env, cwd=vcpkg_root)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vcpkg_root, env=env, text=True).strip()
        if revision != VCPKG_REVISION:
            parser.error(f"vcpkg tooling must be at {VCPKG_REVISION}; found {revision} in {vcpkg_root}. "
                         "Use a fresh --vcpkg-root, or --skip-dependencies for an explicitly prepared provider.")
        if not vcpkg.exists():
            bootstrap = vcpkg_root / ("bootstrap-vcpkg.bat" if system == "windows" else "bootstrap-vcpkg.sh")
            run((["cmd", "/c", bootstrap] if system == "windows" else ["bash", bootstrap]) + ["-disableMetrics"], env=env, cwd=vcpkg_root)
        run([vcpkg, "install", f"--triplet={triplet}", f"--host-triplet={triplet}",
             f"--overlay-triplets={ROOT / 'cmake/triplets'}",
             f"--overlay-ports={ROOT / 'cmake/ports'}",
             f"--x-install-root={dependency_root}"], env=env)
    if not (dependency_root / triplet / "share/lapack-reference/copyright").is_file():
        parser.error(f"Native dependencies are missing from {dependency_root / triplet}")
    if args.dependencies_only:
        return

    settings = {
        "build-dir": build_dir.as_posix(),
        "cmake.define.CMAKE_TOOLCHAIN_FILE": (vcpkg_root / "scripts/buildsystems/vcpkg.cmake").as_posix(),
        "cmake.define.VCPKG_INSTALLED_DIR": dependency_root.as_posix(),
        "cmake.define.VCPKG_TARGET_TRIPLET": triplet,
        "cmake.define.VCPKG_MANIFEST_MODE": "OFF",
        "cmake.define.PGAS_BUILD_GPU": "ON" if args.backend == "cuda" else "OFF",
        "cmake.define.Kokkos_ENABLE_CUDA": "ON" if args.backend == "cuda" else "OFF",
        "cmake.define.CSP_NUMERICAL_PROFILE": "portable-lp64",
        "cmake.define.C_SPIKES_TEST_RESAMPLING": "ON",
        "cmake.define.C_SPIKES_OBSERVATIONAL_DIAGNOSTICS": "ON" if args.diagnostics else "OFF",
    }
    if args.backend == "cuda":
        settings[f"cmake.define.Kokkos_ARCH_{args.cuda_arch}"] = "ON"
        settings["cmake.define.CMAKE_CUDA_ARCHITECTURES"] = ARCHITECTURES[args.cuda_arch]
        nvcc = shutil.which("nvcc", path=env["PATH"])
        if not nvcc:
            parser.error("nvcc was not found; install a CUDA toolkit or pass --cuda-root")
        settings["cmake.define.CMAKE_CUDA_COMPILER"] = Path(nvcc).as_posix()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").unlink(missing_ok=True)
    command = [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", output_dir]
    command.extend(f"-C{key}={value}" for key, value in settings.items())
    run(command, env=env)
    (output_dir / "build-settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    if args.test:
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", OMP_PROC_BIND="false", C_SPIKES_PGAS_BACKEND="cpu")
        if system == "windows":
            env["PATH"] = str(dependency_root / triplet / "bin") + os.pathsep + env["PATH"]
        run(["ctest", "--test-dir", build_dir, "--output-on-failure", "--timeout", "120",
             "--output-junit", output_dir / "ctest.xml"], env=env)
        # Wheel repair and isolated runtime installation are handled by the
        # wheel validator so dependency discovery cannot mask missing DLLs.
        validator = [sys.executable, ROOT / "scripts/test_wheel.py", "--wheel-dir", output_dir,
                     "--dependency-bin", dependency_root / triplet / "bin", "--backend", args.backend]
        if args.cuda_root:
            validator.extend(["--cuda-root", args.cuda_root.resolve()])
        run(validator, env=env)


if __name__ == "__main__":
    main()
