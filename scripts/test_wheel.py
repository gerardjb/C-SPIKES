"""Repair a wheel, install it in an isolated environment, and test its imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, env, cwd, timeout=None):
    command = [str(x) for x in command]
    print("+", subprocess.list2cmdline(command), flush=True)
    with subprocess.Popen(command, env=env, cwd=cwd) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Windows virtualenv launchers spawn the real interpreter as a
            # child. Stop the whole tree so a stuck CUDA kernel cannot survive.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.kill()
            process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--dependency-bin", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--skip-repair", action="store_true", help="Test an already repaired cibuildwheel artifact")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    wheel_dir = args.wheel_dir.resolve()
    (wheel_dir / "validation.json").unlink(missing_ok=True)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        parser.error(f"Expected exactly one wheel in {wheel_dir}, found {len(wheels)}")
    repaired = wheel_dir / "repaired"
    repaired.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    if platform.system() == "Windows":
        command = [sys.executable, "-m", "delvewheel", "repair", "-w", repaired]
        dll_paths = [args.dependency_bin, args.cuda_root / "bin" if args.cuda_root else None]
        if any(dll_paths):
            command.extend(["--add-path", os.pathsep.join(str(p.resolve()) for p in dll_paths if p)])
        if args.backend == "cuda":
            command.extend(["--exclude", "nvcuda.dll"])
        command.append(wheels[0])
    else:
        command = [sys.executable, "-m", "auditwheel", "repair", "-w", repaired, wheels[0]]
        if args.backend == "cuda":
            # The NVIDIA display driver belongs to the host. In particular,
            # never bundle WSL's libcuda proxy into a Linux wheel.
            command.extend(["--exclude", "libcuda.so.1"])
    if args.skip_repair:
        wheel = wheels[0]
    else:
        run(command, env=env, cwd=ROOT)
        repaired_wheels = sorted(repaired.glob("*.whl"))
        if len(repaired_wheels) != 1:
            parser.error("Expected exactly one repaired wheel; use a clean output directory")
        wheel = repaired_wheels[0]
    with zipfile.ZipFile(wheel) as package:
        names = package.namelist()
        if any(Path(name).name.lower().startswith(("nvcuda", "libcuda-", "libcuda.")) for name in names):
            raise RuntimeError("A host NVIDIA display-driver library was bundled in the wheel")
        has_gpu = any("/pgas_bound_gpu." in name for name in names)
        if has_gpu != (args.backend == "cuda"):
            raise RuntimeError("Wheel contents do not match the selected CPU/CUDA variant")
    # Deliberately remove dependency/toolchain lookup overrides before runtime
    # checks. Windows' system PATH (including the display driver) remains intact.
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CMAKE_PREFIX_PATH"):
        env.pop(key, None)
    for directory in (args.dependency_bin, args.cuda_root / "bin" if args.cuda_root else None):
        if not directory:
            continue
        dep_bin = os.path.normcase(str(directory.resolve()))
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                      if os.path.normcase(os.path.normpath(p)) != dep_bin)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", OMP_PROC_BIND="false", C_SPIKES_PGAS_BACKEND="cpu")
    with tempfile.TemporaryDirectory(prefix="c-spikes-wheel-") as temporary:
        work = Path(temporary)
        runtime = work / "venv"
        python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(["uv", "venv", "--python", sys.executable, runtime], env=env, cwd=work)
        run(["uv", "pip", "install", "--link-mode", "copy", "--python", python,
             wheel, "pytest>=8"], env=env, cwd=work)
        for backend in (["cpu", "gpu"] if args.backend == "cuda" else ["cpu"]):
            for import_order in ("numpy-first", "native-first"):
                run([python, ROOT / "tests/packaging/wheel_smoke.py", "--backend", backend,
                     "--import-order", import_order, "--prefix", runtime], env=env, cwd=work, timeout=120)
        run([python, "-m", "pytest", ROOT / "tests", "--csp-profile=native_core",
             "--junitxml", wheel_dir / "native-core.xml"], env=env, cwd=work, timeout=300)
        run([python, "-m", "pytest", ROOT / "tests", "--csp-profile=python_contracts",
             "--junitxml", wheel_dir / "python-contracts.xml"], env=env, cwd=work, timeout=300)
    evidence = {"wheel": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "backend": args.backend, "python": sys.version, "platform": platform.platform(),
                "checks": ["installed-origin", "numpy-import-order", "tiny-analyzer",
                           "oasis", "native_core", "python_contracts"]}
    (wheel_dir / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
