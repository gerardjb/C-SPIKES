"""Check native GPU images and PTX in repaired wheels without requiring a GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

from build_native import ARCHITECTURES


def check_wheel(wheel, cuda_root, architecture="PORTABLE"):
    expected = set(ARCHITECTURES.values()) if architecture == "PORTABLE" else {ARCHITECTURES[architecture]}
    baseline = "75" if architecture == "PORTABLE" else ARCHITECTURES[architecture]
    cuobjdump = cuda_root / "bin" / ("cuobjdump.exe" if (cuda_root / "bin/cuobjdump.exe").exists() else "cuobjdump")
    with zipfile.ZipFile(wheel) as package, tempfile.TemporaryDirectory() as temporary:
        names = package.namelist()
        modules = [name for name in names if re.search(r"/pgas_bound_(cpu|gpu)\.[^/]+\.(so|pyd)$", name)]
        if len(modules) != 2:
            raise ValueError(f"Expected CPU and GPU PGAS modules in {wheel.name}")
        for name in modules:
            binary = Path(temporary) / Path(name).name
            binary.write_bytes(package.read(name))
            counts = []
            for kind, suffix in [("elf", "cubin"), ("ptx", "ptx")]:
                result = subprocess.run([str(cuobjdump), f"--list-{kind}", str(binary)],
                                        text=True, capture_output=True)
                cpu_only = "/pgas_bound_cpu." in name
                diagnostic = result.stdout + result.stderr
                if result.returncode and not (cpu_only and "does not contain device code" in diagnostic):
                    raise RuntimeError(diagnostic)
                output = result.stdout
                counts.append(Counter(re.findall(rf"\.sm_(\d+)\.{suffix}\b", output)))
            native, ptx = counts
            if "/pgas_bound_cpu." in name:
                if native or ptx:
                    raise ValueError(f"CPU extension unexpectedly contains CUDA code: {name}")
                print(f"{wheel.name}: {Path(name).name}: independent CPU extension")
                continue
            if set(native) != expected or len(set(native.values())) != 1:
                raise ValueError(f"Incomplete native CUDA images in {name}: {dict(native)}; expected {sorted(expected)}")
            if set(ptx) != {baseline} or ptx[baseline] != native[baseline]:
                raise ValueError(f"Incomplete compute_{baseline} PTX in {name}: {dict(ptx)}")
            print(f"{wheel.name}: {Path(name).name}: native {sorted(native, key=int)}, PTX {baseline}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cuda-arch", choices=["PORTABLE", *ARCHITECTURES], default="PORTABLE")
    args = parser.parse_args()
    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if not wheels:
        parser.error("No wheels found")
    for wheel in wheels:
        check_wheel(wheel, args.cuda_root.resolve(), args.cuda_arch)


if __name__ == "__main__":
    main()
