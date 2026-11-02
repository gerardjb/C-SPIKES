"""Offline release admission controls. No native imports or network access."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("validate_release", Path(__file__).resolve().parents[2] / "scripts/validate_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseControls(unittest.TestCase):
    def run_record(self):
        return {"repository": {"full_name": release.REPOSITORY}, "head_repository": {"full_name": release.REPOSITORY},
                "path": ".github/workflows/build.yml", "event": "push", "head_branch": "v0.1.2",
                "head_sha": "a" * 40, "status": "completed", "conclusion": "success"}

    def test_only_successful_matching_tag_run(self):
        release.admit_run(self.run_record(), "v0.1.2", "a" * 40)
        for key, value in [("event", "pull_request"), ("head_branch", "main"), ("head_sha", "b" * 40),
                           ("conclusion", "failure"), ("status", "in_progress"), ("path", "other.yml")]:
            run = self.run_record()
            run[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.admit_run(run, "v0.1.2", "a" * 40)
        run = self.run_record()
        run["head_repository"]["full_name"] = "someone/fork"
        with self.assertRaises(ValueError):
            release.admit_run(run, "v0.1.2", "a" * 40)

    def test_release_tags(self):
        for tag in ("v0.1.2", "v1.2.3rc1"):
            self.assertEqual(release.version_for_tag(tag), tag[1:])
        for tag in ("main", "v0.1.2.dev1", "v0.1.2+cuda", "v01.2.3", "v1.2.3;echo bad", "v1.2.3/other"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.version_for_tag(tag)

    def test_exact_backend_artifacts(self):
        items = [{"name": name, "expired": False} for name in
                 ("wheels-Linux-cpu", "wheels-Windows-cpu", "wheels-Linux-cuda", "wheels-Windows-cuda", "sdist")]
        self.assertEqual([a["name"] for a in release.select_artifacts(items, "cpu")],
                         ["wheels-Linux-cpu", "wheels-Windows-cpu", "sdist"])
        for broken in (items[:-1], items + [items[0]], [{**a, "expired": True} for a in items]):
            with self.assertRaises(ValueError):
                release.select_artifacts(broken, "cpu")

    def bundle(self, root, backend="cpu", version="0.1.2", name="c-spikes"):
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
        for python in release.PYTHONS:
            for platform, suffix in [("win_amd64", "pyd"), ("manylinux_2_28_x86_64", "so")]:
                path = root / f"c_spikes-{version}-{python}-{python}-{platform}.whl"
                with zipfile.ZipFile(path, "w") as wheel:
                    wheel.writestr(f"c_spikes-{version}.dist-info/METADATA", metadata)
                    wheel.writestr(f"c_spikes/pgas/pgas_bound_cpu.{suffix}", b"fixture, not executable")
                    if backend == "cuda":
                        wheel.writestr(f"c_spikes/pgas/pgas_bound_gpu.{suffix}", b"fixture, not executable")
        with tarfile.open(root / f"c_spikes-{version}.tar.gz", "w:gz") as archive:
            member = tarfile.TarInfo(f"c_spikes-{version}/PKG-INFO")
            member.size = len(metadata)
            archive.addfile(member, io.BytesIO(metadata))

    def test_complete_bundles_and_wrong_backend(self):
        for backend in ("cpu", "cuda"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.bundle(root, backend)
                self.assertEqual(len(release.validate_bundle(root, "0.1.2", backend)), 9)
                with self.assertRaises(ValueError):
                    release.validate_bundle(root, "0.1.2", "cuda" if backend == "cpu" else "cpu")

    def test_missing_extra_version_and_name(self):
        for issue in ("missing", "extra", "version", "name"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.bundle(root, name="cspikes" if issue == "name" else "c-spikes")
                if issue == "missing":
                    next(root.glob("*.whl")).unlink()
                if issue == "extra":
                    (root / "surprise.txt").write_text("not a distribution")
                with self.subTest(issue=issue), self.assertRaises(ValueError):
                    release.validate_bundle(root, "0.1.3" if issue == "version" else "0.1.2", "cpu")


if __name__ == "__main__":
    unittest.main()
