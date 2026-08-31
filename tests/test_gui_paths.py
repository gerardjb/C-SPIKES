import os
import runpy
import sys
import types
from pathlib import Path

from c_spikes.gui.paths import (
    PROJECT_ROOT_ENV,
    configure_project_root,
    resolve_project_root,
)


def _make_checkout(root: Path) -> Path:
    (root / "src" / "c_spikes").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    return root


def test_configured_project_root_takes_priority(tmp_path):
    configured = tmp_path / "configured"
    module_file = tmp_path / "checkout" / "src" / "c_spikes" / "gui" / "app.py"

    resolved = resolve_project_root(
        module_file=module_file,
        environ={PROJECT_ROOT_ENV: str(configured)},
        cwd=tmp_path / "working",
    )

    assert resolved == configured.resolve()


def test_source_checkout_is_discovered_without_fixed_parent_depth(tmp_path):
    checkout = _make_checkout(tmp_path / "nested" / "checkout")
    module_file = checkout / "src" / "c_spikes" / "gui" / "app.py"

    resolved = resolve_project_root(
        module_file=module_file,
        environ={},
        cwd=tmp_path / "working",
    )

    assert resolved == checkout.resolve()


def test_installed_module_falls_back_to_launch_directory(tmp_path):
    checkout = _make_checkout(tmp_path / "checkout")
    installed_module = (
        tmp_path / "venv" / "lib" / "python3.10" / "site-packages" / "c_spikes" / "gui" / "app.py"
    )

    resolved = resolve_project_root(
        module_file=installed_module,
        environ={},
        cwd=checkout,
    )

    assert resolved == checkout.resolve()


def test_launcher_configuration_preserves_explicit_override(tmp_path):
    explicit = tmp_path / "explicit"
    checkout = tmp_path / "checkout"
    environ = {PROJECT_ROOT_ENV: str(explicit)}

    resolved = configure_project_root(checkout, environ=environ)

    assert resolved == explicit.resolve()
    assert environ[PROJECT_ROOT_ENV] == str(explicit)


def test_launcher_configuration_sets_checkout_root(tmp_path):
    checkout = tmp_path / "checkout"
    environ = {}

    resolved = configure_project_root(checkout, environ=environ)

    assert resolved == checkout.resolve()
    assert environ[PROJECT_ROOT_ENV] == str(checkout.resolve())


def test_documented_launcher_publishes_checkout_root(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    fake_tensorflow_env = types.ModuleType("c_spikes.tensorflow_env")
    fake_tensorflow_env.preload_tensorflow_quietly = lambda: None
    fake_app = types.ModuleType("c_spikes.gui.app")
    fake_app.main = lambda: None
    monkeypatch.setitem(sys.modules, "c_spikes.tensorflow_env", fake_tensorflow_env)
    monkeypatch.setitem(sys.modules, "c_spikes.gui.app", fake_app)
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)

    runpy.run_path(
        str(repo_root / "scripts" / "c_spikes_gui.py"),
        run_name="c_spikes_gui_launcher_test",
    )

    assert Path(os.environ[PROJECT_ROOT_ENV]) == repo_root.resolve()
