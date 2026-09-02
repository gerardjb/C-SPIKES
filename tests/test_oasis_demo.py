from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_demo_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_compare_methods.py"
    spec = importlib.util.spec_from_file_location("c_spikes_oasis_demo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_keeps_oasis_opt_in_and_automatic_ar1_defaults():
    demo = _load_demo_module()

    args = demo.parse_args(["--dataset", "recording.mat"])

    assert args.run_oasis is False
    assert args.oasis_ar_order == 1
    assert demo._resolve_oasis_g(args.oasis_ar_order, args.oasis_g) == (None,)
    assert args.oasis_penalty == 1
    assert args.oasis_decimate == 1


def test_demo_parses_fixed_ar2_configuration():
    demo = _load_demo_module()

    args = demo.parse_args(
        [
            "--dataset",
            "recording.mat",
            "--run-oasis",
            "--oasis-ar-order",
            "2",
            "--oasis-g",
            "1.7",
            "-0.712",
            "--oasis-sn",
            "0.05",
            "--oasis-baseline",
            "-0.1",
            "--oasis-allow-negative-baseline",
            "--oasis-optimize-g",
            "2",
            "--oasis-penalty",
            "0",
            "--oasis-decimate",
            "2",
        ]
    )

    assert args.run_oasis is True
    assert demo._resolve_oasis_g(args.oasis_ar_order, args.oasis_g) == pytest.approx(
        (1.7, -0.712)
    )
    assert args.oasis_sn == pytest.approx(0.05)
    assert args.oasis_baseline == pytest.approx(-0.1)
    assert args.oasis_allow_negative_baseline is True
    demo._validate_oasis_args(args)


def test_demo_rejects_coefficient_count_mismatch():
    demo = _load_demo_module()

    with pytest.raises(ValueError, match="exactly 2"):
        demo._resolve_oasis_g(2, [0.95])


def test_demo_rejects_ar2_solver_options_for_ar1():
    demo = _load_demo_module()
    args = demo.parse_args(
        ["--dataset", "recording.mat", "--run-oasis", "--oasis-shift", "5"]
    )

    with pytest.raises(ValueError, match="require --oasis-ar-order 2"):
        demo._validate_oasis_args(args)
