from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(scope="module")
def cli_run():
    """Import the parser without requiring optional TensorFlow in unit-test envs."""

    previous = os.environ.get("C_SPIKES_TF_QUIET_IMPORT")
    os.environ["C_SPIKES_TF_QUIET_IMPORT"] = "0"
    try:
        return importlib.import_module("c_spikes.cli.run")
    finally:
        if previous is None:
            os.environ.pop("C_SPIKES_TF_QUIET_IMPORT", None)
        else:
            os.environ["C_SPIKES_TF_QUIET_IMPORT"] = previous


def test_oasis_cli_defaults_keep_method_opt_in(cli_run):
    args = cli_run.parse_args([])

    assert args.method is None
    assert args.oasis_ar_order == 1
    assert args.oasis_g == (None,)
    assert args.oasis_sn is None
    assert args.oasis_baseline is None
    assert args.oasis_allow_negative_baseline is False
    assert args.oasis_optimize_g == 0
    assert args.oasis_penalty == 1
    assert args.oasis_decimate == 1
    assert args.oasis_max_iter is None
    assert args.oasis_shift is None
    assert args.oasis_window is None
    assert args.oasis_tol is None
    assert args.oasis_uniformity_rtol == pytest.approx(5e-3)
    assert args.oasis_uniformity_atol == pytest.approx(1e-9)


def test_oasis_cli_ar2_omitted_coefficients_request_estimation(cli_run):
    args = cli_run.parse_args(["--method", "oasis", "--oasis-ar-order", "2"])

    assert args.method == ["oasis"]
    assert args.oasis_g == (None, None)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--oasis-g", "0.94"], (0.94,)),
        (
            [
                "--oasis-ar-order",
                "2",
                "--oasis-g",
                "1.7",
                "-0.712",
                "--oasis-shift",
                "5",
                "--oasis-window",
                "20",
                "--oasis-tol",
                "1e-8",
            ],
            (1.7, -0.712),
        ),
    ],
)
def test_oasis_cli_parses_fixed_ar_coefficients(cli_run, argv, expected):
    assert cli_run.parse_args(argv).oasis_g == pytest.approx(expected)


@pytest.mark.parametrize(
    "argv",
    [
        ["--oasis-g", "0.9", "0.1"],
        ["--oasis-ar-order", "2", "--oasis-g", "0.9"],
        ["--oasis-ar-order", "3"],
        ["--oasis-g", "nan"],
        ["--oasis-sn", "-0.01"],
        ["--oasis-optimize-g", "-1"],
        ["--oasis-penalty", "2"],
        ["--oasis-decimate", "0"],
        ["--oasis-max-iter", "0"],
        ["--oasis-ar-order", "2", "--oasis-shift", "0"],
        ["--oasis-ar-order", "2", "--oasis-window", "-2"],
        ["--oasis-ar-order", "2", "--oasis-tol", "0"],
        ["--oasis-uniformity-rtol", "-1e-3"],
        ["--oasis-uniformity-atol", "inf"],
    ],
)
def test_oasis_cli_rejects_invalid_counts_and_ranges(cli_run, argv):
    with pytest.raises(SystemExit) as exc_info:
        cli_run.parse_args(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("option", ["--oasis-shift", "--oasis-window", "--oasis-tol"])
def test_oasis_cli_rejects_ar2_only_options_for_ar1(cli_run, option):
    with pytest.raises(SystemExit) as exc_info:
        cli_run.parse_args([option, "1"])

    assert exc_info.value.code == 2


def test_oasis_cli_leaves_coefficient_stability_to_adapter(cli_run):
    args = cli_run.parse_args(["--oasis-g", "1.1"])

    assert args.oasis_g == (1.1,)


def test_cli_help_lists_oasis_as_opt_in(cli_run, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_run.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "pgas, ens2, cascade, oasis" in help_text
    assert "OASIS is opt-in" in help_text


def test_main_plumbs_all_oasis_options_to_run_config(cli_run, monkeypatch):
    captured = {}
    sentinel = object()

    def fake_run_config(**kwargs):
        captured["kwargs"] = kwargs
        return sentinel

    def fake_run_batch(config):
        captured["config"] = config
        return []

    monkeypatch.setattr(cli_run, "RunConfig", fake_run_config)
    monkeypatch.setattr(cli_run, "run_batch", fake_run_batch)

    cli_run.main(
        [
            "--dataset",
            "synthetic",
            "--method",
            "oasis",
            "--oasis-ar-order",
            "2",
            "--oasis-g",
            "1.7",
            "-0.712",
            "--oasis-sn",
            "0.04",
            "--oasis-baseline",
            "-0.2",
            "--oasis-allow-negative-baseline",
            "--oasis-optimize-g",
            "3",
            "--oasis-penalty",
            "0",
            "--oasis-decimate",
            "2",
            "--oasis-max-iter",
            "8",
            "--oasis-shift",
            "5",
            "--oasis-window",
            "20",
            "--oasis-tol",
            "2e-8",
            "--oasis-uniformity-rtol",
            "2e-4",
            "--oasis-uniformity-atol",
            "3e-10",
        ]
    )

    assert captured["config"] is sentinel
    kwargs = captured["kwargs"]
    assert kwargs["methods"] == ["oasis"]
    assert kwargs["oasis_g"] == pytest.approx((1.7, -0.712))
    assert kwargs["oasis_sn"] == pytest.approx(0.04)
    assert kwargs["oasis_b"] == pytest.approx(-0.2)
    assert kwargs["oasis_b_nonneg"] is False
    assert kwargs["oasis_optimize_g"] == 3
    assert kwargs["oasis_penalty"] == 0
    assert kwargs["oasis_decimate"] == 2
    assert kwargs["oasis_max_iter"] == 8
    assert kwargs["oasis_shift"] == 5
    assert kwargs["oasis_window"] == 20
    assert kwargs["oasis_tol"] == pytest.approx(2e-8)
    assert kwargs["oasis_uniformity_rtol"] == pytest.approx(2e-4)
    assert kwargs["oasis_uniformity_atol"] == pytest.approx(3e-10)


def test_main_preserves_legacy_default_method_set(cli_run, monkeypatch):
    captured = {}

    def fake_run_config(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli_run, "RunConfig", fake_run_config)
    monkeypatch.setattr(cli_run, "run_batch", lambda config: [])

    cli_run.main([])

    assert captured["methods"] == ("pgas", "ens2", "cascade")
