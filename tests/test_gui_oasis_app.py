from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest


def _load_parser_helpers() -> dict[str, object]:
    """Load the pure parsers without importing Qt, matplotlib, or TensorFlow."""

    app_path = Path(__file__).parents[1] / "src" / "c_spikes" / "gui" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    helper_names = {
        "_parse_oasis_g",
        "_parse_oasis_optional_float",
        "_parse_oasis_discrete_settings",
        "_validate_oasis_g_stability",
    }
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names
    ]
    assert {node.name for node in helper_nodes} == helper_names

    module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "np": np,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
    }
    exec(compile(module, str(app_path), "exec"), namespace)
    return namespace


_HELPERS = _load_parser_helpers()
_parse_oasis_g = _HELPERS["_parse_oasis_g"]
_parse_oasis_optional_float = _HELPERS["_parse_oasis_optional_float"]
_parse_oasis_discrete_settings = _HELPERS["_parse_oasis_discrete_settings"]


def test_parse_oasis_g_accepts_auto_for_each_ar_order() -> None:
    assert _parse_oasis_g("auto", 1) == (None,)
    assert _parse_oasis_g(" AUTO ", 2) == (None, None)


def test_parse_oasis_g_accepts_stable_finite_coefficients() -> None:
    assert _parse_oasis_g("0.95", 1) == (0.95,)
    assert _parse_oasis_g(" 1.7, -0.72 ", 2) == (1.7, -0.72)


@pytest.mark.parametrize(
    ("text", "ar_order", "message"),
    [
        ("0.9,0.8", 1, "exactly 1"),
        ("0.9", 2, "exactly 2"),
        ("nan", 1, "finite"),
        ("1.0", 1, "0 < g < 1"),
        ("0.5,0.5", 2, "two real roots"),
    ],
)
def test_parse_oasis_g_rejects_wrong_count_nonfinite_and_unstable_values(
    text: str,
    ar_order: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_oasis_g(text, ar_order)


def test_parse_oasis_optional_float_accepts_auto_and_finite_values() -> None:
    assert _parse_oasis_optional_float("auto", name="sn", nonnegative=True) is None
    assert _parse_oasis_optional_float("0.25", name="sn", nonnegative=True) == 0.25
    assert _parse_oasis_optional_float("-1.5", name="baseline") == -1.5


@pytest.mark.parametrize("text", ["", "nan", "inf", "-0.1"])
def test_parse_oasis_noise_rejects_invalid_or_negative_values(text: str) -> None:
    with pytest.raises(ValueError, match="OASIS sn"):
        _parse_oasis_optional_float(text, name="sn", nonnegative=True)


def test_parse_oasis_discrete_settings_canonicalizes_disabled_output() -> None:
    assert _parse_oasis_discrete_settings(
        "none", "ignored", "noise_scaled"
    ) == ("none", None, "absolute")


@pytest.mark.parametrize("units", ["absolute", "noise_scaled"])
def test_parse_oasis_discrete_settings_accepts_positive_support_threshold(
    units: str,
) -> None:
    assert _parse_oasis_discrete_settings("support", " 1.25 ", units) == (
        "support",
        1.25,
        units,
    )


@pytest.mark.parametrize("threshold", ["", "auto", "0", "-0.1", "nan", "inf"])
def test_parse_oasis_discrete_settings_rejects_invalid_support_threshold(
    threshold: str,
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _parse_oasis_discrete_settings("support", threshold, "absolute")


@pytest.mark.parametrize(
    ("mode", "units", "message"),
    [
        ("threshold", "absolute", "discrete mode"),
        ("support", "sn", "threshold units"),
    ],
)
def test_parse_oasis_discrete_settings_rejects_unknown_choices(
    mode: str,
    units: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_oasis_discrete_settings(mode, "1", units)
