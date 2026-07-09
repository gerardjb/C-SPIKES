#!/usr/bin/env python3
"""Thin wrapper for PSD peak detection and notch-filter preprocessing."""

from __future__ import annotations

from c_spikes.inference.noise_calibration import run_psd_notch_preprocess_mat


if __name__ == "__main__":
    raise SystemExit(run_psd_notch_preprocess_mat())
