"""Tests for the shared playability/footwork metrics.

The parity test pins `chart_metrics` to the established `metrics` function that the
fatigue governor was calibrated against (duplicated across the diag/calib scripts),
so the shared module the comparison harness uses stays numerically comparable to
the calibration history (experiment-design Rule 14).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.generation.playability_metrics import (
    chart_metrics,
    same_panel_run_lengths,
    run_length_shares,
)
from src.generation.baselines import FootPhysicsBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_calib_metrics():
    """Import the established `metrics` from the calib script by file path."""
    path = PROJECT_ROOT / "experiments" / "generation_typed" / "calib_foot_fatigue.py"
    spec = importlib.util.spec_from_file_location("calib_foot_fatigue", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def test_chart_metrics_matches_calib():
    calib_metrics = _load_calib_metrics()
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = int(rng.integers(40, 200))
        # random typed chart: mostly empty, some taps/holds/tails/rolls
        chart = rng.choice([0, 0, 0, 1, 1, 2, 3, 4], size=(T, 4)).astype(np.int64)
        ref = calib_metrics(chart)                       # (jump_rate, mjs, mjk, ge4)
        got = chart_metrics(chart)
        assert got["jump_rate"] == pytest.approx(ref[0], rel=1e-9, abs=1e-12)
        assert got["max_jump_stream"] == pytest.approx(ref[1])
        assert got["max_jack_run"] == pytest.approx(ref[2])
        assert got["jack_ge4_share"] == pytest.approx(ref[3], rel=1e-9, abs=1e-12)


def test_same_panel_run_lengths_basic():
    chart = np.zeros((10, 4), dtype=np.int64)
    chart[0, 0] = 1; chart[1, 0] = 1; chart[2, 0] = 1   # L,L,L  (run of 3 on panel 0)
    chart[3, 3] = 1                                       # R      (run of 1 on panel 3)
    chart[5, 1] = 1; chart[6, 1] = 1                      # gap of 2 from prev -> new run of 2 on panel 1
    runs = same_panel_run_lengths(chart)
    assert sorted(runs) == [1, 2, 3]


def test_run_length_shares():
    sh = run_length_shares([1, 2, 2, 3, 5])   # runs>=2: [2,2,3,5]
    assert sh["max_run"] == 5
    assert sh["n_runs_ge2"] == 4
    assert sh["len2_share"] == pytest.approx(0.5)
    assert sh["len3_share"] == pytest.approx(0.25)
    assert sh["ge4_share"] == pytest.approx(0.25)


def test_tail_symbol_not_counted_as_press():
    chart = np.zeros((4, 4), dtype=np.int64)
    chart[0, 0] = 2     # hold head (press)
    chart[1, 0] = 3     # tail (release, NOT a press)
    m = chart_metrics(chart)
    assert m["density"] == pytest.approx(0.5)            # both frames non-empty
    # only the head frame is an "onset" -> a single run of length 1, no jacks
    assert m["max_jack_run"] == 1.0


def test_chart_metrics_on_foot_physics_output():
    onsets = np.ones(300, dtype=bool)
    gen = FootPhysicsBaseline(max_jack_run=2).generate(onsets, 2, 180.0, rng=np.random.default_rng(1))
    m = chart_metrics(gen)
    assert m["density"] == pytest.approx(1.0)            # every onset placed
    assert m["max_jack_run"] <= 2                        # jack cap respected
