"""Tests for the shared foot-physics model and the FootPhysicsBaseline generator.

The key test is `test_foot_state_matches_governor_formula`: it pins `FootState`'s
per-note exertion cost to the EXACT formula in `typed_model.generate`'s fatigue
governor (lines ~684-733), so the standalone baseline and the decode-time
governor cannot silently drift apart (experiment-design Rule 14).
"""

import numpy as np
import pytest

from src.generation.foot_model import (
    FootState,
    PAD_DIST,
    PANEL_POS,
    JACK_WEIGHT,
    TRAVEL_WEIGHT,
    FATIGUE_TAU,
    FOOTSWITCH_PEN,
    HARD_CAP,
)
from src.generation.baselines import FootPhysicsBaseline


# ---- geometry -------------------------------------------------------------------

def test_pad_distance_matrix():
    assert PAD_DIST.shape == (4, 4)
    np.testing.assert_allclose(np.diag(PAD_DIST), 0.0)               # self-distance 0
    np.testing.assert_allclose(PAD_DIST[0, 3], 2.0)                  # L <-> R
    np.testing.assert_allclose(PAD_DIST[1, 2], 2.0)                  # D <-> U
    np.testing.assert_allclose(PAD_DIST[0, 1], np.sqrt(2.0), rtol=1e-6)  # L <-> D adjacent
    np.testing.assert_allclose(PAD_DIST, PAD_DIST.T)                 # symmetric


# ---- parity with the decode-time governor (the drift guard) ---------------------

def _governor_E_eff(panels, foot_panel, foot_E, foot_t, sp_run, frame, frame_hz):
    """Independent transcription of typed_model.generate's fatigue cost (per note),
    in plain numpy, for a single sequence. Mirrors lines ~684-733."""
    tau_frames = max(FATIGUE_TAU * 4.0, 1e-3)
    dt = np.maximum(frame - np.asarray(foot_t, float), 0.0)
    decayE = np.asarray(foot_E, float) * np.exp(-dt / tau_frames)
    rate = frame_hz / np.maximum(frame - np.asarray(foot_t, float), 1.0)
    fp = np.asarray(foot_panel, int)

    def cost(f, p):
        if fp[f] < 0:
            unit = 0.0
        elif fp[f] == p:
            unit = JACK_WEIGHT
        else:
            unit = TRAVEL_WEIGHT * float(PAD_DIST[fp[f], p])
        c = rate[f] * unit * (1.0 if fp[f] >= 0 else 0.0)
        other = fp[1 - f]
        stay = fp[f] == p
        if other == p and other >= 0 and not stay:           # footswitch surcharge
            runp = sp_run + 1
            c += HARD_CAP if runp >= 4 else (FOOTSWITCH_PEN if runp == 3 else 0.0)
        return c

    if len(panels) == 1:
        p = panels[0]
        oa = max(decayE[0] + cost(0, p), decayE[1])
        ob = max(decayE[1] + cost(1, p), decayE[0])
        return min(oa, ob)
    x, y = panels
    oa = max(decayE[0] + cost(0, x), decayE[1] + cost(1, y))
    ob = max(decayE[0] + cost(0, y), decayE[1] + cost(1, x))
    return min(oa, ob)


@pytest.mark.parametrize("scenario", [
    # (panels, foot_panel, foot_E, foot_t, sp_run, frame)
    ((0,), [-1, -1], [0.0, 0.0], [-10000, -10000], 0, 0),      # fresh single
    ((3,), [0, -1], [1.0, 0.0], [0, -10000], 0, 4),            # one foot placed, other free
    ((0,), [0, 3], [2.0, 1.5], [2, 1], 1, 4),                  # jack: left foot already on L
    ((0,), [3, 0], [1.0, 2.0], [3, 2], 2, 4),                  # footswitch onto L (other foot on L), runp=3
    ((0,), [3, 0], [1.0, 2.0], [3, 2], 3, 4),                  # footswitch hard cap, runp=4
    ((0, 3), [1, 2], [1.0, 1.0], [1, 1], 0, 6),               # jump LR from feet on D,U
    ((1, 2), [0, 3], [0.5, 0.5], [0, 0], 0, 8),               # jump DU from feet on L,R
])
def test_foot_state_matches_governor_formula(scenario):
    panels, foot_panel, foot_E, foot_t, sp_run, frame = scenario
    frame_hz = 180.0 * 4.0 / 60.0
    state = FootState()
    state.pos = list(foot_panel)
    state.E = list(foot_E)
    state.t = list(foot_t)
    state.sp_run = sp_run
    state.sp_panel = foot_panel[0] if sp_run > 0 else -1

    e_helper, _ = state.eval_pattern(panels, frame, frame_hz)
    e_ref = _governor_E_eff(panels, foot_panel, foot_E, foot_t, sp_run, frame, frame_hz)
    assert e_helper == pytest.approx(e_ref, rel=1e-6, abs=1e-9)


def test_eval_pattern_does_not_mutate_state():
    state = FootState()
    state.pos = [0, 3]
    before = (list(state.pos), list(state.E), list(state.t), state.sp_run)
    state.eval_pattern((1,), frame=4, frame_hz=12.0)
    after = (list(state.pos), list(state.E), list(state.t), state.sp_run)
    assert before == after  # eval is pure; only commit() writes state


# ---- the baseline generator -----------------------------------------------------

def _max_same_single_run(chart: np.ndarray) -> int:
    """Longest run of consecutive frames that are the SAME single panel."""
    best = run = 0
    prev = None
    for row in chart:
        active = np.flatnonzero(row > 0.5)
        if len(active) == 1:
            p = int(active[0])
            run = run + 1 if p == prev else 1
            prev = p
        else:
            run, prev = 0, None
        best = max(best, run)
    return best


def test_baseline_places_exactly_at_onsets():
    rng = np.random.default_rng(0)
    onsets = (rng.random(300) < 0.5)
    gen = FootPhysicsBaseline().generate(onsets, difficulty=2, bpm=160.0, rng=np.random.default_rng(1))
    assert gen.shape == (300, 4)
    # a note exactly where (and only where) an onset was requested
    np.testing.assert_array_equal(gen.any(axis=1), onsets)


def test_baseline_is_deterministic_with_seed():
    onsets = np.ones(200, dtype=bool)
    a = FootPhysicsBaseline().generate(onsets, 1, 175.0, rng=np.random.default_rng(42))
    b = FootPhysicsBaseline().generate(onsets, 1, 175.0, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_baseline_respects_max_jack_run():
    onsets = np.ones(400, dtype=bool)  # dense 16th stream stresses the jack cap
    gen = FootPhysicsBaseline(max_jack_run=2, allow_jumps=False).generate(
        onsets, difficulty=3, bpm=200.0, rng=np.random.default_rng(7))
    assert _max_same_single_run(gen) <= 2


def test_baseline_no_jumps_when_disabled():
    onsets = np.ones(150, dtype=bool)
    gen = FootPhysicsBaseline(allow_jumps=False).generate(
        onsets, 2, 150.0, rng=np.random.default_rng(3))
    assert int(gen.sum(axis=1).max()) <= 1  # never two panels at once
