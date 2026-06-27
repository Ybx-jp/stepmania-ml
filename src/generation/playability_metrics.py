"""Canonical playability / footwork metrics for a generated or real chart.

This is the ONE importable home for the chart-level footwork measurements that
`experiments/generation_typed/*` currently each re-implement inline (the `metrics`
function copied across calib_foot_fatigue / diag_foot_fatigue / diag_stamina /
diag_jack_exertion / ...). New code (e.g. the foot-physics comparison harness)
routes through here instead of adding a 7th copy (experiment-design Rule 14).

`chart_metrics` is a faithful transcription of that established `metrics`; the
parity test `tests/test_playability_metrics.py::test_chart_metrics_matches_calib`
pins it to `calib_foot_fatigue.metrics` so the shared version cannot drift from
the numbers the fatigue governor was calibrated against.

A chart here is a (T, 4) array of per-panel SYMBOLS: 0 empty, 1 tap, 2 hold-head,
3 hold-tail, 4 roll. A panel is "active" (a foot strikes it) for symbols {1,2,4} —
the tail (3) is a release, not a press. Binary {0,1} charts (e.g. the foot-physics
baseline) are a special case (1 = active).
"""

from typing import Dict, List, Sequence, Union

import numpy as np

ACTIVE_SYMBOLS = (1, 2, 4)   # tap / hold-head / roll = a foot press (tail=3 is a release)
_GAP = 4                     # frames: presses within a 16th..quarter gap count as one run


def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _active_onsets(chart: np.ndarray):
    """List of (frame, [active panels]) for frames with >=1 press."""
    out = []
    for t in range(chart.shape[0]):
        a = [k for k in range(chart.shape[1]) if chart[t, k] in ACTIVE_SYMBOLS]
        if a:
            out.append((t, a))
    return out


def same_panel_run_lengths(chart: Union[np.ndarray, "object"]) -> List[int]:
    """Lengths of consecutive SAME single-panel presses (gap <= 4 frames).

    A jack/footswitch run is a maximal stretch of single-panel presses that all
    land on the same panel within a quarter-note gap. Jumps (>=2 panels) break a
    run and are not counted. This is the distribution the foot-physics policy most
    directly shapes (cf. the len2/len3/len>=4 reference in foot_exertion_findings).
    """
    chart = _to_numpy(chart)
    onsets = _active_onsets(chart)
    sg = [(t, a[0]) for t, a in onsets if len(a) == 1]
    runs: List[int] = []
    i = 0
    while i < len(sg):
        j = i
        while j + 1 < len(sg) and sg[j + 1][1] == sg[i][1] and sg[j + 1][0] - sg[j][0] <= _GAP:
            j += 1
        runs.append(j - i + 1)
        i = j + 1
    return runs


def run_length_shares(runs: Sequence[int]) -> Dict[str, float]:
    """Shares of run lengths among runs of length >= 2 (matches the human-reference
    framing len2 / len3 / len>=4), plus the max run length."""
    runs = list(runs)
    multi = [r for r in runs if r >= 2]
    n = max(len(multi), 1)
    return {
        "len2_share": sum(r == 2 for r in multi) / n,
        "len3_share": sum(r == 3 for r in multi) / n,
        "ge4_share": sum(r >= 4 for r in multi) / n,
        "n_runs_ge2": len(multi),
        "max_run": max(runs, default=0),
    }


def chart_metrics(chart: Union[np.ndarray, "object"]) -> Dict[str, float]:
    """Footwork summary for one (T, 4) typed chart.

    Returns jump_rate (share of pressed frames that are jumps), max_jump_stream
    (longest run of jumps within gap 4), max_jack_run (longest same-panel single
    run), jack_ge4_share (share of single-panel runs of length >= 4), and density
    (share of frames with any press). Transcription of the established `metrics`.
    """
    chart = _to_numpy(chart)
    onsets = _active_onsets(chart)
    n = max(len(onsets), 1)

    jumps = [(t, a) for t, a in onsets if len(a) >= 2]
    js = sorted(t for t, _ in jumps)
    mjs = run = 0
    for i in range(len(js)):
        run = run + 1 if i and js[i] - js[i - 1] <= _GAP else 1
        mjs = max(mjs, run)

    runs = same_panel_run_lengths(chart)
    mjk = max(runs, default=0)
    ge4 = sum(1 for r in runs if r >= 4) / max(len(runs), 1)
    density = float((chart != 0).any(axis=1).mean()) if chart.shape[0] else 0.0

    return {
        "jump_rate": len(jumps) / n,
        "max_jump_stream": float(mjs),
        "max_jack_run": float(mjk),
        "jack_ge4_share": ge4,
        "density": density,
    }
