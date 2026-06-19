"""
Typed (multi-step-type) chart representation for the generator.

Extends the binary tap representation to the full Phase-2.5 step vocabulary
(mines excluded). Per-panel symbol:

    0 = none   1 = tap   2 = hold-head   3 = hold/roll-tail   4 = roll-head

A typed chart is a (T, 4) int array over these 5 symbols per panel; an onset is any
frame with a non-empty panel. Holds/rolls are head→tail with empty rows implied
between (matches .sm). This is the representation produced by
StepManiaParser.convert_to_tensor_typed and consumed by the typed generator.
"""

import numpy as np
import torch

SYMBOL_NAMES = ["none", "tap", "hold_head", "tail", "roll_head"]
NUM_SYMBOLS = len(SYMBOL_NAMES)  # 5, per panel
NUM_PANELS = 4

# Layered head: which-panels "pattern" (15 non-empty 4-bit combos) + per-active-panel
# "type" (4 non-none symbols). Decouples is-panel-active from what-type.
NUM_PATTERNS = (1 << NUM_PANELS) - 1   # 15
NUM_TYPES = NUM_SYMBOLS - 1            # 4: tap, hold_head, tail, roll_head
_BITW = (1 << np.arange(NUM_PANELS)).astype(np.int64)  # [1,2,4,8]


def panels_to_pattern(active_bits) -> np.ndarray:
    """(..., 4) bool/int of active panels -> pattern index 0..14 (state 1..15 minus 1)."""
    bits = (np.asarray(active_bits) > 0).astype(np.int64)
    return (bits @ _BITW) - 1


def pattern_to_panels(idx):
    """pattern index 0..14 -> (4,) int active-panel bits."""
    state = int(idx) + 1
    return np.array([(state >> i) & 1 for i in range(NUM_PANELS)], dtype=np.int64)


# (15,4) bit table: which panels are active for each pattern index 0..14
_PATTERN_PANELS = np.stack([pattern_to_panels(i) for i in range(NUM_PATTERNS)])  # (15,4)


def make_pattern_bias(jump=0.0, panel_prefs=None):
    """A (15,) additive logit bias for the pattern (which-panels) head — a decode-time
    'pattern preference' knob (no retraining).

    - jump > 0 boosts multi-panel (jump) patterns (e.g. +2.0 for many more jumps);
      jump < 0 suppresses them.
    - panel_prefs (4,) adds a per-panel bias (L,D,U,R) to every pattern containing that
      panel, e.g. [0,0,1,1] favors Up/Right.
    """
    bias = np.zeros(NUM_PATTERNS, dtype=np.float32)
    npan = _PATTERN_PANELS.sum(1)  # panels per pattern
    bias += np.where(npan >= 2, jump, 0.0)
    if panel_prefs is not None:
        bias += _PATTERN_PANELS @ np.asarray(panel_prefs, dtype=np.float32)
    return bias


def count_crossovers(chart):
    """Greedy foot heuristic: alternate feet on single notes (reset after a jump); a
    crossover is the left foot on R or the right foot on L. Returns (crossings, singles)."""
    arr = to_numpy(chart)
    active = arr != 0
    foot = 0  # next foot: 0=left, 1=right
    crossings = singles = 0
    for t in range(arr.shape[0]):
        panels = np.where(active[t])[0]
        if len(panels) == 1:
            p = int(panels[0]); singles += 1
            if (foot == 0 and p == 3) or (foot == 1 and p == 0):
                crossings += 1
            foot = 1 - foot
        elif len(panels) >= 2:
            foot = 0  # reset alternation after a jump
    return crossings, singles


def to_numpy(chart) -> np.ndarray:
    return chart.detach().cpu().numpy() if isinstance(chart, torch.Tensor) else np.asarray(chart)


def onset_mask(chart) -> np.ndarray:
    """(T,) bool — frame has a step if any panel symbol is non-zero."""
    return (to_numpy(chart) != 0).any(axis=1)


def symbol_histogram(chart) -> dict:
    """Count of each symbol across all panels/frames (diagnostic)."""
    arr = to_numpy(chart).astype(int).reshape(-1)
    counts = np.bincount(arr, minlength=NUM_SYMBOLS)
    return {SYMBOL_NAMES[i]: int(counts[i]) for i in range(NUM_SYMBOLS)}


def pair_holds(chart) -> np.ndarray:
    """Make holds structurally valid per panel so the chart is always playable:
    every hold/roll head (2/4) gets a later tail (3); orphan heads -> tap (1);
    orphan tails -> none (0). Returns a new (T, 4) array."""
    arr = to_numpy(chart).astype(np.int64).copy()
    T, P = arr.shape
    for p in range(P):
        open_head = -1
        for t in range(T):
            s = arr[t, p]
            if s in (2, 4):              # hold-head or roll-head
                if open_head >= 0:       # previous head never closed -> demote to tap
                    arr[open_head, p] = 1
                open_head = t
            elif s == 3:                 # tail
                if open_head >= 0:
                    open_head = -1       # valid close
                else:
                    arr[t, p] = 0        # orphan tail -> none
        if open_head >= 0:               # unclosed head at end -> tap
            arr[open_head, p] = 1
    return arr
