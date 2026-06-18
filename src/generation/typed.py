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
