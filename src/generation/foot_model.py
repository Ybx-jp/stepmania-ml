"""Shared foot-physics model — the single source of truth for pad geometry and the
two-foot exertion cost used by BOTH:

  * the decode-time per-foot FATIGUE governor in `typed_model.generate` (the
    torch-vectorized penalty on `pat_logits`), and
  * the standalone `FootPhysicsBaseline` generator (numpy, no learned model).

WHY this module exists (experiment-design Rule 14 — route standing logic through
ONE code-enforced helper, never re-implement it ad-hoc): the governor and the
baseline must agree on what "foot travel / jack / footswitch cost" means, or a
comparison between them measures the drift between two copies of the math instead
of the thing under test. The governor imports `PANEL_POS` from here; the parity
test `test_foot_model_matches_governor` pins `FootState`'s cost to the governor's
inline formula so the two cannot silently diverge.

The math mirrors `typed_model.generate` (see notes/foot_fatigue_design.md):
per note, assign the arrow(s) to the two feet at MIN added exertion (crossovers
allowed when cheaper, no surcharge); a foot that STAYS & re-hits costs
`jack_weight * rate`, one that MOVES costs `travel_weight * dist * rate`; per-foot
exertion DECAYS exponentially (time constant `fatigue_tau` beats). A pattern's
cost is the more-fatigued foot of its easiest footing: `min over footings of
max(E_left_after, E_right_after)`.
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

NUM_PANELS = 4

# Panel coordinates on the dance "cross", in the project's [Left, Down, Up, Right]
# chart-column order. L<->R and U<->D are distance 2; adjacent panels are sqrt(2).
PANEL_POS = np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)


def pad_distance_matrix() -> np.ndarray:
    """(4, 4) Euclidean foot-travel distance between every pair of panels."""
    d = PANEL_POS[:, None, :] - PANEL_POS[None, :, :]
    return np.sqrt((d ** 2).sum(-1)).astype(np.float32)


PAD_DIST = pad_distance_matrix()

# Cost-model defaults. These MUST match the signature defaults of
# `typed_model.generate` (jack_weight, travel_weight, fatigue_tau, footswitch_pen)
# — the parity test fails if they drift.
JACK_WEIGHT = 1.0       # stay & re-hit (no momentum to ride) — harder than a 1-panel travel
TRAVEL_WEIGHT = 0.6     # per unit pad-distance, per the foot's press rate
FATIGUE_TAU = 2.0       # exertion decay time constant, in beats (~half a measure)
FOOTSWITCH_PEN = 4.0    # added to a 3rd same-panel alternation (2 free / 3 pen / 4+ hard cap)
HARD_CAP = 1e4          # the governor's "effectively forbidden" sentinel


@dataclass
class Footing:
    """The chosen two-foot assignment for one note (what `commit` writes back)."""
    pos: List[int]      # [left_panel, right_panel] after the note (-1 = foot lifted/free)
    E: List[float]      # [left_E, right_E] exertion after the note (pre-decay-to-next)
    used: List[bool]    # which feet actually struck this note


@dataclass
class FootState:
    """Mutable two-foot exertion state, stepped one note at a time.

    Mirrors the per-frame state of the fatigue governor: `pos` (= body
    orientation), per-foot exertion `E` at each foot's last-hit frame, last-hit
    frame `t`, and the same-panel single-run counter used to grade footswitches.
    """

    jack_weight: float = JACK_WEIGHT
    travel_weight: float = TRAVEL_WEIGHT
    fatigue_tau: float = FATIGUE_TAU
    footswitch_pen: float = FOOTSWITCH_PEN

    pos: List[int] = field(default_factory=lambda: [-1, -1])
    E: List[float] = field(default_factory=lambda: [0.0, 0.0])
    t: List[int] = field(default_factory=lambda: [-10000, -10000])
    sp_run: int = 0          # length of the current same-panel single run
    sp_panel: int = -1       # panel of that run (-1 = none)

    @property
    def tau_frames(self) -> float:
        return max(self.fatigue_tau * 4.0, 1e-3)  # tau beats -> 16th frames

    # ---- cost primitives (the shared definition) --------------------------------

    def _decayed(self, frame: int) -> Tuple[List[float], List[float]]:
        """Per-foot (exertion decayed to `frame`, press rate to `frame`)."""
        decayE, rate = [0.0, 0.0], [0.0, 0.0]
        for f in (0, 1):
            dt = max(frame - self.t[f], 0)
            decayE[f] = self.E[f] * float(np.exp(-dt / self.tau_frames))
            rate[f] = 1.0 / max(frame - self.t[f], 1)  # scaled by frame_hz in _cost
        return decayE, rate

    def _unit(self, f: int, p: int) -> float:
        """Per-hit unit cost for foot `f` striking panel `p` (before rate scaling)."""
        if self.pos[f] < 0:
            return 0.0                       # a free foot placing for the first time is free
        if self.pos[f] == p:
            return self.jack_weight          # stay & re-hit
        return self.travel_weight * float(PAD_DIST[self.pos[f], p])  # move

    def _cost(self, f: int, p: int, rate: Sequence[float], frame_hz: float, runp: int) -> float:
        """Added exertion for foot `f` to strike `p`, incl. the footswitch surcharge."""
        c = (frame_hz * rate[f]) * self._unit(f, p) * (1.0 if self.pos[f] >= 0 else 0.0)
        other = self.pos[1 - f]
        if other == p and other >= 0 and self.pos[f] != p:      # footswitch: other foot holds p
            c += HARD_CAP if runp >= 4 else (self.footswitch_pen if runp == 3 else 0.0)
        return c

    def eval_pattern(self, panels: Sequence[int], frame: int, frame_hz: float) -> Tuple[float, Footing]:
        """Min-exertion footing for a candidate pattern (1 panel = step, 2 = jump).

        Returns ``(E_eff, footing)`` where ``E_eff`` is the more-fatigued foot of
        the easiest footing (`min over footings of max(E_L, E_R)`) and ``footing``
        is the assignment to pass to `commit` if this pattern is chosen.
        """
        decayE, rate = self._decayed(frame)
        runp = self.sp_run + 1
        if len(panels) == 1:
            p = int(panels[0])
            cL = self._cost(0, p, rate, frame_hz, runp)
            cR = self._cost(1, p, rate, frame_hz, runp)
            oa = max(decayE[0] + cL, decayE[1])     # left foot hits, right idles
            ob = max(decayE[1] + cR, decayE[0])     # right foot hits, left idles
            if oa <= ob:
                return oa, Footing([p, self.pos[1]], [decayE[0] + cL, decayE[1]], [True, False])
            return ob, Footing([self.pos[0], p], [decayE[0], decayE[1] + cR], [False, True])
        x, y = int(panels[0]), int(panels[1])
        cLx = self._cost(0, x, rate, frame_hz, runp); cRy = self._cost(1, y, rate, frame_hz, runp)
        cLy = self._cost(0, y, rate, frame_hz, runp); cRx = self._cost(1, x, rate, frame_hz, runp)
        oa = max(decayE[0] + cLx, decayE[1] + cRy)  # L=x, R=y
        ob = max(decayE[0] + cLy, decayE[1] + cRx)  # L=y, R=x
        if oa <= ob:
            return oa, Footing([x, y], [decayE[0] + cLx, decayE[1] + cRy], [True, True])
        return ob, Footing([y, x], [decayE[0] + cLy, decayE[1] + cRx], [True, True])

    # ---- state update -----------------------------------------------------------

    def commit(self, panels: Sequence[int], footing: Footing, frame: int) -> None:
        """Write a chosen footing back into the state (call once per placed note)."""
        for f in (0, 1):
            if footing.used[f]:
                self.pos[f] = footing.pos[f]
                self.E[f] = footing.E[f]
                self.t[f] = frame
        # Footswitch LIFT: if both feet now read one panel, the foot that did NOT
        # act lifts (-1) — else the state corrupts to "both feet here" and the
        # footswitch cap is bypassed via the stay path (governor bug #5).
        if self.pos[0] == self.pos[1] and self.pos[0] >= 0:
            for f in (0, 1):
                if not footing.used[f]:
                    self.pos[f] = -1
        # Same-panel single-run tracker (footswitch grading): +1 on a same-panel
        # single, reset to 1 on a new-panel single, 0 on a jump.
        if len(panels) == 1:
            p = int(panels[0])
            if p == self.sp_panel:
                self.sp_run += 1
            else:
                self.sp_run = 1
            self.sp_panel = p
        else:
            self.sp_run = 0
            self.sp_panel = -1
