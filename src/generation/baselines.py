"""
Stage 1 dumb generative baselines.

These establish the floor the Stage 2 autoregressive transformer must beat
(project methodology: baseline first). Both operate over the 16 panel-states
(ChartTokenizer ids 0..15); specials are not used.

- NGramChartModel: difficulty-conditioned bigram P(state_t | state_{t-1}, difficulty).
  Audio-blind — tests how far rhythmic/structural priors alone get you.
- PerFrameMLP: P(state_t | audio_t, difficulty). Conditions on the aligned audio
  frame but ignores history — tests how much a single audio frame predicts steps.

See docs/phase2_generative_design.md.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from .tokenizer import ChartTokenizer, NUM_PANEL_STATES
from .foot_model import (
    FootState,
    NUM_PANELS,
    JACK_WEIGHT,
    TRAVEL_WEIGHT,
    FATIGUE_TAU,
    FOOTSWITCH_PEN,
    HARD_CAP,
)

# A dedicated "start" context for the first frame's bigram lookup.
START_CONTEXT = NUM_PANEL_STATES  # 16 (one past the real states)

# Candidate placements for the foot-physics baseline: the 4 single steps and the
# 6 two-panel jumps. (Holds/rolls are out of this baseline's scope, matching the
# other Phase-1 baselines.)
_SINGLES = [(p,) for p in range(NUM_PANELS)]
_JUMPS = [(a, b) for a in range(NUM_PANELS) for b in range(a + 1, NUM_PANELS)]


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class NGramChartModel:
    """Difficulty-conditioned bigram over panel-states with add-alpha smoothing."""

    def __init__(self, num_difficulties: int = 4, alpha: float = 1.0):
        self.num_difficulties = num_difficulties
        self.alpha = alpha
        # counts[d]: (17 contexts: 16 states + START) x (16 next states)
        self._counts = np.zeros((num_difficulties, NUM_PANEL_STATES + 1, NUM_PANEL_STATES), dtype=np.float64)
        self._probs: Optional[np.ndarray] = None

    def fit(self, charts: Sequence[Union[np.ndarray, torch.Tensor]], difficulties: Sequence[int]) -> "NGramChartModel":
        for chart, d in zip(charts, difficulties):
            tokens = ChartTokenizer.encode(chart).numpy()  # (T,) in 0..15
            prev = START_CONTEXT
            for tok in tokens:
                self._counts[d, prev, tok] += 1.0
                prev = tok
        smoothed = self._counts + self.alpha
        self._probs = smoothed / smoothed.sum(axis=2, keepdims=True)
        return self

    def _check_fit(self):
        if self._probs is None:
            raise RuntimeError("NGramChartModel must be fit() before use")

    def sample(self, length: int, difficulty: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Free-run a (length, 4) chart for the given difficulty."""
        self._check_fit()
        rng = rng or np.random.default_rng()
        probs_d = self._probs[difficulty]
        tokens = np.empty(length, dtype=np.int64)
        prev = START_CONTEXT
        for t in range(length):
            tok = rng.choice(NUM_PANEL_STATES, p=probs_d[prev])
            tokens[t] = tok
            prev = tok
        return ChartTokenizer.decode(tokens)

    def mean_nll(self, charts: Sequence[Union[np.ndarray, torch.Tensor]], difficulties: Sequence[int]) -> float:
        """Teacher-forced mean per-frame negative log-likelihood on held-out charts."""
        self._check_fit()
        total_nll, total_frames = 0.0, 0
        for chart, d in zip(charts, difficulties):
            tokens = ChartTokenizer.encode(chart).numpy()
            prev = START_CONTEXT
            for tok in tokens:
                total_nll += -np.log(self._probs[d, prev, tok])
                prev = tok
            total_frames += len(tokens)
        return total_nll / max(1, total_frames)


class PerFrameMLP(nn.Module):
    """Per-frame audio (+ difficulty) -> panel-state logits. No temporal context."""

    def __init__(self, audio_dim: int, num_difficulties: int = 4,
                 diff_emb_dim: int = 8, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.diff_embedding = nn.Embedding(num_difficulties, diff_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(audio_dim + diff_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_PANEL_STATES),
        )

    def forward(self, audio: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        """audio: (B, T, audio_dim); difficulty: (B,) long. Returns (B, T, 16) logits."""
        B, T, _ = audio.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)  # (B,T,diff_emb)
        return self.net(torch.cat([audio, diff], dim=-1))

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        """Argmax decode -> (B, T, 4) binary charts."""
        self.eval()
        logits = self.forward(audio, difficulty)  # (B,T,16)
        tokens = logits.argmax(dim=-1)  # (B,T)
        # decode each row's 16-state token to 4 panel bits
        bits = ((tokens.unsqueeze(-1) >> torch.arange(4, device=tokens.device)) & 1).float()
        return bits


class FootPhysicsBaseline:
    """A learned-model-FREE generator that chooses panels purely by foot physics.

    This is the standalone counterpart of the decode-time fatigue governor: it
    reuses the SAME `foot_model.FootState` cost (pad geometry, jack/travel/
    footswitch exertion) but as a *generator* rather than a logit penalty. It
    takes onsets as GIVEN (the same "which frames get a note" the model's onset
    head decides) and only picks WHICH arrows — so a comparison against the
    learned pattern head isolates one variable: panel choice, not density
    (experiment-design Rule 11). It is the foot-physics analogue of ArrowVortex's
    auto-stream, and the difficulty CRITIC/oracle anticipated in
    notes/foot_fatigue_design.md.

    At each onset frame it scores every candidate placement P by the more-fatigued
    foot of its easiest footing (`FootState.eval_pattern`), hard-forbids the
    unplayable (`E_eff >= fatigue_cap`) and over-long jacks (`max_jack_run`), then
    picks `P ~ softmax(-beta * E_eff + jump_bias·[P is a jump])` — i.e. it prefers
    low-fatigue footings, with `beta` as an inverse-temperature for variety. If
    every candidate is forbidden it falls back to the globally least-fatiguing one
    (always emit a note where the onset asked for one).

    Difficulty is a COARSE knob here (not the radar's calibrated conditioning): it
    nudges the jump rate and loosens the fatigue ceiling. The headline use is
    pattern-realism comparison (same-panel / jump-stream run-length vs REAL,
    stratified by difficulty), NOT difficulty conditioning — don't overclaim it
    (experiment-design Rule 15).
    """

    def __init__(
        self,
        beta: float = 1.0,
        allow_jumps: bool = True,
        jump_bias: float = -2.0,
        max_jack_run: Optional[int] = 2,
        fatigue_cap: float = 30.0,
        jack_weight: float = JACK_WEIGHT,
        travel_weight: float = TRAVEL_WEIGHT,
        fatigue_tau: float = FATIGUE_TAU,
        footswitch_pen: float = FOOTSWITCH_PEN,
        num_difficulties: int = 4,
    ):
        self.beta = beta
        self.allow_jumps = allow_jumps
        self.jump_bias = jump_bias
        self.max_jack_run = max_jack_run
        self.fatigue_cap = fatigue_cap
        self.jack_weight = jack_weight
        self.travel_weight = travel_weight
        self.fatigue_tau = fatigue_tau
        self.footswitch_pen = footswitch_pen
        self.num_difficulties = num_difficulties

    def _candidates(self, difficulty: int):
        """Candidate placements and their per-placement additive bias."""
        cands = list(_SINGLES)
        biases = [0.0] * len(cands)
        if self.allow_jumps:
            # Higher difficulty -> jumps become relatively more likely (coarse).
            dscale = (difficulty / max(self.num_difficulties - 1, 1)) if self.num_difficulties > 1 else 0.0
            jbias = self.jump_bias + 1.5 * dscale
            cands += list(_JUMPS)
            biases += [jbias] * len(_JUMPS)
        return cands, np.asarray(biases, dtype=np.float64)

    def generate(
        self,
        onsets: Union[np.ndarray, torch.Tensor, Sequence[int]],
        difficulty: int,
        bpm: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate a (T, 4) binary chart placing a note at each onset frame.

        Args:
            onsets: (T,) truthy where a note should be placed (e.g. the onset mask
                of a reference or model-generated chart: `chart.any(axis=1)`).
            difficulty: difficulty class (coarse jump/ceiling knob).
            bpm: chart BPM, for the rate term (notes are at 16th-frame resolution).
            rng: optional numpy Generator for reproducible sampling.

        Returns:
            (T, 4) float32 binary chart.
        """
        rng = rng or np.random.default_rng()
        onset_arr = _to_numpy(onsets).reshape(-1)
        onset_idx = np.flatnonzero(onset_arr > 0.5)
        T = int(onset_arr.shape[0])
        chart = np.zeros((T, NUM_PANELS), dtype=np.float32)

        frame_hz = float(bpm) * 4.0 / 60.0
        cap = self.fatigue_cap * (1.0 + 0.15 * difficulty)  # higher difficulty tolerates more fatigue
        state = FootState(
            jack_weight=self.jack_weight,
            travel_weight=self.travel_weight,
            fatigue_tau=self.fatigue_tau,
            footswitch_pen=self.footswitch_pen,
        )
        cands, biases = self._candidates(int(difficulty))

        for frame in onset_idx:
            frame = int(frame)
            E_eff = np.empty(len(cands), dtype=np.float64)
            footings = []
            forbidden = np.zeros(len(cands), dtype=bool)
            for i, panels in enumerate(cands):
                e, footing = state.eval_pattern(panels, frame, frame_hz)
                E_eff[i] = e
                footings.append(footing)
                if e >= cap:
                    forbidden[i] = True
                # Hard jack cap: a fresh single extending a same-panel run past the cap.
                if (
                    self.max_jack_run is not None
                    and len(panels) == 1
                    and panels[0] == state.sp_panel
                    and state.sp_run + 1 > self.max_jack_run
                ):
                    forbidden[i] = True

            logits = -self.beta * E_eff + biases
            logits[forbidden] = -np.inf
            if not np.isfinite(logits).any():       # everything forbidden -> least-bad footing
                choice = int(np.argmin(E_eff))
            else:
                logits -= np.nanmax(logits[np.isfinite(logits)])
                probs = np.exp(logits)
                probs[~np.isfinite(probs)] = 0.0
                probs /= probs.sum()
                choice = int(rng.choice(len(cands), p=probs))

            panels = cands[choice]
            state.commit(panels, footings[choice], frame)
            for p in panels:
                chart[frame, p] = 1.0

        return chart


def compute_state_class_weights(charts: Sequence[Union[np.ndarray, torch.Tensor]],
                                scheme: str = "inv_sqrt") -> torch.Tensor:
    """Inverse-frequency weights over the 16 panel-states for CE (empty state dominates)."""
    counts = np.ones(NUM_PANEL_STATES, dtype=np.float64)  # +1 smoothing
    for chart in charts:
        tokens = ChartTokenizer.encode(chart).numpy()
        binc = np.bincount(tokens, minlength=NUM_PANEL_STATES)
        counts += binc
    freq = counts / counts.sum()
    if scheme == "inv":
        w = 1.0 / freq
    elif scheme == "inv_sqrt":
        w = 1.0 / np.sqrt(freq)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    w = w / w.mean()  # normalize around 1
    return torch.tensor(w, dtype=torch.float32)
