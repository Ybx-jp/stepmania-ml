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

# A dedicated "start" context for the first frame's bigram lookup.
START_CONTEXT = NUM_PANEL_STATES  # 16 (one past the real states)


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
