"""critic-v3 — WindowedLocalCritic (2026-07-13, taste-critic arc).

Fixes the three captured limitations of the graded_v2 critic in ONE model:
  (iv) LENGTH/LOCALITY  — scores OVERLAPPING WINDOWS (multi-scale) and aggregates with
       a fixed SOFT-MIN (worst-region-dominates; non-gameable for the E4 optimization
       ladder). Length-agnostic (no 2304 truncation) + exposes per-window scores.
  (ii) AUDIO-INPUT      — full 42-dim highres_v2 (the graded_v2 critic sliced audio[:, :23]).
  (i)  CHART-INPUT      — TYPED per-panel symbols {0 none,1 tap,2 hold-head,3 tail,4 roll-head}
                          via a shared Embedding, so hold-type (defect #3) is visible
                          (the old critic binarized -> presence-blind).

Reuses the proven O(T) Conv1DBackbone. Scored PER-SONG (batch = the corruption ladder,
shared length T + one audio) so windowing is over a fixed T (no ragged padding).

The preference objective (E2) rides on top later; this model's train objective is the
graded corruption ladder + a localized-corruption locality term (see train_critic_v3.py).
"""
import torch
import torch.nn as nn
from src.models.components.backbone import Conv1DBackbone


def soft_min(m: torch.Tensor, beta: float, valid: torch.Tensor = None) -> torch.Tensor:
    """Smooth minimum over dim=1.  s = -1/beta * log( mean_i exp(-beta * m_i) ).
    beta->0 => mean (forgiving);  beta->inf => hard min (worst window dominates).
    valid: optional (B,K) bool mask; masked windows excluded."""
    neg = -beta * m
    if valid is not None:
        neg = neg.masked_fill(~valid.bool(), float('-inf'))
        n = valid.float().sum(dim=1).clamp(min=1.0)
    else:
        n = torch.full((m.shape[0],), m.shape[1], dtype=m.dtype, device=m.device)
    lse = torch.logsumexp(neg, dim=1)          # (B,)
    return -(lse - torch.log(n)) / beta        # (B,)


def _win_starts(T: int, W: int, S: int):
    """Overlapping window starts that ALWAYS cover the tail (start = T-W is appended)."""
    if T <= W:
        return [0]
    st = list(range(0, T - W + 1, max(1, S)))
    if st[-1] != T - W:
        st.append(T - W)                       # guarantee the end is a window (the tail fix)
    return st


class WindowedLocalCritic(nn.Module):
    def __init__(self, audio_dim=42, d=128, n_panels=4, n_symbols=5, sym_emb=16,
                 backbone_blocks=3, dropout=0.1,
                 scales=((2304, 1152), (1152, 576), (576, 288)), softmin_beta=1.0):
        super().__init__()
        self.scales = tuple(scales)
        self.beta = float(softmin_beta)
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, d), nn.ReLU())
        self.sym_emb = nn.Embedding(n_symbols, sym_emb)
        self.chart_proj = nn.Sequential(nn.Linear(n_panels * sym_emb, d), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.backbone = Conv1DBackbone(input_dim=d, hidden_dim=d,
                                       num_blocks=backbone_blocks, dropout=dropout)
        self.win_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(d, 1))

    def frame_features(self, audio, chart):
        # audio (B,T,42) float ; chart (B,T,4) long
        a = self.audio_proj(audio)                        # (B,T,d)
        c = self.sym_emb(chart.long())                    # (B,T,4,e)
        c = self.chart_proj(c.flatten(-2))                # (B,T,d)
        x = self.fuse(torch.cat([a, c], dim=-1))          # (B,T,d)
        return self.backbone(x)                           # (B,T,d)  (mask=None: per-song, all valid)

    def window_margins(self, feats):
        """feats (B,T,d) -> allm (B,K) window margins + layout list.
        layout: list of (start, end, W) per window, in the same order as columns of allm."""
        B, T, D = feats.shape
        cols, layout = [], []
        for W, S in self.scales:
            Wc = min(W, T)
            for s in _win_starts(T, Wc, S):
                wf = feats[:, s:s + Wc].mean(dim=1)       # (B,d)  masked-mean == mean (per-song)
                cols.append(self.win_head(wf).squeeze(-1))  # (B,)
                layout.append((s, s + Wc, Wc))
        allm = torch.stack(cols, dim=1)                   # (B,K)
        return allm, layout

    def forward(self, audio, chart, return_windows=False):
        feats = self.frame_features(audio, chart)
        allm, layout = self.window_margins(feats)
        song = soft_min(allm, self.beta)                  # (B,)  the selection score
        if return_windows:
            return song, allm, layout
        return song
