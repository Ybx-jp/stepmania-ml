"""
Factorized onset-then-panel chart generator (Stage 3).

Splits the single 16-way per-frame softmax of ChartGenerator into two heads that
the density-calibration probe showed are the right decomposition:

- **Onset head (audio-driven, non-causal):** predicts P(step occurs) per frame
  from the fixed audio (+ position, difficulty) ONLY — it does not read the
  generated step tokens. Because audio is a fixed input, onset probability can't
  collapse under self-generated context (the autoregressive drift that killed
  threshold decoding on the single-head model). Density becomes an honest, stable
  threshold on this head.
- **Panel head (autoregressive):** predicts which of the 15 non-empty panel
  patterns to play, GIVEN an onset, conditioned on the step history via the
  causal decoder. This is the easy part once "how many / where" is decided.

Onset loss = binary cross-entropy (BCE: cross-entropy for a yes/no target) with a
pos_weight to offset the empty-frame majority. Panel loss = cross-entropy over the
15 non-empty patterns, computed only on real-onset frames.

See notes/density_calibration.md.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.components.encoders import AudioEncoder
from .tokenizer import VOCAB_SIZE, NUM_PANEL_STATES
from .transformer import PositionalEncoding, _causal_mask

NUM_NONEMPTY = NUM_PANEL_STATES - 1  # 15 panel patterns excluding the empty state


class FactorizedChartGenerator(nn.Module):
    def __init__(
        self,
        audio_dim: int = 23,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        onset_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_difficulties: int = 4,
        max_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Shared conditioning encoder (reused / warm-started from Phase 1).
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=d_model)
        self.diff_embedding = nn.Embedding(num_difficulties, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        # --- onset branch: non-causal encoder over audio, no token feedback ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.onset_encoder = nn.TransformerEncoder(enc_layer, num_layers=onset_layers)
        self.onset_head = nn.Linear(d_model, 1)

        # --- panel branch: autoregressive decoder over step tokens ---
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.panel_head = nn.Linear(d_model, NUM_NONEMPTY)

    # ---- warm start (same interface as ChartGenerator) --------------------------

    def load_audio_encoder(self, classifier_state_dict: dict) -> int:
        prefix = "audio_encoder."
        sub = {k[len(prefix):]: v for k, v in classifier_state_dict.items() if k.startswith(prefix)}
        missing, unexpected = self.audio_encoder.load_state_dict(sub, strict=False)
        if unexpected:
            raise ValueError(f"unexpected audio_encoder keys: {unexpected}")
        return len(sub) - len(missing)

    def freeze_audio_encoder(self, freeze: bool = True):
        for p in self.audio_encoder.parameters():
            p.requires_grad = not freeze

    # ---- branch forwards --------------------------------------------------------

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(audio)  # (B, T, d)

    def onset_logits(self, memory: torch.Tensor, difficulty: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Audio-driven onset score per frame. (B, T). No dependence on step tokens."""
        B, T, _ = memory.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)
        x = self.dropout(self.pos_encoding(memory) + diff)
        pad = (~mask.bool()) if mask is not None else None
        h = self.onset_encoder(x, src_key_padding_mask=pad)
        return self.onset_head(h).squeeze(-1)

    def panel_logits(self, memory: torch.Tensor, in_tokens: torch.Tensor,
                     difficulty: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Autoregressive panel-pattern scores. (B, T, 15) over non-empty states."""
        B, T = in_tokens.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)
        tgt = self.dropout(self.pos_encoding(self.token_embedding(in_tokens)) + diff)
        causal = _causal_mask(T, in_tokens.device)
        pad = (~mask.bool()) if mask is not None else None
        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal,
                         tgt_key_padding_mask=pad, memory_key_padding_mask=pad)
        return self.panel_head(h)

    def forward(self, audio: torch.Tensor, in_tokens: torch.Tensor,
                difficulty: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced. Returns (onset_logits (B,T), panel_logits (B,T,15))."""
        memory = self.encode_audio(audio)
        return (self.onset_logits(memory, difficulty, mask),
                self.panel_logits(memory, in_tokens, difficulty, mask))

    # ---- generation -------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        difficulty: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        onset_threshold: float = 0.5,
        onset_sample: bool = False,
        onset_logit_scale: float = 1.0,
        onset_logit_bias: float = 0.0,
        panel_greedy: bool = True,
        panel_temperature: float = 1.0,
        panel_top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode a (B, T, 4) chart.

        Onset decisions are computed in ONE non-autoregressive pass from audio
        (stable, no collapse); density is set by `onset_threshold` (or Bernoulli
        sampling — flip each frame's coin at its predicted probability — if
        `onset_sample`). Only the panel pattern is decoded autoregressively, and
        only where an onset fired.

        `onset_logit_scale`/`onset_logit_bias` apply post-hoc Platt calibration to
        the onset logits: p = sigmoid(scale * logit + bias). Defaults are a no-op.
        """
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        from .tokenizer import BOS_TOKEN

        memory = self.encode_audio(audio)
        p_onset = torch.sigmoid(onset_logit_scale * self.onset_logits(memory, difficulty) + onset_logit_bias)  # (B, T)
        if onset_sample:
            onset = torch.bernoulli(p_onset).bool()
        else:
            onset = p_onset > onset_threshold

        gen = torch.zeros(B, T, dtype=torch.long, device=device)
        cur = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)

        for t in range(T):
            diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, cur.size(1), -1)
            tgt = self.pos_encoding(self.token_embedding(cur)) + diff
            causal = _causal_mask(cur.size(1), device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            logits = self.panel_head(h[:, -1])  # (B, 15) over non-empty states

            if panel_greedy:
                panel = logits.argmax(dim=-1) + 1
            else:
                logits = logits / max(panel_temperature, 1e-6)
                if panel_top_k is not None:
                    kth = torch.topk(logits, panel_top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                panel = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1) + 1

            state = torch.where(onset[:, t], panel, torch.zeros_like(panel))
            gen[:, t] = state
            cur = torch.cat([cur, state.unsqueeze(1)], dim=1)

        bits = ((gen.unsqueeze(-1) >> torch.arange(4, device=device)) & 1).float()
        if lengths is not None:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            bits = bits * valid.unsqueeze(-1)
        return bits
