"""
Stage 2: autoregressive transformer chart generator.

Predicts the panel-state at frame t from the aligned song audio (full,
non-causal) and the steps generated so far (causal). Reuses the Phase 1
AudioEncoder as the conditioning encoder (warm-startable from the best
classifier), so d_model defaults to the encoder's 128-dim output.

Sequence layout (audio/step frames are 1:1 aligned, length T):
    decoder input : [BOS, s_0, ..., s_{T-2}]   (length T)
    target        : [s_0, s_1, ..., s_{T-1}]   (length T)
    audio memory  : AudioEncoder(audio)         (B, T, d)

Position t predicts s_t from [BOS, s_0..s_{t-1}] (causal self-attn) and the
audio memory (cross-attn). Difficulty is a learned embedding added at every
decoder position.

See docs/phase2_generative_design.md.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.components.encoders import AudioEncoder
from .tokenizer import VOCAB_SIZE, NUM_PANEL_STATES, BOS_TOKEN


def _causal_mask(size: int, device) -> torch.Tensor:
    """Bool causal attn mask, True = disallowed (position i may not attend to j>i)."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ChartGenerator(nn.Module):
    """Audio-conditioned, difficulty-conditioned autoregressive step generator."""

    def __init__(
        self,
        audio_dim: int = 23,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_difficulties: int = 4,
        max_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Conditioning encoder (reused / warm-started from Phase 1).
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=d_model)

        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.diff_embedding = nn.Embedding(num_difficulties, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, VOCAB_SIZE)

    # ---- warm start -------------------------------------------------------------

    def load_audio_encoder(self, classifier_state_dict: dict) -> int:
        """Copy AudioEncoder weights from a Phase 1 classifier state_dict.

        Returns the number of tensors loaded. Keys are prefixed 'audio_encoder.'.
        """
        prefix = "audio_encoder."
        sub = {k[len(prefix):]: v for k, v in classifier_state_dict.items() if k.startswith(prefix)}
        missing, unexpected = self.audio_encoder.load_state_dict(sub, strict=False)
        if unexpected:
            raise ValueError(f"unexpected audio_encoder keys: {unexpected}")
        return len(sub) - len(missing)

    def freeze_audio_encoder(self, freeze: bool = True):
        for p in self.audio_encoder.parameters():
            p.requires_grad = not freeze

    # ---- forward ----------------------------------------------------------------

    def _embed_tokens(self, in_tokens: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        B, S = in_tokens.shape
        tok = self.token_embedding(in_tokens) * math.sqrt(self.d_model)
        tok = self.pos_encoding(tok)
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, S, -1)
        return self.dropout(tok + diff)

    def forward(
        self,
        audio: torch.Tensor,
        in_tokens: torch.Tensor,
        difficulty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward.

        Args:
            audio: (B, T, audio_dim) aligned audio features.
            in_tokens: (B, T) decoder input ids ([BOS, s_0, ..., s_{T-2}]).
            difficulty: (B,) long difficulty class.
            mask: (B, T) bool, True = valid frame. Padding is ignored in attention.

        Returns:
            (B, T, VOCAB_SIZE) logits.
        """
        memory = self.audio_encoder(audio)  # (B, T, d)
        tgt = self._embed_tokens(in_tokens, difficulty)  # (B, T, d)

        S = in_tokens.size(1)
        causal = _causal_mask(S, in_tokens.device)

        # key_padding_mask expects True = ignore (opposite of our valid mask).
        pad = (~mask.bool()) if mask is not None else None

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad,
            memory_key_padding_mask=pad,
        )
        return self.output_head(out)

    # ---- diagnostics ------------------------------------------------------------

    @torch.no_grad()
    def onset_posteriors(
        self,
        audio: torch.Tensor,
        in_tokens: torch.Tensor,
        difficulty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced per-frame P(onset) = 1 - P(empty panel-state). Shape (B, T).

        Decoupled from autoregressive sampling and density: a clean "does the model
        know where notes go, given clean context" signal.
        """
        self.eval()
        logits = self.forward(audio, in_tokens, difficulty, mask)  # (B, T, VOCAB)
        probs = torch.softmax(logits[..., :NUM_PANEL_STATES], dim=-1)  # over panel-states
        return 1.0 - probs[..., 0]  # state 0 = empty (0b0000)

    # ---- generation -------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        difficulty: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = True,
        onset_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Batched autoregressive decode. Returns (B, T, 4) binary charts.

        Steps are sampled only over the 16 panel-states (specials are masked out).
        `lengths` (B,) optionally caps each song's valid length; frames beyond it
        are emitted as empty.

        If `onset_threshold` is set, decoding is density-controlled: a frame gets a
        step iff P(onset) = 1 - P(empty) > threshold, and the panel pattern is the
        argmax over the 15 non-empty states. This decouples "how many" (threshold)
        from "which arrows" (panel argmax), bypassing temperature.
        """
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        memory = self.audio_encoder(audio)

        gen = torch.zeros(B, T, dtype=torch.long, device=device)  # generated states
        cur = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)

        for t in range(T):
            tgt = self._embed_tokens(cur, difficulty)  # (B, t+1, d)
            S = cur.size(1)
            causal = _causal_mask(S, device)
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            logits = self.output_head(out[:, -1])  # (B, VOCAB)
            logits = logits[:, :NUM_PANEL_STATES]  # restrict to panel-states

            if onset_threshold is not None:
                probs = torch.softmax(logits, dim=-1)  # (B, 16)
                p_onset = 1.0 - probs[:, 0]
                # best non-empty panel pattern (states 1..15)
                panel = probs[:, 1:].argmax(dim=-1) + 1
                nxt = torch.where(p_onset > onset_threshold, panel,
                                  torch.zeros_like(panel))
            elif greedy:
                nxt = logits.argmax(dim=-1)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)

            gen[:, t] = nxt
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)

        # decode (B,T) states -> (B,T,4) bits
        bits = ((gen.unsqueeze(-1) >> torch.arange(4, device=device)) & 1).float()
        if lengths is not None:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            bits = bits * valid.unsqueeze(-1)
        return bits
