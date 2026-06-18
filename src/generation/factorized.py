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
import torch.nn.functional as F

from src.models.components.encoders import AudioEncoder
from .tokenizer import VOCAB_SIZE, NUM_PANEL_STATES
from .transformer import PositionalEncoding, _causal_mask

NUM_NONEMPTY = NUM_PANEL_STATES - 1  # 15 panel patterns excluding the empty state


def _project(mha: nn.MultiheadAttention, x: torch.Tensor, which: str) -> torch.Tensor:
    """Apply one of nn.MultiheadAttention's packed in-projections (q/k/v) to x."""
    E = mha.embed_dim
    w = mha.in_proj_weight
    b = mha.in_proj_bias
    sl = {'q': slice(0, E), 'k': slice(E, 2 * E), 'v': slice(2 * E, 3 * E)}[which]
    return F.linear(x, w[sl], b[sl] if b is not None else None)


def _attend(mha: nn.MultiheadAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multi-head scaled-dot-product attention with pre-projected q/k/v, then out_proj.

    q: (B, Lq, E); k/v: (B, Lk, E). No causal mask needed — for incremental self-attn
    the key cache only holds past+current tokens.
    """
    B, Lq, E = q.shape
    H = mha.num_heads
    hd = E // H
    qh = q.view(B, Lq, H, hd).transpose(1, 2)            # (B,H,Lq,hd)
    kh = k.view(B, -1, H, hd).transpose(1, 2)
    vh = v.view(B, -1, H, hd).transpose(1, 2)
    out = F.scaled_dot_product_attention(qh, kh, vh)      # (B,H,Lq,hd)
    out = out.transpose(1, 2).reshape(B, Lq, E)
    return mha.out_proj(out)


class _LayerCache:
    """Per-decoder-layer cache: growing self-attn K/V + fixed cross-attn K/V from memory."""
    def __init__(self, layer: nn.TransformerDecoderLayer, memory: torch.Tensor):
        self.k = None  # (B, t, E) self-attention keys so far
        self.v = None
        self.mem_k = _project(layer.multihead_attn, memory, 'k')  # cross-attn K (computed once)
        self.mem_v = _project(layer.multihead_attn, memory, 'v')


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
        onset_override: Optional[torch.Tensor] = None,
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

        `onset_override` (B, T) bool, if given, is used directly as the onset
        decision — letting the caller implement any policy (threshold, Bernoulli,
        hybrid) from the precomputed non-AR onset posteriors. Bypasses the args above.
        """
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        from .tokenizer import BOS_TOKEN

        memory = self.encode_audio(audio)
        if onset_override is not None:
            onset = onset_override.bool().to(device)
        else:
            p_onset = torch.sigmoid(onset_logit_scale * self.onset_logits(memory, difficulty) + onset_logit_bias)
            onset = torch.bernoulli(p_onset).bool() if onset_sample else (p_onset > onset_threshold)

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

    def _decoder_step_cached(self, x: torch.Tensor, caches) -> torch.Tensor:
        """One post-norm decoder step for the new token x (B,1,d), using per-layer caches.

        Mirrors nn.TransformerDecoderLayer (norm_first=False, relu): self-attn over
        cached K/V (append new), cross-attn to fixed memory K/V, then FFN.
        """
        for layer, cache in zip(self.decoder.layers, caches):
            # --- self-attention (incremental) ---
            q = _project(layer.self_attn, x, 'q')
            k_new = _project(layer.self_attn, x, 'k')
            v_new = _project(layer.self_attn, x, 'v')
            cache.k = k_new if cache.k is None else torch.cat([cache.k, k_new], dim=1)
            cache.v = v_new if cache.v is None else torch.cat([cache.v, v_new], dim=1)
            sa = _attend(layer.self_attn, q, cache.k, cache.v)
            x = layer.norm1(x + sa)
            # --- cross-attention to fixed audio memory ---
            cq = _project(layer.multihead_attn, x, 'q')
            ca = _attend(layer.multihead_attn, cq, cache.mem_k, cache.mem_v)
            x = layer.norm2(x + ca)
            # --- feed-forward ---
            ff = layer.linear2(F.relu(layer.linear1(x)))
            x = layer.norm3(x + ff)
        return x

    @torch.no_grad()
    def generate_cached(
        self,
        audio: torch.Tensor,
        difficulty: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        onset_threshold: float = 0.5,
        onset_sample: bool = False,
        onset_logit_scale: float = 1.0,
        onset_logit_bias: float = 0.0,
        onset_override: Optional[torch.Tensor] = None,
        panel_greedy: bool = True,
        panel_temperature: float = 1.0,
        panel_top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """KV-cached equivalent of generate(): O(T) instead of O(T^2) decoding.

        Same arguments and (for greedy) same output as generate(), but each step
        processes only the new token against cached keys/values — enabling
        full-length (1440-frame) generation at reasonable cost.
        """
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        from .tokenizer import BOS_TOKEN

        memory = self.encode_audio(audio)
        if onset_override is not None:
            onset = onset_override.bool().to(device)
        else:
            p_onset = torch.sigmoid(onset_logit_scale * self.onset_logits(memory, difficulty) + onset_logit_bias)
            onset = torch.bernoulli(p_onset).bool() if onset_sample else (p_onset > onset_threshold)

        caches = [_LayerCache(layer, memory) for layer in self.decoder.layers]
        diff_emb = self.diff_embedding(difficulty).unsqueeze(1)  # (B,1,d)
        gen = torch.zeros(B, T, dtype=torch.long, device=device)
        tok = torch.full((B,), BOS_TOKEN, dtype=torch.long, device=device)

        for t in range(T):
            # embed the current token at position t (matches non-cached pos encoding)
            x = self.token_embedding(tok).unsqueeze(1)  # (B,1,d)
            x = x + self.pos_encoding.pe[:, t:t + 1] + diff_emb
            h = self._decoder_step_cached(x, caches)
            logits = self.panel_head(h[:, -1])  # (B, 15)
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
            tok = state  # next step's input token

        bits = ((gen.unsqueeze(-1) >> torch.arange(4, device=device)) & 1).float()
        if lengths is not None:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            bits = bits * valid.unsqueeze(-1)
        return bits
