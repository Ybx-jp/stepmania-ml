"""
Typed (multi-step-type) autoregressive chart generator.

Same factorization that won Stage 3 — audio-driven onset head (binary: is there a
step) + autoregressive panel head — but the panel head now predicts a PER-PANEL
SYMBOL over {none, tap, hold-head, tail, roll-head} instead of one of 15 binary
patterns. Onset still controls density (calibrated threshold / Bernoulli); the panel
head fills in step *types* given an onset.

Almost the whole model is warm-startable from the focal factorized checkpoint
(audio encoder, onset branch, decoder, difficulty embedding) — only the per-panel
symbol embedding and the typed panel head are new.

See notes/step_types.md.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.encoders import AudioEncoder
from .transformer import PositionalEncoding, _causal_mask
from .factorized import _project, _attend, _LayerCache
from .typed import NUM_SYMBOLS, NUM_PANELS, NUM_PATTERNS, NUM_TYPES  # 5 symbols, 4 panels, 15 patterns, 4 types


class TypedChartGenerator(nn.Module):
    def __init__(self, audio_dim: int = 23, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, onset_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, num_difficulties: int = 4, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=d_model)
        self.diff_embedding = nn.Embedding(num_difficulties, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        # onset branch (identical to FactorizedChartGenerator -> warm-startable)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.onset_encoder = nn.TransformerEncoder(enc_layer, num_layers=onset_layers)
        self.onset_head = nn.Linear(d_model, 1)

        # panel branch (typed)
        self.symbol_embedding = nn.Embedding(NUM_SYMBOLS, d_model)   # per-panel symbol
        self.panel_pos = nn.Parameter(torch.randn(NUM_PANELS, d_model) * 0.02)  # which panel
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)         # decoder input at t=0
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.panel_head = nn.Linear(d_model, NUM_PANELS * NUM_SYMBOLS)  # -> (.,4,5)

    # ---- warm start ------------------------------------------------------------

    def load_compatible(self, factorized_state_dict: dict) -> int:
        """Load every parameter that matches (name+shape) from a focal factorized
        checkpoint — audio encoder, onset branch, decoder, diff embedding. The typed
        symbol embedding / panel head / bos / panel_pos stay freshly initialized.
        """
        own = self.state_dict()
        load = {k: v for k, v in factorized_state_dict.items()
                if k in own and own[k].shape == v.shape}
        own.update(load)
        self.load_state_dict(own)
        return len(load)

    def freeze_audio_encoder(self, freeze: bool = True):
        for p in self.audio_encoder.parameters():
            p.requires_grad = not freeze

    # ---- branch forwards -------------------------------------------------------

    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def onset_logits(self, memory, difficulty, mask=None):
        B, T, _ = memory.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)
        x = self.dropout(self.pos_encoding(memory) + diff)
        pad = (~mask.bool()) if mask is not None else None
        return self.onset_head(self.onset_encoder(x, src_key_padding_mask=pad)).squeeze(-1)

    def _state_emb(self, states: torch.Tensor) -> torch.Tensor:
        """(B, L, 4) symbols -> (B, L, d): sum of per-panel (symbol + panel) embeddings."""
        e = self.symbol_embedding(states.long()) + self.panel_pos  # (B,L,4,d)
        return e.sum(dim=2)

    def _decoder_input(self, states: torch.Tensor) -> torch.Tensor:
        """Right-shifted decoder input embeddings: pos 0 = BOS, pos t = embed(state_{t-1})."""
        B, T, _ = states.shape
        e = self._state_emb(states)  # (B,T,d) embed of each frame's full state
        return torch.cat([self.bos.expand(B, 1, -1), e[:, :-1]], dim=1)

    def panel_logits(self, memory, states, difficulty, mask=None):
        B, T, _ = states.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)
        tgt = self.dropout(self.pos_encoding(self._decoder_input(states)) + diff)
        causal = _causal_mask(T, states.device)
        pad = (~mask.bool()) if mask is not None else None
        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal,
                         tgt_key_padding_mask=pad, memory_key_padding_mask=pad)
        return self.panel_head(h).view(B, T, NUM_PANELS, NUM_SYMBOLS)

    def forward(self, audio, states, difficulty, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced. states (B,T,4) true symbols.
        Returns (onset_logits (B,T), panel_logits (B,T,4,5))."""
        memory = self.encode_audio(audio)
        return (self.onset_logits(memory, difficulty, mask),
                self.panel_logits(memory, states, difficulty, mask))

    # ---- generation (KV-cached) ------------------------------------------------

    def _decoder_step_cached(self, x, caches):
        for layer, cache in zip(self.decoder.layers, caches):
            q = _project(layer.self_attn, x, 'q')
            kn = _project(layer.self_attn, x, 'k'); vn = _project(layer.self_attn, x, 'v')
            cache.k = kn if cache.k is None else torch.cat([cache.k, kn], dim=1)
            cache.v = vn if cache.v is None else torch.cat([cache.v, vn], dim=1)
            x = layer.norm1(x + _attend(layer.self_attn, q, cache.k, cache.v))
            cq = _project(layer.multihead_attn, x, 'q')
            x = layer.norm2(x + _attend(layer.multihead_attn, cq, cache.mem_k, cache.mem_v))
            x = layer.norm3(x + layer.linear2(F.relu(layer.linear1(x))))
        return x

    @torch.no_grad()
    def generate(self, audio, difficulty, lengths=None, onset_threshold=0.5,
                 onset_sample=False, onset_logit_scale=1.0, onset_logit_bias=0.0,
                 onset_override=None, panel_greedy=True, panel_temperature=1.0,
                 panel_top_k=None) -> torch.Tensor:
        """KV-cached decode -> typed (B, T, 4) chart (symbols 0..4).

        Onset (audio-driven, one pass) sets where steps go; at onset frames each panel's
        symbol is decoded autoregressively. At least one panel is forced non-empty on an
        onset frame.
        """
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        memory = self.encode_audio(audio)
        if onset_override is not None:
            onset = onset_override.bool().to(device)
        else:
            p = torch.sigmoid(onset_logit_scale * self.onset_logits(memory, difficulty) + onset_logit_bias)
            onset = torch.bernoulli(p).bool() if onset_sample else (p > onset_threshold)

        caches = [_LayerCache(layer, memory) for layer in self.decoder.layers]
        diff_emb = self.diff_embedding(difficulty).unsqueeze(1)
        gen = torch.zeros(B, T, NUM_PANELS, dtype=torch.long, device=device)
        prev_emb = self.bos.expand(B, 1, -1)  # decoder input at t=0

        for t in range(T):
            x = prev_emb + self.pos_encoding.pe[:, t:t + 1] + diff_emb
            h = self._decoder_step_cached(x, caches)
            logits = self.panel_head(h[:, -1]).view(B, NUM_PANELS, NUM_SYMBOLS)  # (B,4,5)
            if panel_greedy:
                sym = logits.argmax(dim=-1)  # (B,4)
            else:
                lg = logits / max(panel_temperature, 1e-6)
                if panel_top_k is not None:
                    kth = torch.topk(lg, panel_top_k, dim=-1).values[..., -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                probs = torch.softmax(lg, dim=-1)
                sym = torch.multinomial(probs.view(-1, NUM_SYMBOLS), 1).view(B, NUM_PANELS)

            # enforce >=1 non-empty panel where onset fired
            empty_all = (sym == 0).all(dim=1) & onset[:, t]
            if empty_all.any():
                # for those rows, pick the panel with the highest non-none logit
                nonnone = logits[:, :, 1:]  # (B,4,4)
                best = nonnone.reshape(B, -1).argmax(dim=1)
                bp, bs = best // (NUM_SYMBOLS - 1), best % (NUM_SYMBOLS - 1) + 1
                rows = empty_all.nonzero(as_tuple=True)[0]
                sym[rows, bp[rows]] = bs[rows]

            state = torch.where(onset[:, t].unsqueeze(1), sym, torch.zeros_like(sym))
            gen[:, t] = state
            prev_emb = self._state_emb(state.unsqueeze(1))  # (B,1,d) for next step

        if lengths is not None:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            gen = gen * valid.unsqueeze(-1)
        return gen


class LayeredTypedChartGenerator(nn.Module):
    """Layered factorization: onset (frame has a step) -> pattern (WHICH panels are active,
    15-way — the binary model's proven head) -> type (per active panel: tap/hold-head/tail/roll,
    4-way). Decouples is-panel-active from what-type, avoiding the none/active conflation that
    made the flat per-panel 5-way head none-biased.
    """

    def __init__(self, audio_dim: int = 23, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, onset_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, num_difficulties: int = 4, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=d_model)
        self.diff_embedding = nn.Embedding(num_difficulties, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.onset_encoder = nn.TransformerEncoder(enc_layer, num_layers=onset_layers)
        self.onset_head = nn.Linear(d_model, 1)
        self.symbol_embedding = nn.Embedding(NUM_SYMBOLS, d_model)
        self.panel_pos = nn.Parameter(torch.randn(NUM_PANELS, d_model) * 0.02)
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.pattern_head = nn.Linear(d_model, NUM_PATTERNS)          # which panels (15-way)
        self.type_head = nn.Linear(d_model, NUM_PANELS * NUM_TYPES)   # per-panel type (4-way)

    def load_from_factorized(self, sd: dict) -> int:
        """Warm-start: name+shape matches (audio enc, onset branch, decoder, diff emb) PLUS
        map the binary panel_head (Linear d->15) onto the pattern_head (same shape, same job)."""
        own = self.state_dict()
        load = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        for src, dst in [("panel_head.weight", "pattern_head.weight"),
                         ("panel_head.bias", "pattern_head.bias")]:
            if src in sd and own[dst].shape == sd[src].shape:
                load[dst] = sd[src]
        own.update(load); self.load_state_dict(own)
        return len(load)

    def freeze_audio_encoder(self, freeze: bool = True):
        for p in self.audio_encoder.parameters():
            p.requires_grad = not freeze

    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def onset_logits(self, memory, difficulty, mask=None):
        B, T, _ = memory.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)
        x = self.dropout(self.pos_encoding(memory) + diff)
        pad = (~mask.bool()) if mask is not None else None
        return self.onset_head(self.onset_encoder(x, src_key_padding_mask=pad)).squeeze(-1)

    def _state_emb(self, states):
        return (self.symbol_embedding(states.long()) + self.panel_pos).sum(dim=2)

    def _decoder_input(self, states):
        B = states.shape[0]
        e = self._state_emb(states)
        return torch.cat([self.bos.expand(B, 1, -1), e[:, :-1]], dim=1)

    def _decode(self, memory, states, difficulty, mask=None):
        B, T, _ = states.shape
        diff = self.diff_embedding(difficulty).unsqueeze(1).expand(B, T, -1)
        tgt = self.dropout(self.pos_encoding(self._decoder_input(states)) + diff)
        causal = _causal_mask(T, states.device)
        pad = (~mask.bool()) if mask is not None else None
        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal,
                         tgt_key_padding_mask=pad, memory_key_padding_mask=pad)
        return self.pattern_head(h), self.type_head(h).view(B, T, NUM_PANELS, NUM_TYPES)

    def forward(self, audio, states, difficulty, mask=None):
        """Returns (onset_logits (B,T), pattern_logits (B,T,15), type_logits (B,T,4,4))."""
        memory = self.encode_audio(audio)
        ol = self.onset_logits(memory, difficulty, mask)
        pat, typ = self._decode(memory, states, difficulty, mask)
        return ol, pat, typ

    def _decoder_step_cached(self, x, caches):
        for layer, cache in zip(self.decoder.layers, caches):
            q = _project(layer.self_attn, x, 'q')
            kn = _project(layer.self_attn, x, 'k'); vn = _project(layer.self_attn, x, 'v')
            cache.k = kn if cache.k is None else torch.cat([cache.k, kn], dim=1)
            cache.v = vn if cache.v is None else torch.cat([cache.v, vn], dim=1)
            x = layer.norm1(x + _attend(layer.self_attn, q, cache.k, cache.v))
            cq = _project(layer.multihead_attn, x, 'q')
            x = layer.norm2(x + _attend(layer.multihead_attn, cq, cache.mem_k, cache.mem_v))
            x = layer.norm3(x + layer.linear2(F.relu(layer.linear1(x))))
        return x

    @torch.no_grad()
    def generate(self, audio, difficulty, lengths=None, onset_threshold=0.5,
                 onset_sample=False, onset_logit_scale=1.0, onset_logit_bias=0.0,
                 onset_override=None, greedy=True, temperature=1.0,
                 type_sample=False, type_temperature=1.0, hold_aware=False,
                 pattern_sample=False, pattern_temperature=1.0, pattern_top_k=None,
                 repetition_penalty=1.0, pattern_bias=None, no_crossovers=False):
        """KV-cached decode -> typed (B, T, 4). onset -> pattern (which panels, >=1 guaranteed)
        -> per-active-panel type. No enforcement needed (all 15 patterns are non-empty).

        `type_sample` samples the per-panel TYPE (so rare holds appear at ~their predicted
        rate) while keeping the pattern (which-panels) greedy — holds never beat tap under
        greedy argmax, so sampling the type is how they surface.

        `hold_aware` runs a per-panel hold automaton: a hold/roll-head OPENS a hold and the
        panel is then occupied; the hold CLOSES (emits a tail) at the next frame the model
        places a note on that panel — so holds span a musically coherent, audio-aligned
        duration (head -> next note) and are always valid (no orphans), like a human author.
        Frames between head and tail emit nothing on the held panel."""
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        memory = self.encode_audio(audio)
        if onset_override is not None:
            onset = onset_override.bool().to(device)
        else:
            p = torch.sigmoid(onset_logit_scale * self.onset_logits(memory, difficulty) + onset_logit_bias)
            onset = torch.bernoulli(p).bool() if onset_sample else (p > onset_threshold)

        # precompute which-panel bit table for the 15 patterns: (15,4)
        states_tab = torch.arange(1, NUM_PATTERNS + 1, device=device)
        panel_bits = ((states_tab.unsqueeze(-1) >> torch.arange(NUM_PANELS, device=device)) & 1)  # (15,4)

        caches = [_LayerCache(layer, memory) for layer in self.decoder.layers]
        diff_emb = self.diff_embedding(difficulty).unsqueeze(1)
        gen = torch.zeros(B, T, NUM_PANELS, dtype=torch.long, device=device)
        prev_emb = self.bos.expand(B, 1, -1)
        held = torch.zeros(B, NUM_PANELS, dtype=torch.bool, device=device)  # hold automaton state
        prev_pat = torch.full((B,), -1, dtype=torch.long, device=device)    # previous note's pattern (for rep penalty)
        next_foot = torch.zeros(B, dtype=torch.long, device=device)         # 0=left, 1=right (crossover automaton)
        if pattern_bias is not None:
            pattern_bias = torch.as_tensor(pattern_bias, dtype=torch.float32, device=device)  # (15,)
        n_panels = panel_bits.sum(1)                                        # (15,) panels per pattern

        for t in range(T):
            x = prev_emb + self.pos_encoding.pe[:, t:t + 1] + diff_emb
            h = self._decoder_step_cached(x, caches)[:, -1]
            pat_logits = self.pattern_head(h)                                  # (B,15)
            typ_logits = self.type_head(h).view(B, NUM_PANELS, NUM_TYPES)      # (B,4,4)
            # pattern (which panels). greedy collapses to Left/jacks; sampling adds variety.
            if pattern_bias is not None:                  # pattern-preference knob (jumps, panel prefs)
                pat_logits = pat_logits + pattern_bias
            if repetition_penalty != 1.0:  # discourage repeating the previous note's pattern (jacks)
                has_prev = prev_pat >= 0
                if has_prev.any():
                    rows = has_prev.nonzero(as_tuple=True)[0]
                    pat_logits[rows, prev_pat[rows]] -= float(np.log(repetition_penalty))
            if no_crossovers:  # forbid single-note pattern that crosses for the current foot
                forbid_panel = torch.where(next_foot == 0, 3, 0)             # left->R, right->L
                forbid_idx = (1 << forbid_panel) - 1                         # single-panel pattern index
                pat_logits[torch.arange(B, device=device), forbid_idx] = float("-inf")
            if pattern_sample or not greedy:
                lg = pat_logits / (pattern_temperature if pattern_sample else temperature)
                if pattern_top_k is not None:
                    kth = torch.topk(lg, pattern_top_k, dim=-1).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                pat = torch.multinomial(torch.softmax(lg, dim=-1), 1).squeeze(-1)
            else:
                pat = pat_logits.argmax(-1)
            # type (per panel): sample if requested (lets rare holds surface), else greedy
            if type_sample or not greedy:
                tt = type_temperature if type_sample else temperature
                typ = torch.multinomial(torch.softmax(typ_logits / tt, -1).view(-1, NUM_TYPES), 1).view(B, NUM_PANELS)
            else:
                typ = typ_logits.argmax(-1)
            active = panel_bits[pat].bool() & onset[:, t].unsqueeze(1)         # panels the model notes this frame

            if hold_aware:
                proposed = typ + 1                                            # symbol 1..4
                close = held & active                                         # model notes a held panel -> close it
                free_act = (~held) & active                                   # fresh note on a free panel
                prop = torch.where(proposed == 3, torch.ones_like(proposed), proposed)  # tail-on-free -> tap
                state = torch.zeros(B, NUM_PANELS, dtype=torch.long, device=device)
                state = torch.where(close, torch.full_like(state, 3), state)  # tail closes the hold
                state = torch.where(free_act, prop, state)                    # tap/head/roll on free panels
                held = (held & ~close) | (free_act & ((prop == 2) | (prop == 4)))
            else:
                state = torch.where(active, typ + 1, torch.zeros_like(typ))   # stateless per-panel symbol

            gen[:, t] = state
            prev_emb = self._state_emb(state.unsqueeze(1))
            on = onset[:, t]
            prev_pat = torch.where(on, pat, prev_pat)  # track last note's pattern
            npc = n_panels[pat]                        # panels in the chosen pattern
            next_foot = torch.where(on & (npc == 1), 1 - next_foot, next_foot)   # alternate on singles
            next_foot = torch.where(on & (npc >= 2), torch.zeros_like(next_foot), next_foot)  # reset after jump

        if lengths is not None:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            gen = gen * valid.unsqueeze(-1)
        return gen
