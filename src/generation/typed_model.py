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

from src.models.components.encoders import AudioEncoder, ChartEncoder
from .transformer import PositionalEncoding, _causal_mask
from .factorized import _project, _attend, _LayerCache
from .typed import NUM_SYMBOLS, NUM_PANELS, NUM_PATTERNS, NUM_TYPES  # 5 symbols, 4 panels, 15 patterns, 4 types

MOTIF_DIM = 12  # H15 motif-knob conditioning width (MotifBasis.K; see src/generation/motif_codebook.py)


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


class StyleEncoder(nn.Module):
    """Reference-chart style encoder (Step 3). Reuses the Phase 1 ChartEncoder (embedding +
    Conv1D blocks) to map a reference chart's per-frame occupancy (B,L,4) -> (B,L,d), then
    masked-mean-pools over time to a single (B,d) STYLE LATENT. The temporal pool is the
    bottleneck: it can only carry global feel (density, which-panels / jack tendencies),
    not the exact note sequence, so conditioning on it transfers style rather than copying.
    """

    def __init__(self, d_model: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.encoder = ChartEncoder(input_dim=NUM_PANELS, embedding_dim=embedding_dim, hidden_dim=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ref_occupancy: torch.Tensor, ref_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ref_occupancy (B,L,4) binary; ref_mask (B,L) bool (valid frames) -> (B,d) latent."""
        h = self.encoder(ref_occupancy)  # (B,L,d)
        if ref_mask is not None:
            m = ref_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(1)
        return self.norm(pooled)


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
        # target groove-radar conditioning (Step 2): 5 dims -> d; null embedding for CFG dropout
        self.radar_proj = nn.Linear(5, d_model)
        self.null_radar = nn.Parameter(torch.zeros(d_model))
        # reference-chart style conditioning (Step 3): encode a reference chart -> (B,d) latent
        self.style_encoder = StyleEncoder(d_model=d_model)
        self.null_style = nn.Parameter(torch.zeros(d_model))
        # H15 motif-vocabulary conditioning: radar-orthogonal motif-style knobs (MotifBasis) -> d; null for CFG.
        # zero-init the proj so a warm-start begins as an identity no-op and the motif lever is LEARNED
        # (preserves the warm-started model's quality at step 0; nothing changes until motif gradients flow).
        self.motif_dim = MOTIF_DIM
        self.motif_proj = nn.Linear(MOTIF_DIM, d_model)
        nn.init.zeros_(self.motif_proj.weight); nn.init.zeros_(self.motif_proj.bias)
        self.null_motif = nn.Parameter(torch.zeros(d_model))

    def encode_style(self, reference, reference_mask=None):
        """Reference chart -> (B,d) style latent. `reference` is (B,L,4) typed symbols (0..4) or
        binary occupancy; we condition on per-frame occupancy (step present on a panel).
        Padded frames are zeroed before encoding so they read as empty (the conv mixes
        neighbors, so garbage in the pad would otherwise leak across the mask boundary)."""
        occ = (reference != 0).float() if reference.dtype != torch.float32 else (reference > 0).float()
        if reference_mask is not None:
            occ = occ * reference_mask.unsqueeze(-1).float()
        return self.style_encoder(occ, reference_mask)

    def _cond(self, difficulty, radar, style=None, motif=None):
        """Per-position conditioning, shape (B,1,d) (broadcast over time) — or (B,T,d) when `motif`
        is a PER-FRAME schedule (B,T,K): the local-motif path (notes/h15_local_motif_plan.md). The
        global motif vector (B,K) collapses to one constant added to every frame (weak per-frame
        gradient — H15 Phase-2 root cause); a (B,T,K) schedule lets conditioning VARY by section.
        difficulty + groove-radar (or null) + reference-style (or null) + motif knobs (or null)."""
        c = self.diff_embedding(difficulty)
        c = c + (self.radar_proj(radar) if radar is not None else self.null_radar.unsqueeze(0))
        c = c + (style if style is not None else self.null_style.unsqueeze(0))
        c = c.unsqueeze(1)                                                # (B,1,d)
        if motif is None:
            m = self.null_motif.view(1, 1, -1)                           # (1,1,d)
        else:
            m = self.motif_proj(motif)                                   # (B,K)->(B,d) | (B,T,K)->(B,T,d)
            m = m.unsqueeze(1) if m.dim() == 2 else m                    # (B,1,d) | (B,T,d)
        return c + m                                                     # (B,1,d) | (B,T,d)

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

    def onset_logits(self, memory, difficulty, mask=None, radar=None, style=None, motif=None):
        # Motif shapes WHICH panels (a pattern-head concern) — the MotifBasis is which-panels only, rhythm &
        # density excluded. Timing/density is the onset head's job, controlled by radar. Conditioning onset on
        # motif was spurious and let CFG inflate density (notes/h15_local_motif_plan.md), so onset sees
        # difficulty+radar+style only; motif is applied in _decode. `motif` kept in the signature for call-site
        # symmetry but intentionally NOT passed to the onset conditioning.
        cond = self._cond(difficulty, radar, style, motif=None)   # onset: NO motif (density stays radar-controlled)
        x = self.dropout(self.pos_encoding(memory) + cond)
        pad = (~mask.bool()) if mask is not None else None
        return self.onset_head(self.onset_encoder(x, src_key_padding_mask=pad)).squeeze(-1)

    def _state_emb(self, states):
        return (self.symbol_embedding(states.long()) + self.panel_pos).sum(dim=2)

    def _decoder_input(self, states):
        B = states.shape[0]
        e = self._state_emb(states)
        return torch.cat([self.bos.expand(B, 1, -1), e[:, :-1]], dim=1)

    def _decode(self, memory, states, difficulty, mask=None, radar=None, style=None, motif=None):
        B, T, _ = states.shape
        cond = self._cond(difficulty, radar, style, motif)   # (B,1,d) broadcast | (B,T,d) per-frame
        tgt = self.dropout(self.pos_encoding(self._decoder_input(states)) + cond)
        causal = _causal_mask(T, states.device)
        pad = (~mask.bool()) if mask is not None else None
        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal,
                         tgt_key_padding_mask=pad, memory_key_padding_mask=pad)
        return self.pattern_head(h), self.type_head(h).view(B, T, NUM_PANELS, NUM_TYPES)

    def forward(self, audio, states, difficulty, mask=None, radar=None,
                reference=None, reference_mask=None, style=None, motif=None):
        """Returns (onset_logits (B,T), pattern_logits (B,T,15), type_logits (B,T,4,4)).
        `radar` (B,5) optional target groove-radar conditioning (None -> null/CFG).
        `reference` (B,L,4) optional reference chart for style conditioning; or pass a
        precomputed `style` (B,d) latent directly. Both None -> null style/CFG.
        `motif` (B,MOTIF_DIM) optional motif-knob conditioning (None -> null motif/CFG)."""
        memory = self.encode_audio(audio)
        if style is None and reference is not None:
            style = self.encode_style(reference, reference_mask)
        ol = self.onset_logits(memory, difficulty, mask, radar, style, motif)
        pat, typ = self._decode(memory, states, difficulty, mask, radar, style, motif)
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
                 repetition_penalty=1.0, pattern_bias=None, no_crossovers=False, radar=None,
                 guidance_scale=1.0, reference=None, reference_mask=None, style=None, motif=None,
                 no_jump_during_hold=False, onset_phase_penalty=0.0, no_cross_during_hold=False,
                 boundary_reset=None, onset_phase_alloc=None, onset_phase_calib=None, max_jack_run=None):
        """KV-cached decode -> typed (B, T, 4). onset -> pattern (which panels, >=1 guaranteed)
        -> per-active-panel type. No enforcement needed (all 15 patterns are non-empty).

        `type_sample` samples the per-panel TYPE (so rare holds appear at ~their predicted
        rate) while keeping the pattern (which-panels) greedy — holds never beat tap under
        greedy argmax, so sampling the type is how they surface.

        `hold_aware` runs a per-panel hold automaton: a hold/roll-head OPENS a hold and the
        panel is then occupied; the hold CLOSES (emits a tail) at the next frame the model
        places a note on that panel — so holds span a musically coherent, audio-aligned
        duration (head -> next note) and are always valid (no orphans), like a human author.
        Frames between head and tail emit nothing on the held panel.

        `no_jump_during_hold` (needs hold_aware): a pad player holding one panel has only one
        free foot, so a jump (>=2 fresh presses) while a hold is open is unhittable. When set,
        any pattern that would place >=2 fresh notes on non-held panels is forbidden while a
        hold is open (closing the held panel and single taps stay allowed).

        `onset_phase_penalty` (logits; on threshold/Bernoulli onsets only): a metric gate that
        subtracts from off-beat onset logits (on-beat 0, 8th -p, 16th -2p) so off-beats survive
        only where the onset head is confident. Restores the downbeat anchor under chaos
        conditioning (which otherwise floods off-beats into a uniform smear). ~0.5-1.5 is a useful
        range; 0 = off.

        `onset_phase_alloc` (3-tuple of note shares quarter,8th,16th; e.g. real's (0.707,0.252,0.041)):
        a phase-aware onset threshold. The single density-quantile threshold buries 16ths -- they are
        out-budgeted by the more-confident 8ths, so the model's real 16th confidence (p_on@16th ~0.4)
        never clears tau. This keeps the SAME note budget but allocates it across the three phase bands by
        the given shares, picking the top-p_on frames within each band, so the model's own ranking chooses
        WHICH 16ths. Density is preserved; the rhythm distribution is steered to the target. None = off.
        NOTE: a flat per-song quota is smearing (forces the same 16th share on every song); prefer
        `onset_phase_calib` for VARIABLE per-song chaos.

        `onset_phase_calib` (b8, b16): per-phase calibration offset in LOGIT space (8th += b8, 16th += b16),
        applied before the per-song threshold. Corrects the model's systematic 16th under-confidence so the
        16th count floats with the audio -- chaotic songs get many 16ths, calm songs ~none -- per-song
        normalized (vs the flat `onset_phase_alloc` quota). The caller's threshold must be computed from the
        same offset logits. Validated in diag_song_chaos.py (variable, real-range; corr-capped by the
        model's frame-local song-level signal ~0.4). None = off.

        `max_jack_run` (H13 EXERTION): cap on consecutive same-single-panel presses at FAST (16th-adjacent)
        spacing -- one foot hammering one arrow at 16th speed is physically brutal (playtest: "a 6-note jack
        on 1/16s is crazy"). Real Hard charts essentially never do this (jack-pair-rate ~0.006, max fast run
        ~1; measured over 786 charts) -- they ALTERNATE panels so the feet alternate. The pattern head,
        sampled per-frame, repeats a panel ~28% of fast pairs (diag_exertion_h13.py). When set, once a fast
        same-panel run reaches the cap, that panel is forbidden on the next 16th-adjacent single onset,
        forcing a different panel (foot alternation). =1 matches real (strict alternation); only 16th-adjacent
        runs are capped (normal slower jacks untouched); jumps reset the run. None = off."""
        self.eval()
        device = audio.device
        B, T, _ = audio.shape
        memory = self.encode_audio(audio)
        if style is None and reference is not None:
            style = self.encode_style(reference, reference_mask)
        do_cfg = (guidance_scale != 1.0) and (radar is not None or style is not None or motif is not None)  # amplify radar/style/motif conditioning
        if onset_override is not None:
            onset = onset_override.bool().to(device)
        else:
            ol = self.onset_logits(memory, difficulty, radar=radar, style=style, motif=motif)
            if do_cfg:  # classifier-free guidance: push onset toward the conditioned prediction
                ol_u = self.onset_logits(memory, difficulty, radar=None, style=None, motif=None)
                ol = ol_u + guidance_scale * (ol - ol_u)
            if onset_phase_penalty != 0.0:
                # metric gate: off-beat frames need higher onset confidence (on-beat free, 8th -p, 16th -2p).
                # Restores the downbeat anchor under chaos conditioning -> off-beats survive only where the
                # audio-driven onset head is confident, instead of a uniform off-grid smear.
                ph = torch.arange(T, device=device) % 4          # 16th-grid phase within a beat
                pen = torch.where(ph == 0, 0.0, torch.where(ph == 2, onset_phase_penalty, 2.0 * onset_phase_penalty))
                ol = ol - pen.unsqueeze(0)                         # (T,) broadcast over batch
            if onset_phase_calib is not None:
                # per-phase calibration offset (b8 on 8th frames, b16 on 16th frames), ADDED to the onset
                # logits, then the usual per-song threshold runs on the recalibrated probs. Corrects the
                # model's systematic 16th under-confidence so the 16th COUNT floats with the audio (chaotic
                # song -> many 16ths, calm song -> none), per-song-normalized (unlike a flat quota). The
                # caller must compute its threshold from the SAME offset logits. See diag_song_chaos.py.
                b8, b16 = onset_phase_calib
                ph = torch.arange(T, device=device) % 4
                off = torch.where(ph == 2, float(b8), torch.where((ph == 1) | (ph == 3), float(b16), 0.0))
                ol = ol + off.unsqueeze(0)
            p = torch.sigmoid(onset_logit_scale * ol + onset_logit_bias)
            if onset_phase_alloc is not None:
                # phase-aware threshold: keep the budget N = (p>tau).sum() per row, split across the three
                # 16th-grid phase bands (quarter t%4==0, 8th t%4==2, 16th t%4 in {1,3}) by `onset_phase_alloc`
                # shares, picking the top-p_on frames WITHIN each band. Each band gets its own implicit
                # threshold -> the model's own 16th confidence wins 16th slots instead of losing globally to
                # the more-confident 8ths (which a single tau buries; see diag_phase_threshold.py).
                shares = torch.as_tensor(onset_phase_alloc, device=device, dtype=p.dtype)
                shares = shares / shares.sum()
                ph = torch.arange(T, device=device) % 4
                bands = [ph == 0, ph == 2, (ph == 1) | (ph == 3)]
                onset = torch.zeros(B, T, dtype=torch.bool, device=device)
                for b in range(B):
                    N = int((p[b] > onset_threshold).sum())
                    for share, band in zip(shares, bands):
                        idx = band.nonzero(as_tuple=True)[0]
                        nk = min(int(round(N * float(share))), int(idx.numel()))
                        if nk <= 0:
                            continue
                        top = idx[torch.topk(p[b, idx], nk).indices]
                        onset[b, top] = True
            else:
                onset = torch.bernoulli(p).bool() if onset_sample else (p > onset_threshold)

        # precompute which-panel bit table for the 15 patterns: (15,4)
        states_tab = torch.arange(1, NUM_PATTERNS + 1, device=device)
        panel_bits = ((states_tab.unsqueeze(-1) >> torch.arange(NUM_PANELS, device=device)) & 1)  # (15,4)

        caches = [_LayerCache(layer, memory) for layer in self.decoder.layers]
        cond_emb = self._cond(difficulty, radar, style, motif)  # (B,1,d) | (B,T,d) if motif is a per-frame schedule
        if do_cfg:  # parallel unconditioned (null radar + null style + null motif) path for guidance
            uncond_caches = [_LayerCache(layer, memory) for layer in self.decoder.layers]
            uncond_emb = self._cond(difficulty, None, None, None)  # (B,1,d)
        gen = torch.zeros(B, T, NUM_PANELS, dtype=torch.long, device=device)
        prev_emb = self.bos.expand(B, 1, -1)
        held = torch.zeros(B, NUM_PANELS, dtype=torch.bool, device=device)  # hold automaton state
        prev_pat = torch.full((B,), -1, dtype=torch.long, device=device)    # previous note's pattern (for rep penalty)
        next_foot = torch.zeros(B, dtype=torch.long, device=device)         # 0=left, 1=right (crossover automaton)
        if pattern_bias is not None:
            pattern_bias = torch.as_tensor(pattern_bias, dtype=torch.float32, device=device)  # (15,)
        n_panels = panel_bits.sum(1)                                        # (15,) panels per pattern
        # no_cross_during_hold state: while a hold pins one foot, the free foot handles other notes; a fast
        # cross to the OPPOSITE panel (L<->R, D<->U) is un-dance-able (the B4U "jacks with one foot during a
        # hold" — see notes/choreography_metrics_findings.md). Track the free foot's last panel + recency.
        OPP = torch.tensor([3, 2, 1, 0], device=device)                     # opposite panel: L<->R, D<->U
        single_panel = torch.where(n_panels == 1, panel_bits.float().argmax(1),
                                   torch.full((NUM_PATTERNS,), -1, device=device).long())  # (15,) panel if single else -1
        free_last = torch.full((B,), -1, dtype=torch.long, device=device)   # free foot's last panel (this hold)
        free_gap = torch.full((B,), 99, dtype=torch.long, device=device)    # frames since free foot's last note
        SINGLE_IDX = (1 << torch.arange(NUM_PANELS, device=device)) - 1      # single-panel pattern idx per panel
        # H13 exertion: track the running FAST (16th-adjacent) same-single-panel jack to cap it (max_jack_run).
        since_onset = torch.full((B,), 99, dtype=torch.long, device=device)  # frames since last onset (==1 -> 16th-adj)
        jack_panel = torch.full((B,), -1, dtype=torch.long, device=device)   # panel of the current fast same-panel run
        jack_len = torch.zeros(B, dtype=torch.long, device=device)           # length of that run

        reset_at = set(int(x) for x in boundary_reset) if boundary_reset is not None else set()
        for t in range(T):
            if t in reset_at:  # H11 boundary reset: drop note-history momentum (flush self-attn KV +
                prev_emb = self.bos.expand(B, 1, -1)                       # BOS) so the pattern head
                for c in caches:                                          # re-derives choreography from
                    c.k = None; c.v = None                                # audio (cross-attn mem_k/v kept)
                if do_cfg:
                    for c in uncond_caches:
                        c.k = None; c.v = None
            held_start = held                                              # hold state at frame start (hold_aware rebinds held later)
            pe_t = self.pos_encoding.pe[:, t:t + 1]
            cond_t = cond_emb[:, t:t + 1] if cond_emb.size(1) > 1 else cond_emb  # index the per-frame motif schedule
            h = self._decoder_step_cached(prev_emb + pe_t + cond_t, caches)[:, -1]
            pat_logits = self.pattern_head(h)                                  # (B,15)
            typ_logits = self.type_head(h).view(B, NUM_PANELS, NUM_TYPES)      # (B,4,4)
            if do_cfg:  # blend toward the radar-conditioned prediction (run the null path in lockstep)
                hu = self._decoder_step_cached(prev_emb + pe_t + uncond_emb, uncond_caches)[:, -1]
                pat_u = self.pattern_head(hu)
                typ_u = self.type_head(hu).view(B, NUM_PANELS, NUM_TYPES)
                pat_logits = pat_u + guidance_scale * (pat_logits - pat_u)
                typ_logits = typ_u + guidance_scale * (typ_logits - typ_u)
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
            if no_jump_during_hold:  # the pad has 2 feet: total occupancy (held + fresh presses) must not exceed 2,
                fresh_cnt = (panel_bits.unsqueeze(0).bool() & (~held).unsqueeze(1)).sum(-1)  # (B,15) fresh presses per pattern
                n_held = held.sum(1, keepdim=True)                           # (B,1) feet already pinned by open holds
                forbid = (n_held + fresh_cnt) > 2                            # more presses than free feet (incl. note-during-2holds)
                pat_logits = pat_logits.masked_fill(forbid, float("-inf"))
            if no_cross_during_hold:  # free foot fast-crossing panels while a hold pins the other foot (un-danceable)
                in_hold = (free_last >= 0) & held_start.any(1)
                g16 = in_hold & (free_gap <= 1)   # 16th gap (worst): forbid ALL different-panel singles (allow jack)
                g8 = in_hold & (free_gap == 2)    # 8th gap: forbid only the OPPOSITE single (dist-2 cross)
                SINGLE_IDX = (1 << torch.arange(NUM_PANELS, device=device)) - 1  # single-panel pattern idx per panel
                if g16.any():
                    rows = g16.nonzero(as_tuple=True)[0]
                    for p in range(NUM_PANELS):                                # forbid single panel p where p != free_last
                        bad = rows[free_last[rows] != p]
                        if len(bad):
                            pat_logits[bad, SINGLE_IDX[p]] = float("-inf")
                if g8.any():
                    rows = g8.nonzero(as_tuple=True)[0]
                    pat_logits[rows, (1 << OPP[free_last[rows]]) - 1] = float("-inf")
            if max_jack_run is not None:  # H13 exertion: forbid a FRESH single press that extends a fast jack
                at_cap = (since_onset == 1) & (jack_panel >= 0) & (jack_len >= max_jack_run)  # (B,)
                if at_cap.any():
                    fresh = panel_bits.unsqueeze(0).bool() & (~held).unsqueeze(1)             # (B,15,4) fresh per pattern
                    on_jack = fresh.gather(2, jack_panel.clamp(min=0).view(B, 1, 1)
                                           .expand(B, NUM_PATTERNS, 1)).squeeze(-1)           # (B,15) jack_panel pressed?
                    forbid = at_cap.unsqueeze(1) & (fresh.sum(-1) == 1) & on_jack             # fresh SINGLE on jack_panel
                    pat_logits = pat_logits.masked_fill(forbid, float("-inf"))               # (incl. {jack, hold-close} jumps)
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
                fresh_press = free_act                                        # panels FRESHLY pressed (what a foot hits)
            else:
                state = torch.where(active, typ + 1, torch.zeros_like(typ))   # stateless per-panel symbol
                fresh_press = active                                          # no holds -> every active panel is fresh

            gen[:, t] = state
            prev_emb = self._state_emb(state.unsqueeze(1))
            on = onset[:, t]
            prev_pat = torch.where(on, pat, prev_pat)  # track last note's pattern
            npc = n_panels[pat]                        # panels in the chosen pattern
            next_foot = torch.where(on & (npc == 1), 1 - next_foot, next_foot)   # alternate on singles
            next_foot = torch.where(on & (npc >= 2), torch.zeros_like(next_foot), next_foot)  # reset after jump
            # free-foot tracking for no_cross_during_hold: a single note placed while a hold was already open
            free_gap = free_gap + 1
            sp = single_panel[pat]                                              # (B,) panel if single pattern else -1
            is_free_single = on & (sp >= 0) & held_start.any(1)
            free_last = torch.where(is_free_single, sp, free_last)
            free_gap = torch.where(is_free_single, torch.zeros_like(free_gap), free_gap)
            no_hold = ~held.any(1)                                              # hold closed (end of frame) -> reset
            free_last = torch.where(no_hold, torch.full_like(free_last, -1), free_last)
            free_gap = torch.where(no_hold, torch.full_like(free_gap, 99), free_gap)
            # H13 jack tracking on FRESH single presses (NOT the pattern): a {tap, hold-close} jump reads as a
            # single in the chart, so counting the pattern leaks jacks -- count what a foot actually re-hits.
            nfresh = fresh_press.sum(1)                                         # (B,) fresh presses this frame
            fsp = torch.where(nfresh == 1, fresh_press.float().argmax(1),       # panel of the lone fresh press, else -1
                              torch.full((B,), -1, dtype=torch.long, device=device))
            is_fs = on & (nfresh == 1)                                          # exactly one fresh press = a single
            extend = is_fs & (since_onset == 1) & (jack_panel == fsp)           # reads start-of-frame since_onset
            jack_len = torch.where(is_fs, torch.where(extend, jack_len + 1, torch.ones_like(jack_len)), jack_len)
            jack_panel = torch.where(is_fs, fsp, jack_panel)
            reset = on & (nfresh != 1)                                          # jump (>=2 fresh) or pure hold-close breaks it
            jack_panel = torch.where(reset, torch.full_like(jack_panel, -1), jack_panel)
            jack_len = torch.where(reset, torch.zeros_like(jack_len), jack_len)
            since_onset = torch.where(on, torch.ones_like(since_onset), since_onset + 1)

        if lengths is not None:
            valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            gen = gen * valid.unsqueeze(-1)
        return gen
