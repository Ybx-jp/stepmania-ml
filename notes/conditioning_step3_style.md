# Conditioning Step 3: reference-chart style embedding

*2026-06-19. The last requested knob: "make it feel like this chart."*

Steps 1–2b give scalar/categorical knobs (pattern prefs, a 5-dim groove radar). Step 3 adds a
*reference chart* as the control signal: encode an existing chart into a latent and condition on it,
so the model transfers that chart's feel onto new audio.

## How

- **StyleEncoder** (`typed_model.py`): reuses the Phase 1 `ChartEncoder` (embedding + Conv1D blocks)
  to map a reference chart's per-frame occupancy `(B,L,4)` → `(B,L,d)`, then **masked-mean-pools over
  time** to a single `(B,d)` style latent. The pool is the bottleneck — it can carry global feel
  (density, which-panels / jack tendencies) but not the exact note sequence, so it *transfers* style
  instead of copying. Padded frames are zeroed before encoding (the conv mixes neighbours, so pad
  garbage would otherwise leak across the mask boundary).
- **Conditioning**: `_cond(difficulty, radar, style)` adds the style latent (or a learned
  `null_style` for CFG dropout) to the conditioning vector, exactly mirroring `radar`/`null_radar`.
  Style and radar are independent additive knobs — use either, both, or neither.
- **Training** (`train_style.py`): reference = the *target chart itself* (autoencoder-style
  conditioning; the bottleneck stops trivial copying). Warm-starts from the radar checkpoint, keeps
  radar as a knob, and drops radar and style **independently** for CFG so each can be amplified alone.
- **CFG at inference**: `generate(reference=, reference_mask=, guidance_scale=g)` — the existing dual
  KV-cache now nulls *both* radar and style on the unconditioned path, so `g>1` amplifies style too.

## Result

15-epoch run warm-started from the radar checkpoint (best val epoch 13). Self-reference sanity:
onset_F1 0.750, density 0.198 (real 0.217).

Cross-song transfer (`eval_style_transfer.py`, 24 val songs): same audio conditioned on a SPARSE
reference (density 0.023) vs a DENSE reference (density 0.461), measuring generated density:

| guidance | density sparse-ref → dense-ref | gap |
|---|---|---|
| g=1 | 0.182 → 0.373 | 0.19 |
| g=2 | 0.126 → 0.488 | 0.36 |
| g=3 | 0.098 → 0.565 | **0.47** |

The reference chart's feel transfers onto *different* audio: a sparse reference pulls density down,
a dense one pulls it up, and CFG widens the gap monotonically (~2.5× from g=1 to g=3). Jump rate is
roughly flat here because the archetypes were selected by density, not jumps — density is the clean
readout for this pair.

## Cost / notes

The style latent is encoded once per generation (cheap, non-AR). CFG on top is the same ~2× per-step
cost as Step 2b when `guidance_scale != 1`; default `g=1` and `reference=None` → no overhead.

Conditioning roadmap complete: Step 1 (pattern prefs) ✓, Step 2 (radar) ✓, Step 2b (CFG) ✓,
Step 3 (reference-chart style) ✓.

Code: `StyleEncoder` + `encode_style` + `reference=`/`style=` plumbing in `LayeredTypedChartGenerator`;
`train_style.py`, `eval_style_transfer.py`.
