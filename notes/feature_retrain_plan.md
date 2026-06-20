# Feature Retrain Plan — give the model musical content

*2026-06-19. Motivated by the playtest findings (see `playtest_log.md`): timing is solved, but
choreography is musically blind. Three converging lines of evidence point at the audio features.*

## The problem, precisely

The 23-dim audio features are **timbre + energy only**: 13 MFCC + onset_env + onset_rate + tempo +
7 spectral-contrast. There is **no melody/harmony (no chroma/pitch) and no source separation
(drums vs tuned)**, and the audio encoder is a shallow 2-block Conv1D (receptive field ~5 frames, no
global/structural view). Consequences measured in playtest:

- **H1 (local choreography):** arrow choice can't track musical events → feels arbitrary.
- **H4/chaos:** the model can only express "syncopation" as a *uniform grid shift* (66% 8th-offbeat
  at g=1.3, or a 6%-on-beat smear at g=2.0) because it can't see *which* offbeats hold a musical hit.
  Base even under-syncopates vs real (0.91 vs 0.80 on-beat).
- **H5 (structure):** generated density is flat and fades at the outro while real charts build to a
  climax; the constant-energy probe proved this is the model faithfully tracking audio energy, not a
  decode artifact — it has no notion of song *sections*.

## What to add

| feature | dims | addresses | how (librosa, chart-aligned hop) |
|---|---|---|---|
| **chroma** (chromagram = per-pitch-class energy) | 12 | H1 melody/harmony | `chroma_cqt` at `chart.hop_length` |
| **HPSS onsets** (harmonic/percussive source separation) | 2 | H1 drums-vs-melody mapping | `effects.hpss` → onset strength on each |
| **metric phase** (position in beat + in measure, as sin/cos) | 4 | metric-aware syncopation | from the 16th-grid index (free, no audio) |
| **structural novelty** (self-similarity boundary curve) | 1 | H5 sections/climax | `librosa.segment` novelty / recurrence |

Current 23 → ~42 dims. (Metric phase is nearly free and a cheap H5/syncopation aid; structural
novelty is the more experimental H5 piece.)

Optional architecture change (separable, do AFTER features prove out): **widen the audio encoder's
receptive field** (dilated convs / more layers, or a downsampled global-context branch) so the
decoder can see beyond ~5 frames — the other half of H5.

## Staging — one change at a time, measured

Each stage = a retrain (warm-start everything except the audio encoder's first conv, whose input dim
changes) + eval. Stop/keep by the metrics below + a playtest.

- **Stage 1 — chroma + HPSS** (the local-musicality bundle; the most-felt gap, H1). One change vs the
  current timbre-only baseline. If it helps, optionally ablate chroma-only vs +HPSS to attribute.
- **Stage 2 — metric phase** (cheap; should sharpen on/off-beat placement so chaos becomes
  metric-aware rather than a uniform smear).
- **Stage 3 — structural novelty + (optional) wider encoder** (H5: section/climax/start-end).

## How we'll measure it (beyond onset_F1, which is blind to all this)

New quantitative proxies, generated vs real (these are the whole point — the existing metrics can't
see musicality):

1. **Within-beat phase KL/Δ** — does the generated note-phase distribution match real? Target: close
   the base gap (gen 0.91 on-beat vs real 0.80) and make the *chaos knob* shift phase in a way that
   **correlates with audio events**, not uniformly.
2. **Structure correlation** — corr(generated density-vs-position, real density-vs-position) and
   vs an audio-novelty curve. Target: recover the intro→build→climax→outro arc (fix the end-fade).
3. **Choreography↔event alignment** — do arrow *changes* coincide with chroma/HPSS changes more than
   chance? A direct H1 readout.
4. **Playtest** (the ground truth): re-run the radar toggle sets; does **chaos** finally feel like
   *musical syncopation*, and does the chart track phrase changes? Plus the standing onset_F1/crit_adj
   as regression guards (must not drop).

## Cost / mechanics

- Feature extraction changes → **cache invalidation**: new `cache_dir` (e.g. `cache/samples_v2`),
  re-extract (the slow part; chroma/HPSS/segment are heavier than MFCC).
- `audio_dim` changes → audio-encoder first layer re-inits; rest warm-starts from `gen_style`/`gen_layered`.
- Config: extend `AudioFeatureConfig` with toggles per feature so stages are flag-controlled and
  ablatable. Keep the old 23-dim path working (don't break existing checkpoints/tests).
- RTX 3060: same batch 8 / 1024-cap regime; the `autotune` skill can re-tune if the wider feature
  vector or encoder changes the memory profile.

## Decision (2026-06-19)

**Stage 1 = chroma + HPSS + metric phase (23 → 41 dims).** User chose the broadest local-musicality
bundle: it targets H1 (chroma + HPSS) and the chaos uniform-offbeat-smear (metric phase) in one retrain.
Stages 2–3 (structural novelty, wider encoder) follow after Stage 1 is measured.

## Stage 1 implementation checklist

1. **`AudioFeatureConfig` + extractor** (`src/data/audio_features.py`): add per-feature toggles
   `use_chroma`, `use_hpss_onsets`, `use_metric_phase`; extract at the chart-aligned `hop_length`:
   - chroma: `librosa.feature.chroma_cqt(y, sr, hop_length=...)` → 12 dims, normalized.
   - HPSS: `librosa.effects.hpss(y)` → onset strength on each of harmonic/percussive → 2 dims.
   - metric phase: from the 16th-grid frame index — beat phase `2π·(t%4)/4` and measure phase
     `2π·(t%16)/16`, each as (sin, cos) → 4 dims. (No audio; derived from the BPM-aligned grid.)
   - append in `get_aligned_features()` after the existing 23; reuse `_align_features` to the chart length.
2. **Cache v2**: new `cache_dir='cache/samples_v2'` so old MFCC caches aren't reused; re-warm (slow part).
3. **Model**: `audio_dim=41`; warm-start everything except `audio_encoder` first conv (input dim
   changed) — extend `load_from_factorized` to skip shape-mismatched audio-encoder input layer cleanly.
4. **Train**: `train_factorized`/`train_layered`-style run from the `gen_layered`/`gen_style` warm-start;
   keep batch 8 / 1024-cap; `autotune` if memory shifts.
5. **Eval** (the new metrics): within-beat phase Δ vs real, structure correlation, arrow-change↔
   chroma-change alignment; regression-guard onset_F1/crit_adj; then a playtest of the radar sets
   (does chaos become musical?).
6. **Tests**: feature-extractor unit test (shape 41, deterministic, chart-aligned); keep the 23-dim
   path green (old checkpoints/tests must still load).
