# Data-augmentation roadmap

*2026-06-21.* On-the-fly augmentation ideas for the chart generator, ranked by value **for this project**.

## The domain constraint (why this is "mathy", not just image transforms)

Audio and chart are **tightly coupled** — the chart is choreographed to *that* audio on *that* BPM-aligned
16th grid. So you can't augment the input and keep the label (the image-classification pattern). Every
augmentation must either be **label-invariant** (change audio, chart still valid) or a **joint/equivariant
transform** (transform audio and chart together, or transform the chart along a symmetry that keeps it
musical for the same audio). Finding those symmetries is the whole game.

Two practical buckets, with very different infra cost:
- **Label-space augmentations** (mirror, crop): operate on the *cached* chart label at dataload — free,
  no GPU, no re-extraction, reuse the existing cache.
- **Audio-domain augmentations** (SpecAugment, pitch): change the audio *before* feature extraction, so
  they **invalidate the cache** → require on-the-fly GPU extraction (the "Rung 1" infrastructure).

We're data-limited (~3,800 train charts), so augmentation is a real anti-overfit / generalization lever,
and two of these map directly onto known weaknesses (panel bias, H5 structure).

## Ranked menu

### 1. Mirror / panel permutation — HIGH value, ~free  ⭐ (prototyping first)
Panels `[L,D,U,R]` (cols 0–3). The music decides *when* and *how many* notes; *which arrow* is a
partly-arbitrary spatial assignment, so a horizontal mirror **L↔R** = column permutation `[3,1,2,0]` on
the typed `states` gives an equally-musical chart for the identical audio (the DDR "Mirror" mod). Audio
untouched.
- **Equivariance, propagates for free:** the pattern-head target is *derived* from `states` via
  `panels_to_pattern(active)`, so permuting `states` auto-corrects the which-panels target. Onset/timing
  head is invariant. **Groove radar is invariant** (density/jumps/holds/chaos all unchanged) → no
  conditioning labels to recompute.
- **Benefits specific to this project:** (a) ~2× effective data (anti-overfit, we're data-limited);
  (b) directly attacks the **panel-bias "always-Left" degeneracy** — teaches that panel choice is
  arbitrary w.r.t. the music; (c) reuses `cache/samples_v3` — no re-extraction.
- **Implementation:** per sample per epoch, p=0.5, apply `[3,1,2,0]` to `states` in the batch builder
  (train only; val un-augmented for clean A/B). Start with **L↔R only** (highest confidence).
- **Extension:** the reflection group {identity, L↔R, U↔D, 180°} for ~4×. U↔D has a slightly different
  play-feel so keep it out of the first unambiguous test. **Exclude 90° rotations** — they turn
  horizontal patterns into vertical, changing ergonomics/crossovers (not feel-preserving).
- **Measure vs the H4 model:** panel balance (L/D/U/R) + jack rate (the targets), onset_F1/crit_adj
  (must not regress), taste-critic P(real), then playtest.

### 2. Random temporal crop — HIGH value, ~free
Training currently truncates to the first ~1024 frames (`typed_full[:T]`), so the model trains almost
entirely on **song openings** and rarely sees climaxes/outros.
- **Attacks H5 directly** (no global structure / density fades at the end is partly a *never-saw-the-end*
  problem) and gives far more varied musical context.
- **Implementation:** slice the cached arrays at a random start offset each epoch (also free, label-space).
- **Caveat:** the groove-radar label is computed per *full* chart, so a crop's local density drifts from
  its radar — minor, or recompute radar per crop.

### 3. SpecAugment / noise / EQ / reverb — MODERATE, needs on-the-fly audio
SpecAugment = masking random time + frequency bands of the spectrogram; plus additive noise, light
EQ/reverb. Audio-only, chart-invariant.
- **Benefit:** robustness + regularization. The corpus is heterogeneous community audio, so this pushes
  the model toward musical structure over mix/recording artifacts. Real anti-overfit value given the
  small dataset.
- **Cost:** changes audio pre-extraction → invalidates the cache → on-the-fly GPU extraction (Rung 1).

### 4. Pitch shift (phase vocoder: pitch without changing tempo/timing) — MODERATE, needs on-the-fly
Timing-driven choreography stays valid under a pitch shift → helps **chroma-dependent choreography (H1)**
generalize across keys.
- **Mathy catch:** chroma (pitch-class energy) *rotates* under a semitone shift — either shift by octaves
  (chroma-invariant) or rotate the chroma feature consistently. Audio-domain → needs on-the-fly.

### 5. Time-stretch / tempo perturbation — LOW value here (good insight)
Normally an audio-augmentation staple, but **the grid already bakes in tempo invariance**: everything is
in 16th-note frames (`hop = sr·60/(bpm·4)`), so the representation is tempo-normalized by construction.
Stretching tempo + re-gridding lands ~back where you started. The same fixed-BPM constraint that limits
us (see `constraint_relaxation_roadmap.md`) *gives* tempo invariance for free, so don't augment for it.

## Sequencing
1. **Mirror (L↔R)** — first prototype (cheap, reuses cache, attacks panel bias + doubles data). Queued
   behind the H4 high-res-onset retrain so it's a clean one-variable test on top of that model.
2. **Random crop** — second cheap label-space win; pairs naturally with H5 work.
3. **Audio-domain (SpecAugment, pitch)** — only when overfitting shows or for a larger corpus; this is
   what justifies building the on-the-fly GPU extraction path (Rung 1). Cross-ref the CPU/GPU
   preprocessing discussion: cache (Rung 0, current) → GPU dataloading (Rung 1) → custom CUDA (rare).

Cross-refs: `constraint_relaxation_roadmap.md` (fixed-BPM ↔ tempo invariance; data-layer v2),
`h4_offbeat_signal_findings.md`, `playtest_log.md` (panel-bias/always-Left, H5).
