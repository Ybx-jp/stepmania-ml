# Groove / rhythmic-periodicity metric — findings

*2026-06-21. `experiments/realism_critic/groove_periodicity.py`.* The metric we'd been missing for
chaos/syncopation: on-beat% measures syncopation QUANTITY; this measures STRUCTURE (H10 — a groove is
REPEATED off-beat figures). Demeaned onset autocorrelation at musical lags; the key signal is **ac_off**
= autocorr at lag 4 (one beat) of the OFF-beat-only onset signal = "do off-beat notes recur at the same
beat-phase across beats?" (groove >> scatter).

## Validation + baseline (40 val songs, recommended decode)

| source | on-beat% | ac_beat (lag4) | ac_meas (lag16) | **ac_off (lag4, off-beat)** |
|---|---|---|---|---|
| REAL | 87.7% | 0.367 | 0.725 | **0.187** |
| REAL-shuffled null | 87.7% | 0.292 | 0.658 | **0.012** |
| gen_stage1 (plain) | 93.3% | 0.341 | 0.862 | **0.109** |

The shuffled null relocates off-beat onsets to random off-beat frames (same density + on-beat%, periodicity
destroyed).

## Findings
1. **METRIC VALIDATED.** REAL ac_off 0.187 vs null 0.012 = ~15× at IDENTICAL density/on-beat% → ac_off
   captures groove structure, not density. The null collapses to ~0 as designed. (ac_beat is a weaker
   discriminator — null only drops to 0.292 because the on-beat backbone alone is beat-periodic; ac_off
   is the clean one.)
2. **Base generator UNDER-GROOVES but isn't pure scatter:** ac_off 0.109 = ~58% of the way from null
   (0.012) to real (0.187). The model places SOME periodic off-beats, short of real.
3. **Two stacked deficits:** under-syncopates (93.3% on-beat vs real 87.7% = too few off-beats) AND the
   off-beats are less periodic (0.109 vs 0.187).
4. **Over-regular globally, under-grooved locally:** ac_meas 0.862 (gen) > 0.725 (real) — the generator
   is MORE measure-periodic than real (too repetitive across measures; real has fills/variation), while
   being under-grooved at the off-beat scale. Distinct from H5 (density arc) — this is rhythmic-figure
   repetition.

## Use
- **Target to close:** gen_stage1 ac_off 0.109 → real 0.187 (null floor 0.012). This is the number the
  expert-data + numeric-difficulty retrain must move (`constraint_relaxation_roadmap.md`,
  `numeric_difficulty_conditioning_plan.md`).
- **Next cheap confirmation:** run the metric on the chaos-gate output (`chaos_gate.py` gated/smear). H10
  predicts the gate's "felt arbitrary" scatter has ac_off ≈ null despite landing on audio events — which
  would make the metric AGREE with the playtest hands and fully confirm H10.

## Localization (2026-06-21, `groove_localize.py`, 40 songs)

ac_off by decode stage: **1 REAL 0.187 → 2 p_on continuous 0.940 → 3 raw threshold 0.109 → 4 full decode 0.109.**
- **(3)=(4): the pattern/type/hold-aware AR decode is INNOCENT.** Groove is already gone at raw onset
  placement, not lost in the decode we kept suspecting.
- **The continuous-posterior 0.940 is a TRAP (likely a smoothness artifact):** ac_off is lag-4 (short), and
  p_on is smooth, so off-beat values 4 frames apart are similar from smoothness, not beat-periodicity. Do
  NOT read 0.940 as "groove is in the posterior, ready to recover."
- **Localized to: onset head + density threshold.** Open fork — (a) threshold decode discards usable
  structure (cheap fix), or (b) the non-causal audio-only onset head doesn't rank groove off-beats (H4
  weak-audio, ~architectural). Decided by the periodicity-aware decode test (below).

## DECODE-vs-ARCHITECTURE test (2026-06-21, `groove_periodic_decode.py`, 40 songs, off-beat count matched to real)

ac_off (subset filtered to n_off≥4, so REAL here is 0.300, a more-syncopated subset):

| placement | ac_off |
|---|---|
| REAL | 0.300 |
| **(a) threshold = top off-beats by `p_on`** | **0.318** |
| (b) measure-template (flawed) | 0.055 |
| null (random off-beats) | 0.021 |

**ANSWER: NOT architectural — it's a DECODE under-placement of off-beats (cheap fix).** Given an adequate
off-beat BUDGET, selecting top off-beats by the model's own `p_on` yields **real-level groove (0.318 ≈
0.300, 15× null)**. The onset head DOES carry groove. The full-generation deficit (0.109) is because the
single global density threshold keeps mostly on-beats (on-beat `p_on` ≫ off-beat `p_on` → the model
decodes *more* on-beat-biased than real, 93% vs 88%) — too few off-beats get through.
- *(b) was a bad probe: it enforced per-MEASURE repetition (lag-16) while ac_off is lag-4 (beat); ignore.*
- **Connects to the chaos gate:** the gate felt "arbitrary" because it picked off-beat accents by **dim41
  audio saliency** (weak, 0.66); picking by **`p_on`** instead gives real-level periodicity. The off-beat
  signal the model needs is its OWN posterior, not the raw audio onset.

## THE FIX (cheap, no retrain): phase-aware off-beat-budget decode
Replace the single global onset threshold with a per-phase budget: allocate a target off-beat fraction
(≈ real's, or a `chaos` dial), select the top on-beat frames AND the top off-beat frames by `p_on`
separately. Recovers real-level ac_off offline. Build → measure ac_off + critic P(real) → playtest (does
real-level periodicity FEEL musical? — the gate's dim41 version didn't, but this uses p_on).
**Reorders strategy: groove is decode-fixable; the expert-data rebuild + numeric difficulty drop in
priority.**

## Budget-decode A/B (2026-06-21, `groove_decode.py`, 8 songs) — METRICS CONFLICT, playtest needed

normal (global threshold) vs budget (top on-beat + top off-beat by p_on, off-beat count = real):

| | real | normal | budget |
|---|---|---|---|
| ac_off (groove) | 0.240 | 0.035 | 0.139 |
| P(real) (taste critic) | 0.698 | 0.599 | 0.410 |

**Budget RAISED periodicity (ac_off, toward real) but LOWERED taste (critic), hardest on the syncopated
songs (Reach 0.974→0.008, 大和撫子 0.374→0.004).** The two validated-ish metrics disagree:
- ac_off is blind to musicality (established) — periodicity ≠ groove-that-feels-good.
- the critic may be ON-BEAT-BIASED here: it was trained on the under-syncopated filtered data, and it
  prefers the tame chart even when the REAL chart is groovy (Reach) → suspect it penalizes off-beats per
  se on this axis, so not a trustworthy judge of syncopation quality.
- Echoes the chaos-gate hands ("arbitrary") — but that used dim41; this uses p_on. **Unresolved offline.**
Exported `~/sm-generated/groove_decode_{budget,normal}`; playtest the songs where they differ (Deja loin,
大和撫子, Reach). If budget feels groovy-and-musical → decode fix stands + the critic needs expert-aware
retraining to judge syncopation. If busier/arbitrary → periodicity is still not musicality; groove may
genuinely need the rhythmic-structure modeling (H10) after all.
