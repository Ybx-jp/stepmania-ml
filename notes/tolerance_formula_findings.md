# Tolerance formula — the DEPLOYABLE predictor: audio strong-beat mass fraction

> **⚠️ CORRECTION 2026-07-04 (read the bottom section first — `## EXPANDED k=4 run + second-factor hunt`).**
> The expanded 32-song k=4 flip-point run **DOWNGRADED the formula**: `SB→g₀` fell from the small-n
> **Spearman +0.72 / R²=0.44** (below) to **+0.29 clean (n.s.) / +0.39 censored (p=0.027), R²~0.09**, and
> SB-only **LOO-CV R² ≈ 0**. The `R²=0.44`/`R²≈0.33` figures below are **small-n optimism** — SB is a real but
> WEAK single-factor rank trend, not a variance-explaining model. A disciplined second-factor hunt (84-dim
> fingerprint, permutation null) came back a **clean NEGATIVE** — no audio-poolable second factor. The **3/3
> prospective ear result still stands** (those songs sit on the clean SB spine). Treat the numbers below as
> superseded by the correction section.

**Question (the good-settings thread's core goal):** tolerance(song) = f(song features) — how far can a song be
cranked (CFG guidance, at the fixed milestone HIGH-chaos `--style` spec) before its 1/4 backbone collapses to a
1/16 smear? The n=40 sweep (`probe_backbone_tolerance.py`) found the only significant predictor was `real_density`
(ρ≈−0.37) — but that's the REFERENCE CHART's density, unavailable for an unseen song. This note reports the
**deployability check** (user-chosen 2026-07-04): does any AUDIO-DERIVABLE feature predict tolerance directly?

## Answer — YES, and it SUBSUMES density: `env_strongbeat_frac`
**`env_strongbeat_frac`** = fraction of the audio onset-envelope MASS (highres dim 13) that falls on strong-beat
frames (16th-frame index `t%4 ∈ {0,2}` = quarter/8th) vs the 16th-offbeats (`t%4 ∈ {1,3}`):
`Σ env[strong] / Σ env`. A pure **audio × phase-grid** feature — no reference chart, no model forward.

| tolerance metric | ρ(env_strongbeat_frac) | ρ(real_density) [prior lead] |
|---|---|---|
| ongrid_tol (backbone kept vs phase-shift) | **+0.636** (p<0.001) | −0.366 |
| anch_tol (16th coherent-runs vs smear)    | **+0.564** (p<0.001) | −0.364 |
| qrep_tol (strict downbeat coverage)       | **+0.632** (p<0.001) | −0.372 |

Raw first-order formula (OLS, n=40): `ongrid_tol ≈ 0.031 + 0.517·SB` · `anch_tol ≈ 0.097 + 0.699·SB` ·
`qrep_tol ≈ 0.030 + 0.645·SB`. **Single-predictor R² ≈ 0.33** (ongrid) — ~a third of tolerance variance from
AUDIO ALONE. Sign is mechanism-correct: HIGH strong-beat mass → the song's energy already sits on-grid → cranking
chaos (a global off-grid shift, H4) can't smear it → HIGH tolerance.

## Why this is the real driver, not density (confirmatory checks — all passed)
- **Robust, not an outlier/rank artifact:** Spearman ≈ Pearson (+0.64/+0.58); **leave-one-out over all 40 songs
  keeps ρ in [+0.61, +0.66]** (ongrid) — no single-song driver.
- **It SUBSUMES the density lead:** partial `strongbeat|density` stays **+0.60** (p<0.001); `density|strongbeat`
  collapses to −0.25 (n.s., p≈0.12). Adding density to the fit lifts R² only 0.34→0.37 with a small/ill-signed β.
  ⇒ the prior ρ=−0.37 density signal was largely a SHADOW of audio on-grid-ness. **The formula's core term is
  `env_strongbeat_frac`; density is redundant given it.**
- **BPM stays null** (the [[quality-feature-attribution]] top *quality* driver is NOT a *tolerance* driver — the
  two targets are distinct).

## Rule-8 artifact grounding (the convincing part)
Ranking the 40 songs by `env_strongbeat_frac`:
- **BOTTOM (predicted LOW tolerance):** OH WORLD 0.25, **Deja loin 0.28**, **High School Love 0.32**, KIM POSSIBLE
  0.36 — ongrid@g3.0 all ≈ 0.00 (collapse). **The two songs prior sessions caught smearing BY EAR — Deja loin (arc
  LOSS #2/#3, "vacates downbeat") and HSL (the chaos-gate smear song) — are at the BOTTOM of the audio predictor.**
  The feature independently flags exactly the ear-flagged failures.
- **TOP (predicted HIGH tolerance):** Our Soul 0.78, ONE TWO 0.70, BRILLIANT 2U 0.72 — hold the backbone at g3.0
  (ongrid 0.34–0.39).
- **taste_grid songs land where the inverted-U predicts:** Grand Chariot SB=0.52 (41st pct) and NIGHT IN MOTION
  0.58 (62nd) = MEDIUM strong-beat mass → medium tolerance = "great at gentle g=1.5, OVERLOADS at g=3.0" — exactly
  the by-ear referee verdict (`goodregion_findings.md`).

## Deployable rule of thumb (n=40, provisional)
`env_strongbeat_frac` range across the 40 Hard songs = 0.25–0.84 (mean 0.56).
- **SB < ~0.40** → LOW tolerance: keep guidance gentle (g≤1.5); expect a smear at g=3.0.
- **SB > ~0.65** → HIGH tolerance: can push guidance / chaos harder and keep the backbone.
- **SB 0.40–0.65** (incl. GC/NIM) → medium: the milestone chaos=0.9 lands well at g≈1.5, overloads by g≈3.0.
This operationalizes the referee's "crank chaos, keep guidance gentle" AS A PER-SONG budget: the lower the audio
strong-beat mass, the tighter the guidance ceiling.

## HONESTY / what would strengthen this (Rule 9 — not yet ear-refereed prospectively)
- The pre-registered hypothesis was the MODEL's `p_onset` strong-beat fraction — that came back NULL (−0.09). The
  winner is the RAW audio-envelope strong-beat fraction. The *mechanism* (strong-beat mass resists the smear) held;
  the operationalization that worked is the raw envelope (the learned p_onset washes out the raw on-grid structure).
  Not p-hacking (same mechanism, ρ=0.63 at p<0.001, LOO-stable) but report the env variant, not the p_onset one.
- **n=40, k=2 gens/song** → the tolerance LABELS carry sample noise ([[quality-feature-attribution]] ICC: a single
  gen ~46% noise), which ATTENUATES ρ — the true effect is likely STRONGER than 0.63. Scaling n and/or k would
  tighten R² and de-attenuate.
- **The offline tolerance metric is the ear-validated OVERLOAD DETECTOR** (`goodregion_findings.md`), so predicting
  it = predicting the overload cliff. But this predictor has NOT been PROSPECTIVELY ear-tested. **The binding next
  gate (Rule 8):** generate the milestone crank on a fresh predicted-LOW-SB song vs a predicted-HIGH-SB song and
  play them — does the low one smear and the high one hold? That closes the audio→ear loop prospectively.

## 2026-07-04 (cont.) — PREDICTING THE FLIP POINT g₀ (user: "which guidance flips a given song?")
Reframed the target from a scalar tolerance RANK to the FLIP GUIDANCE g₀ (the crank where 16th-anchoring falls off
a cliff). First, Rule-8 on the EXISTING n=40 curves: **32/40 songs are monotone anchoring CLIFFs** (user's intuition
confirmed). SB predicts g₀ (crossing) at Spearman +0.54 but only R²≈0.25 / resid≈0.58 guidance-units on the coarse
5-pt/k2 sweep. So ran a FOCUSED experiment (`probe_flip_point.py`): 16 songs spanning SB × DENSE 8-pt guidance grid
{1.0…3.0, dense in [1,2]} × k=4, fit a **logistic cliff** per song `anch(g)=floor+(plat−floor)/(1+exp((g−g₀)/w))`
(g₀=inflection=flip point; resolution-robust vs a noisy threshold crossing).

**Result — the flip point is predictable from audio (`cache/flip_point.csv`):**
- Fits are near-perfect (**r²=1.00** on most) → the cliff model is literally correct, not approximate.
- **SB → g₀: Spearman +0.72 (p=0.0035), R²=0.44, resid_std=0.28 guidance-units** — the denser grid + k4 denoising
  HALVED the residual (0.58→0.28) and lifted ρ (0.54→0.72), exactly the two attenuators (grid resolution, k-noise).
- **SB subsumes density here too** (partial SB|density +0.65; density alone −0.44; BPM null −0.25).
- **Formula: `g₀ ≈ 0.77 + 1.62·SB`** (in-sample OLS, n=14 clean fits) = a per-song GUIDANCE CAP good to ±0.28.
- **Cliff WIDTH w (sharpness) is an INDEPENDENT axis** — range 0.10–0.52, uncorrelated with SB (+0.14) or g₀ (−0.02).
  SB predicts WHERE the cliff is, not HOW SHARP (HSL w=0.10 abrupt vs 突撃 w=0.52 gradual). A separate feature-hunt.

**Honest caveats (Rule 9):** n=14 clean fits, IN-SAMPLE OLS → the ±0.28 band is optimistic (out-of-sample larger).
Real residual outliers: **LOVE** (SB=0.80 high but flips early g₀=1.79) and **BUMBLE BEE** (flips late +0.58) — SB is
~44% of the story. Censoring handled: flat-lo (never had a backbone) + flat-hi (never flips) excluded from the fit,
not fudged.

**✅ PROSPECTIVE EAR VALIDATION (2026-07-04, `notes/playtest_log.md`) — the binding gate CLEARED.** Generated 3 Hard
songs spanning SB (Heart Attack SB0.48, IN BETWEEN 0.61, Take It To The Morning Light 0.84), each at a guidance BELOW
and ABOVE its predicted g₀ (milestone chaos=0.9 spec, deployed decode; exported charts re-measured at the k=4 anchoring
means → faithful). **User played all 6 — CONFIRMED 3/3:** every SAFE (below-g₀) chart read coherent, every OVERLOAD
(above-g₀) read degraded; the flip-guidance ORDER held (Heart Attack overloaded by g=2.0 while Take It was fine at 2.0,
only "degraded, not ruined" at 3.0 → the g=2.0 same-guidance-opposite-verdict cross-check LANDED); Take It's graceful
degradation matched the high-SB "shallow cliff" prediction. **BONUS (inverted-U re-confirmed by ear):** Heart Attack
g=1.0 was MORE expressive than g=2.0 → past the flip more guidance is WORSE; the expressiveness peak is GENTLE-side of
g₀. So **g₀ is a per-song SAFETY CEILING; the recommended setting sits below it** (crank CHAOS, keep GUIDANCE gentle —
`goodregion_findings.md`). Caveat: n=3 by ear. This is the FIRST ear result on this thread that AGREED with the offline
metric (the prior 5 were misreads the ear overturned).

## 2026-07-04 (cont.) — SECOND-FACTOR HUNT + fit optimization: the ceiling is LABEL NOISE, not features
User: "chase the second factor SB misses, optimize the fit." Cheap analysis on the n=40 scalar-tolerance CSV,
judged by **LOO cross-validated R²** (in-sample R² just fits the k=2 label noise).
- **The SB-residual outliers look like DENSITY** (flip-later songs all low-density 0.28–0.36, flip-earlier all
  0.32–0.38; resid vs real_density ρ=−0.31 p=0.06, same direction as its original lead). **BUT density FAILS CV:**
  SB+real_density raises in-sample R² 0.33→0.37 yet LOWERS LOO-CV 0.260→0.243. bpm/onset_rate/env_abs_rate likewise
  flat-to-worse. The mechanistic candidate OVERFITS — the residual correlation was in-sample noise-fitting.
- **Only `d22_std` improved CV, and only inconsistently** — nested-CV (feature picked INSIDE each fold) selects it
  37/37 folds and HELPS ongrid_tol (+0.06) but HURTS anch_tol (−0.05). Metric-inconsistent + mechanistically opaque
  (a mid-block harmonic channel, chroma/spectral region; 7-dim accounting gap → exact channel uncertain). NOT trusted.
- **No better SB variant** (`probe_sb_variants.py`): the deployed env_frac (onset_env dim13) beats strong-beat
  CONTRAST, quarter-only combs, and — notably — the **highres_onset (dim34) strong-beat frac is NULL** (ρ≈−0.09,
  CV<0). The COARSE smoothed envelope is the right signal (it tracks sustained rhythmic WEIGHT, which chaos smears);
  the sharp-transient channel doesn't. The smoothing IS the feature.
- **VERDICT: label-noise-limited, not feature-limited.** SB sits at the CV ceiling (~0.26) for k=2/n=40 labels; every
  added feature overfits. The PROVEN lever is DENOISING: the flip-point g₀ run (k=4 + dense grid) already hit R²=0.44
  vs the coarse k=2 scalar's 0.25. → to optimize the fit, get CLEAN g₀ labels on MORE songs (higher k, more n), then
  re-hunt the second factor on labels clean enough for a real one to surface. (Feature engineering is tapped on n=40/k2.)

## 2026-07-04 (cont.) — EXPANDED k=4 run + second-factor hunt: FORMULA DOWNGRADED, no audio 2nd factor
The "optimize the fit" plan was: get CLEAN g₀ labels on MORE songs (the prior R²=0.44 rested on n=14 clean fits),
then re-hunt the second factor on labels clean enough for a real one to surface. Both ran. **Both cut against the
prior headline.**

**Expanded flip-point run (`probe_flip_point.py --n_build 60 --n_pick 32 --k 4` → `cache/flip_point_v2.csv`):**
32 songs spanning SB **0.07–0.84** × dense 8-pt guidance × k=4, logistic-cliff fit per song (fits still r²≈1.00).
- **SB → g₀ WEAKENED, it did NOT tighten** (the HANDOFF prediction "denser+k4 tightens" was WRONG):

  | fit | Spearman | Pearson R² | n |
  |---|---|---|---|
  | coarse 5-pt / k=2 (original) | +0.54 | ~0.25 | ~20 |
  | focused 8-pt / k=4, **n=14 clean** (the R²=0.44 headline) | **+0.72** | **0.44** | 14 |
  | **expanded 8-pt / k=4, clean flippers** | **+0.29 (p=0.13, n.s.)** | **0.088** | 28 |
  | expanded, **fallbacks kept as censored** | **+0.39 (p=0.027)** | — | 32 |

  ⇒ the +0.72/R²=0.44 was **small-n optimism** (14 songs). Adding songs regressed it toward the true, weaker value
  — the *expected* direction when a first estimate was lucky, not a bug. The formula `g₀≈0.77+1.62·SB` still
  brackets the flip in RANK, but SB explains ~9% of variance out-of-sample, not ~44%.
- **The reported n=28 number is pessimistically censored.** `probe_flip_point.py` drops 4 songs whose cliff never
  appears in-range (`fit_ok=0`); 3 of those (Abyss, Dead Heat, ONE TWO) are high-tolerance RESISTERS — dropping
  them flattens the slope. Kept as censored (rank-safe), Spearman rises to **+0.39, p=0.027** (still significant).
  **Report the censored number.**
- **The weakening is a HIGH-SB FORK.** Among SB>0.65 songs, some resist (Take It g₀2.34, BUMBLE 2.48, Abyss/ONE TWO
  never flip) and some flip EARLY (MEANING OF LIFE g₀1.05, And Then We Kiss 1.40, LOVE 1.79). SB can't separate them.
- **The 3 ear-tested songs still land on-formula** in the clean data (Heart Attack v2 g₀1.28 vs formula 1.55; IN
  BETWEEN 1.94 vs 1.76; Take It 2.34 vs 2.12) — they sampled the clean SB spine, so the 3/3 ear result is intact;
  it just wasn't a fair sample of the scattered high-SB fork.

**Second-factor hunt (`probe_flip_secondfactor.py`, judged by LOO-CV increment + PERMUTATION NULL):** the same
84-dim pooled fingerprint (mean|std of the 42 highres dims, recomputed per song so titles align exactly), asking
whether any dim explains the g₀ residual after SB. Disciplined for n=28 vs 84 candidates.
- **CLEAN NEGATIVE.** Best-of-84 LOO-CV increment over SB-only = **+0.267** (clean) / +0.129 (censored) — BELOW the
  permutation-null 95th pct (**+0.387** / +0.341); **p=0.23 / 0.60 = within chance.** No reliable second factor.
- **The negative control validated the harness (Rule 11).** `real_density` — the known overfitter — raised in-sample
  R² (+0.106) but LOWERED LOO-CV (−0.076), exactly as on the n=40 scalar. The null test can tell signal from overfit.
- **SB barely survives out-of-sample even alone:** SB-only LOO-CV R² = **−0.055** (clean) / +0.016 (censored) — once
  SB can't refit on the held-out song it's ≈ the mean. The +0.29/+0.39 is a weak rank trend, not a predictive fit.

**VERDICT (Rule 9, stated conditionally): the flip point is SB-bracketed but essentially SINGLE-FACTOR and WEAK; the
high-SB fork is NOT audio-poolable at n=32.** This is consistent with the thread's own standing result that off-beat
placement is not audio-reachable (the parked chaos×onset gate: the discriminating signal lives in NOTE-CONTEXT, not
in any pooling of the audio). What would change it: (a) many more songs (n=32 is underpowered for a subtle 2nd
factor); (b) a NON-pooled, phase-structured candidate (SB itself is phase-structured — a mean/std fingerprint may be
the wrong feature family); or (c) the note-context gate, which is where the fork most plausibly lives. **Fit
optimization by adding audio features is TAPPED** — the same conclusion as the n=40 scalar hunt, now on clean labels.

## Tooling
`probe_tolerance_audio_density.py` (base p_onset + env strong-beat/occupancy features; merges with the tolerance
CSV) → `cache/tolerance_audio_density.csv`. `probe_flip_point.py` (dense guidance sweep + logistic-cliff fit →
g₀) → `cache/flip_point.csv` + `cache/flip_point.log`; the expanded 32-song version →
`cache/flip_point_v2.csv`. `probe_flip_secondfactor.py` (the LOO-CV increment + permutation-null second-factor hunt
on the v2 labels vs the 84-dim pooled fingerprint). Confirmatory partials/LOO/OLS inline. Depends on
`cache/backbone_tolerance.csv` (the n=40 label sweep). Lineage `experiment_lineage/good-settings-region-arc.md`;
memory [[good-settings-region]]; parent `goodregion_findings.md` (the referee) + `real_phase_reference_findings.md`
(the real envelope the metric measures distance from).
