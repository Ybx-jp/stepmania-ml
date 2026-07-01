# Which audio features drive per-song QUALITY under canonical defaults? — findings

*2026-07-01. Probe: `probe_quality_features.py` (repo root). Question (user): "which audio features influence the
variation in quality among songs applying the canonical defaults." Quality = taste/realism critic
`checkpoints/realism_critic`; per-song target = the **generator deficit** `P(real)_human − P(real)_gen` (user's
choice), generation replicating the DEPLOYED canonical decode via `CANONICAL_DECODE` + `decode_harness`
(NOT `eval_taste_current.py`'s stale pattern_temp=0.7 / no-16th-unlock block). Features = 42-dim highres input
aggregates (mean/std) + raw timbre/harmony descriptors recomputed from audio (spectral centroid, flatness,
chroma entropy, rms) — the raw set recovers the cross-song timbre/harmony that per-song z-scoring in the cache erases.*

## VERDICT (two levels)
1. **The only axis on which generator quality genuinely varies is chart DIFFICULTY / note DENSITY, not a subtle
   audio feature.** Across difficulties (n=8 smoke): `Spearman(difficulty, deficit)=+0.90`; the generator scores
   **near-human on Beginner** (deficit −0.22, i.e. it BEATS the human chart) and is **railed to "fake" on
   Medium/Hard** (deficit +0.71 / +0.77). `real_density` rides along (r=+0.86) because harder = denser. Every
   audio feature that merely proxies difficulty (density, onset-rate, tempo) inherits a spurious deficit
   correlation — the classic Rule-12 pooling trap.
2. **WITHIN a fixed difficulty (Hard, n=48), NO audio feature explains generator-quality variation above the
   multiple-testing noise floor.** Best `|Spearman(feature, m_gen)| = 0.342`, but the FAMILY-WISE permutation
   null of the max over 99 features has mean 0.370 / 95th pct 0.472 → **family-wise p = 0.67**. The top-ranked
   features (raw_rms_mean −0.34, MFCC/chroma dims ~0.3) are indistinguishable from chance. Neither the 42-dim
   aggregates nor the raw timbre/harmony descriptors carry within-Hard quality signal.

## WHY within-Hard is null: the taste critic SATURATES on canonical Hard generations (the binding finding)
- **47/48 (94%) of canonical Hard generations sit on the critic's LOW rail** (P_gen mean 0.036, only 1 escaped
  >0.1 — "LOVE" at 0.93, a fluke). "gen scores in the discriminating band 0.1–0.9: **0%** (k=3) / 6% (k=1)."
- Therefore the **deficit is 91% correlated with P_real and 81% of its variance is the HUMAN chart's score** —
  the user's chosen target measures "which human reference charts the critic likes," NOT where the generator
  succeeds. (This is a Rule-11 dynamic-range failure at the level that matters: the *deficit* varies, but its
  variance is in the wrong term.)
- **The logit-MARGIN rescue does not help a rank metric.** `P(real)=sigmoid(logit_real−logit_fake)` is a strictly
  monotonic transform of the margin ⇒ identical Spearman ranks ⇒ identical feature correlations (verified: the two
  ranking tables are byte-identical). The margin confirms the saturation is GENUINE (not float-precision ties),
  not that hidden signal exists. m_gen sd=1.6 in logit space gives *some* ordering, but it's the same order as the
  saturated probability and is not audio-feature-explained (the permutation test above).

## Interpretation
The taste/realism critic, at fixed difficulty, is **not a usable per-song quality instrument** for this question:
it rates essentially all canonical same-difficulty generations as uniformly "fake," so there is barely any
generator-quality variation for any audio feature to explain. This corroborates and SHARPENS
`taste_critic_transfer_findings.md`'s "near-binary separator, not a graded scorer" — on Hard specifically it is
~100% railed (0% mid-band), worse than the pooled 14–30% mid-band reported there. Attribution (experiment-design
HARNESS→DATA→MODEL): the null is a **measurement-instrument limitation**, not a demonstrated absence of
audio-feature effects on quality.

## What would actually answer the question (untested; state-what-would-change-the-conclusion)
A quality signal with **dynamic range among same-difficulty generations** is the prerequisite:
- **By-ear ratings** (the project's binding gate) as the target — but tiny N.
- A **recalibrated / low-end-spread critic** (temperature-rescale; the exact need `taste_critic_transfer` flagged
  for best-of-N), or a non-saturating proxy (distance-to-real-distribution, choreography metrics).
- Re-ask ACROSS difficulty with difficulty **partialled out** — but the smoke suggests P_gen variance ≈ difficulty,
  so partialling likely returns the same railed residual.

## Method notes / reproduction
- `probe_quality_features.py --data_dir data/ --audio_dir data/ --difficulty 3 --n 48 --k 1` (within-Hard).
  Drop `--difficulty` for the pooled/across-difficulty view. CSVs: `cache/quality_features_hard{,_margin}.csv`,
  `cache/quality_features_smoke.csv`.
- **k=3 → k=1 is safe here:** within-song sampling sd of P_gen = 0.016 (the near-binary critic is stable per song).
- Gotchas fixed while building (both cost a run): (1) `val_ds.warm_cache()` eagerly extracts the WHOLE val set
  (~30 min CPU) — dropped; lazy per-song `val_ds[i]` extraction, safe since we index the full val_ds in order
  (no subset/--match cache aliasing, [[dataset-cache-footgun]]). (2) `librosa.feature.chroma_stft` SEGFAULTS via
  `estimate_tuning`→`piptrack` numba gufunc — pass `tuning=0.0` (matches `audio_features.py`'s own call).
- Canonical-decode fidelity confirmed: generation used `gen_motif_full_fixed` (42-dim), pattern_temp 1.0, fatigue
  2/free 6, stamina 50/breathe 1.2, the (0,1.0) 16th-unlock, tau from the conditioned+phase-calibrated onset
  logits, playability forced — via `CANONICAL_DECODE`/`decode_harness`, per `generation-defaults`.

## FOLLOW-UP — choreography distance-to-real as a NON-SATURATING quality proxy (2026-07-01)
*Probe `probe_quality_choreo.py` (Hard, n=64). Swaps the saturated critic for a GRADED choreography
distance-to-real from the validated battery (`choreography_metrics_findings.md`): trans_KL (panel-transition
matrix KL to POOLED real Hard), holdburst_excess (the fast one-foot-cross-during-hold rate — the ONE metric shown
to PREDICT a play-feel complaint, B4U), panel_TV, + a z-summed composite. Generation via the shared canonical
TYPED helpers in `probe_quality_features.py`.*

**Instrument = SUCCESS (the durable methodological win).** Unlike the critic, the proxy is NON-saturating and
VALIDATED: composite sd 1.77 (range −3.3→+4.1); it DISCRIMINATES gen from real on trans_KL (gen 0.158 > real 0.120)
and holdburst (gen 0.064 > real ~0), reproducing "generator footwork ≈ random-shuffle + extra one-foot-during-hold
bursts." (panel_TV did NOT discriminate — gen 0.050 = real 0.050 — correctly dropped.) **Choreography distance-to-
real is the right quality instrument for this question; use it over the critic at fixed difficulty.**

**Substantive answer = still essentially NULL, with ONE marginal lead.**
- `choreo_composite` and `trans_KL`: family-wise permutation p_fw = 0.31 / 0.36 → **noise floor** (no audio feature).
- `holdburst_excess`: best |r| = 0.437 (d31_std, a chroma-dim time-variability), family-wise p_fw = **0.027 → SIGNAL**
  — BUT: (a) I tested 3 targets, so ≈0.08 after that correction — MARGINAL; (b) the signal lives ENTIRELY in the
  `d##_std` block (z-scored spectral/chroma std over the truncated window — a truncation-sensitive, murky quantity);
  the INTERPRETABLE descriptors (bpm, density, perc/harm, centroid/flatness/rms/chroma-entropy) are FLAT
  (family-wise p_fw = 0.51, best = real_density +0.25). The hit is robust to density (partial = −0.41) and not
  outlier-driven (−0.31 after dropping top-10% holdburst), but is not a clean audio property.
- Direction (if real): more (windowed) spectral/chroma time-variability → LESS hold-burst defect. Speculative.

**Verdict:** no INTERPRETABLE audio feature drives within-Hard generator quality (choreography OR critic axis) above
the family-wise noise floor. The lone marginal lead is audio spectral/chroma DYNAMICS vs the hold-burst defect —
a weak hypothesis, not a finding.

**"Recalibrated critic" (the planned fallback) is a DEAD END as a rescale.** Temperature/Platt scaling is MONOTONIC
→ identical Spearman ranks → identical feature correlations (same invariance already shown for the logit margin,
§ above). Only a RETRAINED graded critic (ranking loss / graded-corruption targets) could differ — a real training
effort, and choreography (a validated orthogonal quality axis) already returned near-null, so expect the same.

### The hold-burst lead — REFUTED by a pre-registered clean test (`probe_holdburst_dynamics.py`, n=64)
The one family-wise-significant hit above (holdburst_excess vs the `d##_std` block, best r=−0.44) was tested
properly: ONE target (the already-computed `g_holdburst_excess`, reused — no regeneration), ONE directional
hypothesis ("audio spectral/chroma/timbral DYNAMICS → NEGATIVE → fewer awkward held-note bursts"), a SMALL
pre-specified interpretable feature set (spectral flux, chroma flux, spectral-contrast temporal std, MFCC flux,
centroid std) recomputed from RAW audio on the **first-T-frames window the generator actually saw** (de-artifacts
the z-score/truncation mismatch that made `d##_std` murky).
- **Primary composite dynamics index: r = +0.059, one-sided (negative) permutation p = 0.685 → does NOT support H**
  (not even the hypothesized sign). Family-wise over the 6 tests: best |r| = 0.219 (speccontrast_std, WRONG sign
  vs the artifact), p_fw = 0.29 → not significant.
- **Conclusion: the d##_std signal was a z-score/TRUNCATION ARTIFACT, not a real audio→choreography relationship.**
  Cleanly measured, spectral/chroma dynamics do NOT predict the hold-burst defect. The lone lead is dead.

## FOLLOW-UP 2 — the GRADED critic retrain (the "recalibrated critic" done right; 2026-07-01)
*User-directed after the choreography near-null. `experiments/realism_critic/train_graded_critic.py` → checkpoint
`checkpoints/realism_critic_graded/best_val.pt`; re-attribution `cache/quality_features_hard_graded.csv`.*

**WHY a retrain (not a rescale):** the deployed critic saturates because it's trained with BINARY cross-entropy on
SEVERE corrupted-real negatives (full panel-scramble / shift = 0) vs real = 1 — the objective bakes in a saturated
boundary. A temperature/Platt rescale is monotonic → identical ranks. The fix must change the OBJECTIVE.

**The graded objective (keeps the v2 anti-fingerprint win — NO generator in training):** GRADED corrupted-real
ladders (panel-scramble FRACTION {0,.2,.45,.7,1.0} + audio-shift {0,2,6,16}) + a WITHIN-SONG MARGIN-RANKING loss
(score must decrease monotonically along each song's ladder; within-song pairs hold density/timing/audio fixed =
the taste isolation, now graded) + a light end-anchor BCE. Score = the logit margin; warm-started from the binary
critic.

**Instrument = SUCCESS.** Trained ladder is cleanly graded (real +1.99 → .2 +0.29 → .45 −0.95 → .7 −1.63 →
1.0 −2.06, pooled monotone). On canonical Hard GENERATIONS the saturation is FIXED: **gen PROB in the 0.1–0.9 band
0% → 44%**, `m_gen` sd 0.75 over [−3.55, +0.32]. A non-saturating realism critic now exists (reusable asset).
CAVEAT: per-song ladder monotonicity only ~0.35 → single-chart scores are NOISY; cross-critic agreement (graded vs
binary `m_gen`) is only +0.32.

**Substantive answer = STILL NULL.** Family-wise permutation over 99 features: best |r|=0.455 → **p_fw=0.118**;
interpretable-only best (real_density −0.365) → **p_fw=0.157**. Nothing clears. The top chroma-dim-MEAN cluster
(d25/26/27) is the same z-scored/truncation-artifact type and is n.s. The only recurring interpretable hint across
instruments is `real_density` (−0.37, denser Hard → worse gen quality) — but n.s. AND it is the within-Hard shadow
of the difficulty/density axis (the macro driver), not a new audio feature.

## ⚠️ OVERTURNED 2026-07-01 — the "null" was NOISE ATTENUATION; the driver is BPM (`probe_quality_variance.py`)
**The three-instrument "null" below is WRONG — corrected here.** It assumed one generation per song = the song's
quality. But generation is STOCHASTIC, so a single score is mostly sample noise. Measuring it (user's idea: K
generations/song on the graded critic, which unlike the saturated binary critic has the range to show within-song
spread):
- **VARIANCE DECOMPOSITION (n=30, K=8):** within-song sd 0.67 ≈ between-song sd 0.76 → **ICC (single generation) =
  0.54** (a single score is ~46% sampling noise). The **8-gen MEAN is 0.90-reliable** (Spearman-Brown; split-half
  +0.85). So a stable per-song quality signal EXISTS; single-generation attribution ATTENUATED it below the
  family-wise floor (true r≈0.68 → ≈0.50 single-gen → buried). **The earlier null was a reliability failure, not an
  absence of signal.**
- **DENOISED ATTRIBUTION (features vs the 8-gen-mean quality): BPM r = −0.682, family-wise p = 0.004 (SIGNAL).**
  Faster Hard songs → worse generations. Validated against every overturning check:
  - NOT density: bpm↔density −0.17; bpm partial|density = −0.745 (stronger); density partial|bpm = −0.45.
  - NOT an outlier: −0.55 on the middle-80% BPM; Pearson −0.53 ≈ Spearman −0.68 (monotone).
  - NOT a critic bias: bpm↔m_real (human chart score) = −0.08 → the critic scores fast HUMAN charts fine; only the
    GENERATOR degrades on fast songs (bpm↔m_gen −0.68). A generation defect, localized by tempo.
  - Sign-consistent across denoising: graded single-gen bpm −0.15 → 8-gen −0.68 (grows as attenuation predicts);
    the earlier POSITIVE binary bpm (+0.04/+0.21) was confirmed noise (those runs were p_fw 0.67).
  - Coherent co-drivers survive partial|bpm: spectral-centroid variability +0.30, perc/harm +0.28, onset-rate +0.23
    ("fast, percussive, spectrally-flat songs are harder to chart").
- **Mechanism (plausible, not isolated):** the fatigue/stamina governor is BPM-coupled (`frame_hz=bpm·4/60`) and
  stressed hardest on fast songs; fast songs are also denser-per-second and likely less represented in training.
  ACTIONABLE — points at fast-song generation / the governor BPM-coupling as the quality bottleneck. Caveat: n=30,
  observational; which mechanism dominates is a follow-up.
- **METHOD LESSON (the keeper):** before concluding "no feature explains Y", check the RELIABILITY (ICC) of Y — a
  null on a target that is mostly sample noise is uninformative. Denoise (average K samples) or measure the ICC
  ceiling FIRST. Artifacts: `probe_quality_variance.py`, `cache/quality_variance_hard.csv`.

## SUPERSEDED — the pre-denoising "three-instrument null" (kept for the record; see the OVERTURN above)
**No interpretable audio feature drives within-Hard generator quality** — confirmed on TWO independent quality
**THREE independent quality instruments now agree** that no interpretable audio feature drives within-Hard
generator quality above the family-wise noise floor: (1) the deployed realism critic (saturated → deficit is the
human score); (2) the validated choreography distance-to-real (non-saturating; the one lead refuted pre-registered);
(3) a purpose-built GRADED critic (retrained to be non-saturating — 44% in-band on generations — STILL p_fw=0.12).
The single robust axis of generator-quality variation is COARSE chart difficulty / note density (across
difficulties; `real_density` is the only recurring interpretable within-Hard hint, n.s., and is that same axis's
shadow). Methodological keepers: **choreography distance-to-real** AND **the graded critic
(`checkpoints/realism_critic_graded`, `train_graded_critic.py`)** are both NON-SATURATING quality instruments —
use either over the near-binary deployed critic for fixed-difficulty quality questions. A "recalibrated critic" via
monotonic rescale cannot help (identical ranks) — the graded RETRAIN was required, and it confirmed the null rather
than overturning it.

Artifacts: `probe_quality_features.py` (critic; `--critic` swaps in the graded checkpoint), `probe_quality_choreo.py`
(choreography), `probe_holdburst_dynamics.py` (pre-registered refutation), `experiments/realism_critic/train_graded_critic.py`
(the graded critic). CSVs `cache/quality_{features_hard,features_hard_margin,features_hard_graded,choreo_hard}.csv`,
`cache/holdburst_dynamics.csv`. Shared infra in `probe_quality_features.py` (`load_val_dataset`, `build_songs`,
`canonical_gen_typed`).

Cross-refs: `choreography_metrics_findings.md` (the validated battery this reuses — trans_KL + holdburst),
`taste_critic_transfer_findings.md` (the near-binary caveat this sharpens), `conditioning-mechanics`
§ + `generation-defaults` (the decode fidelity), experiment-design Rules 1/11/12 (the traps this probe walked
through and cleared).
