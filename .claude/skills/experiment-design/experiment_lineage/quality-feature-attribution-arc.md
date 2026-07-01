# Quality-feature attribution — lineage

**Status (2026-07-01): CLOSED NEGATIVE (substantive) + a methodological WIN.** Question (user): *"which audio
features influence the variation in QUALITY among songs applying the canonical defaults?"* Answer: **no
interpretable audio feature drives within-difficulty generator quality** (two independent instruments agree; the
one lead was an artifact, refuted by a pre-registered test). The only robust axis is coarse difficulty/density.
**Active fork:** the user directed a **retrained GRADED critic** as the next instrument.

Primary notes: [`notes/quality_feature_attribution_findings.md`](../../../../notes/quality_feature_attribution_findings.md).
Probes: `probe_quality_features.py` (critic), `probe_quality_choreo.py` (choreography), `probe_holdburst_dynamics.py`
(the pre-registered refutation). Memory: [[quality-feature-attribution]].

## Hypothesis chain (believed → learned)
1. **Believed:** per-song generator quality (critic deficit `P(real)_human − P(real)_gen`) varies across songs, and
   audio features explain the variation. **Operationalized** faithfully = deployed canonical decode via
   `CANONICAL_DECODE`+`decode_harness` (NOT the stale `eval_taste_current.py`, which is pattern_temp 0.7 / no
   16th-unlock).
2. **Learned (pooled smoke, n=8):** the deficit's dominant axis is **chart DIFFICULTY / density**
   (`Spearman(difficulty,deficit)=+0.90`; generator beats humans on Beginner, railed on Hard). Pooling would let any
   difficulty-proxy feature (density/onset-rate/tempo) masquerade as a quality driver → **Rule 12**: stratify.
3. **Learned (within Hard, n=48):** the critic **SATURATES** — 94% of canonical Hard gens rail to "fake" (0%
   mid-band). So generator-quality has ~no dynamic range; the deficit becomes **91% the HUMAN chart's score**
   (Rule 11 in the wrong term). Feature ranking looked like it had top drivers → **family-wise permutation floor
   p_fw=0.67 = noise**.
4. **Tested the obvious rescue (logit MARGIN):** identical ranks (P=sigmoid(margin), monotonic) → **a
   recalibrated/temperature-rescaled critic CANNOT change a rank-based result.** Saturation is genuine, not a
   float-precision tie.
5. **Swapped the instrument → choreography distance-to-real** (`probe_quality_choreo.py`, n=64), the validated
   battery ([choreography-metrics arc / notes](../../../../notes/choreography_metrics_findings.md)): `trans_KL`
   (panel-transition KL to pooled real) + `hold_burst` excess (the play-feel-validated defect) + panel_TV +
   composite. **The proxy VALIDATED** (non-saturating: composite sd 1.77; discriminates gen>real on trans_KL &
   holdburst; panel_TV correctly flagged non-discriminating). **Substantive:** composite & trans_KL = noise floor
   (p_fw 0.31/0.36); ONE target `holdburst_excess` hit p_fw=0.027 — best feature d31_std (a z-scored chroma std).
6. **Ran the check that would overturn it (pre-registered, `probe_holdburst_dynamics.py`, n=64):** the hit lived in
   the `d##_std` block (z-scored spectral/chroma variability = a z-score/TRUNCATION artifact) and was absent from
   interpretable features. Reused the already-computed `holdburst_excess` (no regen) + recomputed CLEAN dynamics
   (spectral/chroma/MFCC flux) from RAW audio on **the generator's actual first-T window**; ONE target, ONE
   directional hypothesis. **Result: composite r=+0.059, one-sided p=0.685 (wrong sign); family-wise p_fw=0.29.**
   The lead is an **ARTIFACT**. Dead.

## Attribution corrections (what would have made each conclusion wrong)
- **Rule 12 (stratify):** pooling across difficulty would have "found" density/tempo as quality drivers — they only
  proxy the difficulty axis. Fixed by within-Hard.
- **Rule 11 (dynamic range in the RIGHT term):** the deficit *varied* (sd 0.32) so a naive gate passed — but 81% of
  that variance was the HUMAN chart's score, not the generator's. The generator target was saturated. Decompose the
  target, don't gate the composite.
- **Family-wise permutation floor:** the raw ranking always shows ~5 features above the uncorrected |r|>0.28 line by
  chance across ~99 features; the permutation max-null (p_fw) is what separates signal from noise. Used at every step.
- **Pre-registration + de-artifacting (Rule 7/9/10):** the one "signal" was killed only by (a) recomputing the
  feature cleanly (raw audio, generator's window, no z-score/truncation) and (b) committing to one target/one
  hypothesis before looking. Had I stopped at step 5 I'd have shipped a false "spectral dynamics drive the hold-burst
  defect."
- **Monotonic invariance (twice):** the logit-margin (step 4) and any temperature/Platt recalibration give identical
  Spearman ranks — a whole class of "fix the instrument" moves that cannot work for a rank question.

## Durable wins (don't re-derive)
- **Choreography distance-to-real is THE non-saturating quality instrument** for fixed-difficulty questions — reuse
  `trans_KL` + `hold_burst` from `choreography_metrics.py`/`bipedal_metrics.py`. Prefer it over the near-binary critic.
- **The critic saturates at fixed difficulty (~94% railed on Hard)** — sharpens the taste-critic "near-binary"
  finding; a monotonic recalibration is a dead end.
- Gotchas: `val_ds.warm_cache()` = ~30min whole-val extract (drop it, lazy `val_ds[i]`); `librosa.chroma_stft`
  segfaults via `estimate_tuning`→`piptrack` numba → `tuning=0.0`. k=3→k=1 safe (within-song critic sd 0.016).

## Corroborates / depends-on
- **depends-on** the taste-critic thread (🟡 lineage stub, INDEX) / [[taste-critic-transfer]] — this thread is the
  quantitative sharpening of its "near-binary separator, not a graded scorer" caveat; the retrained-graded-critic
  fork is the shared next step.
- **depends-on** the choreography battery (`notes/choreography_metrics_findings.md`) — reuses its validated
  `trans_KL`/`hold_burst` (incl. the B4U play-feel meta-validation) as the alternative instrument.
- **corroborates** [[generation-defaults]] / `decode-harness-single-source` — a probe built on the harness matched
  deployment by construction; the stale `eval_taste_current.py` is the counter-example.

## Open fork
Retrain a **GRADED critic** (ranking loss / graded-corruption targets) → re-run feature attribution on the
holistic-realism axis. `/autotune` first. Expect a possible repeat of the null (choreography, an orthogonal
validated axis, already returned null), but it reads a different quality dimension.
