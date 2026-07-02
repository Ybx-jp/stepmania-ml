# Quality-feature attribution — lineage

**Status (2026-07-01): RESOLVED POSITIVE — the driver is BPM.** Question (user): *"which audio features influence
the variation in QUALITY among songs applying the canonical defaults?"* Answer: **song TEMPO (BPM) — faster Hard
songs → worse generations (r=−0.68, family-wise p=0.004).** ⚠️ This OVERTURNED an earlier committed "three-instrument
NULL": that null was a NOISE-ATTENUATION artifact (a single generation's quality score is ~46% sample noise, ICC=0.54).
The user's fix — measure/average K generations/song — revealed the signal (8-gen mean 0.90-reliable). Validated:
not density (partial −0.75), not outlier, not a critic bias (fast HUMAN charts score fine). Mechanism: the
BPM-coupled GOVERNOR is RULED OUT (paired governor-on/off ablation, flat: spearman(bpm, q_off−q_on)=−0.04 p=0.59) →
the defect is INTRINSIC to the generator. Narrowed: NOT training coverage (fast region well-sampled; slowest bin
fewest charts yet good) and NOT the onset head (n=176: p_onset placement AUC vs BPM flat/better) → BY ELIMINATION the
PATTERN/TYPE head (which-panel / AR sequence quality) at high density — CONFIRMED (`probe_bpm_head_decomp.py`,
onset_override A/B: real perfect onsets STILL slope −0.38 with BPM; controlled paired real-vs-gen delta flat +0.11 p=0.65).

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

## THE OVERTURN — reliability/denoising revealed BPM (`probe_quality_variance.py`, the culminating step)
The three "instruments" all scored ONE generation per song. But generation is stochastic → a single score = quality
+ big sample noise. The user's two ideas (denoise; measure within-song variance across K generations) were the fair
test the committed null had skipped.
- **Variance decomposition (graded critic, n=30, K=8):** within-song sd 0.67 ≈ between-song sd 0.76 → **ICC(single)
  =0.54**; the K=8 MEAN is **0.90-reliable** (split-half +0.85). A stable per-song signal EXISTS; single-gen
  attribution attenuated it below the family-wise floor (true r≈0.68 → ≈0.50 single → buried).
- **Denoised attribution: BPM r=−0.68, family-wise p=0.004.** Overturning checks all pass (density-partial −0.75;
  outlier-robust −0.55 mid-80%; **critic-bias ruled out — bpm↔m_real human-score −0.08 vs bpm↔m_gen −0.68**, so it's
  a GENERATION defect; sign-consistent single-gen −0.15 → 8-gen −0.68). Co-drivers survive partial|bpm.
- **MECHANISM ablation (`probe_bpm_governor_ablation.py`, n=30, K=6/arm):** paired governor-ON vs governor-OFF
  (`fatigue_penalty=None, stamina_ceiling=None`; one labeled variable, playability still forced). Within-song paired
  test spearman(bpm, q_off−q_on) = −0.04 (p=0.59), slope did NOT flatten off (ON −0.47 / OFF −0.67), main effect ~0.
  → the BPM-coupled governor is NOT the cause; the defect is INTRINSIC (upstream of decode). The within-song PAIRED
  design was the power win — comparing the two noisy slopes alone (−0.47 vs −0.67) would have been ambiguous.
- **Attribution correction (the arc's biggest):** a NULL is only meaningful if the TARGET is reliable. I committed a
  three-instrument null without checking the target's ICC — it was ~46% noise. **Rule add: before "no feature
  explains Y", measure reliability(Y)/ICC and denoise.** Mirror image of the earlier lessons (there, POSITIVES were
  overturned by fair tests; here a NULL was). The graded critic was still essential — its dynamic range is what made
  the within-song variance visible (the binary critic railed all K to ~0, ICC undefined).

## The graded-critic follow-up (the "recalibrated critic", done right — the instrument that enabled the overturn)
The rescale being a dead end, the genuine version = a RETRAINED GRADED critic
(`experiments/realism_critic/train_graded_critic.py` → `checkpoints/realism_critic_graded`). Objective: GRADED
corrupted-real ladders (panel-scramble fraction {0,.2,.45,.7,1.0} + shift {0,2,6,16}) + a WITHIN-SONG margin-ranking
loss (monotone-decreasing score along each ladder; within-song pairs hold density/timing/audio fixed = the v2 taste
isolation, now graded) + end-anchor BCE; score = logit margin; warm-started from the binary critic. **KEEPS the v2
no-generator anti-fingerprint property.**
- **Instrument SUCCESS:** trained ladder cleanly graded; on canonical Hard GENERATIONS the saturation is fixed —
  gen 0.1–0.9 band **0%→44%**, `m_gen` sd 0.75. A non-saturating realism critic now exists (reusable).
- **Substantive: STILL NULL** — family-wise p_fw=0.118 (all) / 0.157 (interpretable). Top = z-scored chroma-mean
  cluster (same truncation-artifact type) + `real_density` −0.37 (n.s., = the difficulty-axis shadow). CAVEAT:
  per-song ladder monotonicity ~0.35 → single-chart scores noisy (cross-critic agreement only +0.32).
- **Conclusion:** three instruments now agree on the null; the graded retrain confirmed rather than overturned it.
  No open fork. If ever revisited, reduce the graded critic's per-chart noise (wider/fewer ladder levels, more
  epochs, a listwise loss) — but the substantive question is answered.
