# H4 — where is the off-beat/syncopation signal? (offline, no model)

*2026-06-20. Driven by the Hard best-of-N result (`stage2a_critic_findings.md` Stage 2b): generated Hard
charts are density-matched but **tame/on-grid**; the critic + the user agree they're un-Hard-like. H4
says the model renders chaos as a uniform global grid manipulation "because it can't see which offbeats
deserve a note." This note tests that premise at the **feature** level — is the signal absent, or just
unused?*

Each audio frame is one 16th-note grid position (audio is BPM-aligned via `chart.hop_length`). Diagnostic
`experiments/realism_critic/diag_offbeat_signal.py`: for REAL charts (difficulty ≥ Medium), within each
metric-phase bucket, does an audio feature distinguish note-frames from no-note-frames? (per-song ROC-AUC,
averaged over 80 songs.)

## Result 1 — alignment is correct, but the onset envelope is nearly phase-FLAT

Mean shipped `onset_env` by phase `t%4`: **0=0.121, 1=0.105, 2=0.106, 3=0.106**. The downbeat (t%4==0)
*is* the maximum → **audio is correctly grid-aligned** (rules out the alignment confound). But the
downbeat is only **1.065×** the offbeats — the broadband onset envelope barely emphasizes the beat.

## Result 2 — per-frame, the onset features barely predict note placement, even ON the beat

Per-song ROC-AUC (note vs no-note within phase), mean over songs:

| phase | note% | onset_env | onset_rate | perc_onset(HPSS) | harm_onset(HPSS) |
|---|---|---|---|---|---|
| on-beat | 72.9% | 0.569 | 0.582 | **0.585** | 0.528 |
| 8th-off | 32.4% | 0.534 | 0.523 | **0.544** | 0.509 |
| 16th-off | 4.7% | 0.527 | 0.532 | **0.546** | 0.430 |

(3-frame max window — tolerant of STFT smear — lifts these only ~0.02–0.05; perc on-beat 0.585→0.601.
So it is NOT a simple ±1-frame misalignment.)

**Even on-beat, a single onset feature gives AUC ≤ 0.60; off-beat it is ~0.53 (near chance).** HPSS
percussive onset is marginally best but still weak.

## Interpretation

**The off-beat-saliency signal is largely ABSENT from the current audio representation, not merely
unused by the model.** Consequences:
- The model's strong global onset AUC (~0.9) rests on the **metrical prior** (metric-phase + learned
  density), NOT on audio-driven event detection. It learns "notes go on-beat at this density," which
  scores well globally because notes really do cluster on-beat (73%).
- For **syncopation/offbeats there is essentially no per-frame audio signal** in these features for the
  model to latch onto. This *explains H4's symptom mechanistically*: the chaos conditioning has nothing
  local to key on, so amplifying it (CFG) can only act as a **uniform global shift** — exactly the
  "degenerate global grid manipulation" observed.

**Open ambiguity (one more test resolves it):** weak AUC could mean (a) the *feature* is too coarse
(broadband, ~93ms STFT window smears the 16th grid) — a RESOLUTION problem fixable by a sharper onset; or
(b) broadband onset just doesn't carry offbeat placement — a REPRESENTATION problem; or (c) offbeat
placement in real charts isn't audio-onset-determined at all (it's groove/pattern-driven), in which case
the lever is the AR pattern/rhythm model, not features. `diag_highres_onset.py` (high-res + superflux
onset, max-pooled per grid cell) tests (a) vs (b/c). **RESULT below: it's (a) — resolution.**

## Result 3 — it's substantially a RESOLUTION problem; a high-res onset recovers the off-beat signal

`diag_highres_onset.py` (60 charts ≥ Medium): recompute onset at **hop=128 (~5.8ms), n_fft=512**, then
**max-pool into each 16th-grid cell** (so the cell keeps its within-cell transient peak instead of an
STFT average). Per-song ROC-AUC, note vs no-note:

| phase | shipped onset_env | high-res flux | high-res superflux |
|---|---|---|---|
| on-beat | 0.569 | **0.654** | 0.653 |
| 8th-off | 0.534 | **0.596** | 0.592 |
| 16th-off | 0.527 | **0.662** | 0.659 |

Alignment/phase contrast also sharpens: high-res mean onset by `t%4` = **0=0.273, 2=0.231, 1=0.171,
3=0.179** (downbeat ≈1.6× the 16ths, vs only 1.065× for the shipped feature). The biggest gain is at
**16th-off (0.527→0.662)** — exactly where syncopation lives. Superflux ≈ plain flux, so the win is the
**resolution + max-pool**, not transient-emphasis per se.

**Conclusion:** the shipped onset feature throws away the off-beat signal by computing onset detection at
the coarse 16th-note hop with a ~93ms window. The signal IS in the audio; high-res-then-pool recovers it.
(0.66 is a real signal, not a slam dunk — a chunk of note placement is still groove/pattern-driven, not
onset-driven — but the resolution fix is concrete and meaningful, and this is a single-feature LOWER
bound on what the transformer could extract.)

## H4 fix — concrete, validated next step

**Add a high-resolution, grid-cell-max-pooled onset feature** to `AudioFeatureExtractor`: compute
`librosa.onset.onset_strength` at hop≈128 / n_fft≈512 on the offset-sliced audio, then max-pool the
high-res frames falling in each 16th cell `[t·hop, (t+1)·hop)`. Append as a new dim (keep dims 0..40
unchanged, like the Stage-1 additions). This is the prerequisite for event-driven syncopation: it gives
the onset head + chaos conditioning a *local* off-beat cue to key on, which today's smeared feature
denies them.

**Generalizes a prior finding:** H8 found HPSS onsets near-redundant — but those were also computed at
the coarse grid hop, so they inherited the same smearing. The lesson is broader: **onset-family features
should be extracted high-res then pooled to the grid, never computed at the grid hop directly.** Worth
re-checking whether high-res HPSS perc-onset adds beyond high-res broadband.

## Result 4 — offline gate PASSED: feature implemented, survives the real pipeline

Added `use_highres_onset` to `AudioFeatureExtractor` (config flag + `_highres_pooled_onset`, appended as
the last dim so 0..40 are unchanged; assembled dim = **42**, new dim = index 41). `diag_confirm_highres_feature.py`
runs the *real* extractor (slice → high-res detect → max-pool per 16th cell → `_normalize_envelope` →
`get_aligned_features`) and re-scores the new dim in place:

| phase | as-assembled AUC | standalone target |
|---|---|---|
| on-beat | 0.655 | 0.654 |
| 8th-off | 0.596 | 0.596 |
| 16th-off | **0.662** | 0.662 |

Identical to standalone; alignment contrast preserved (downbeat 0.273 vs 16ths ~0.17). The feature
survives normalization + grid-alignment with no smearing reintroduced. **Retrain de-risked.**

**Test plan after adding the feature:**
1. (offline, no train) confirm the new dim's off-beat AUC ≈ 0.66 inside the assembled feature vector.
2. retrain/fine-tune the generator with the new dim; measure generated-chaos phase histogram (does chaos
   now add off-beats *at audio onsets* instead of a uniform smear?) and off-beat onset↔audio correlation
   (was +0.10 under chaos in the H4 06-19 diagnostic).
3. playtest the chaos knob (the original H4 failure) + a fresh Hard best-of-N (the 2b tameness).

**Bearing on 2c:** critic-guided fine-tuning can't teach salient-offbeat placement if salience isn't in
the inputs. This feature is plausibly a prerequisite for 2c to move Hard at all.

## Result 5 — the warm-started retrain ran but DID NOT ENGAGE the feature (negative result)

`train_highres.py` completed (warm-start gen_stage1 + conv 41→42 expand, zero-init new col; 15 epochs;
checkpoint `checkpoints/gen_highres/best_val.pt`). Offline gen metrics on par with gen_stage1 (onset_F1
0.757, density matched, crit_adj 0.938) — but those are blind to syncopation, so uninformative here.

**The feature is unused.** Two checks:
- **Weight:** the new conv input column (dim 41) has norm **0.127 = smallest of all 42 dims (rank 1/42),
  ~11% of the mean (1.09)**; coarse onset_env (dim 13) is 1.15. It barely moved off its zero-init.
- **Behavior (`diag_h4_engagement.py`):** teacher-forced **ablation KL = 0.0000** — zeroing dim 41 changes
  the onset logits by nothing (cf. Stage-1 chroma KL 10.3 "used heavily", HPSS 0.29 "near-ignored").
  Conclusive: the model ignores it.

**Why (the real lesson):** we warm-started from a model that had *already converged without* the feature,
and off-beat frames — the only place a sharper onset helps — are ~5% of frames, so they contribute almost
nothing to the average onset CE. **There is essentially no gradient pressure to grow the new column.**
Adding the feature is necessary but not sufficient (H6 again): the *objective* also has to reward
off-beat correctness, and frame-wise CE doesn't — the same "the loss doesn't care about syncopation/taste"
root that motivated the Stage-2 critic.

**Fix = supply incentive, not just the feature (H4-v2 plan):**
1. **Off-beat-upweighted onset loss** — weight off-beat frames (t%4≠0, esp. 16ths) higher in the onset
   BCE so syncopation errors actually move the gradient. (The single most important lever — without it,
   even a from-scratch init likely under-uses the feature, since on-beat prediction dominates the loss.)
2. **Re-initialize the audio encoder** (don't warm-start it / random-init the new column) so it must
   relearn a representation over all 42 dims rather than sitting in the no-feature optimum. Keep the
   warm-start for the decoder/pattern/type heads (those work).
3. Optionally **oversample high-chaos charts** (the feature matters most under chaos; low-chaos charts
   dominate the data and dilute the signal) and re-test chaos WITH CFG (g≈2) — the smear baseline was at
   g=2, not g=1.
Cache is already warm (`cache/samples_v3`), so H4-v2 is ~a retrain, no re-extraction.

## Result 6 — H4-v2 engaged the feature but it BARELY MATTERS; chaos still smears. CONCLUSION: not a feature problem.

`train_highres_v2.py` (random-init new col at full magnitude + off-beat-weighted onset loss, `--offbeat_weight 3.0`;
`checkpoints/gen_highres_v2/best_val.pt`). The incentive worked *at the weight level* but not behaviorally:
- **Column retained:** new-col(41) norm **1.044** (rank 9/42, vs v1's suppressed 0.127). So random-init +
  off-beat weighting kept it engaged — the v1 "dead column" problem is fixed.
- **…but its effect is tiny:** zeroing dim41 shifts onset logits by only **~0.017 max** (mean Bernoulli
  KL <5e-5, prints 0.0000). The feature is *used, weakly*. off-beat dim41-AUC of generated notes ticks up
  (0.50→0.62 vs stage1) but that's a small, partly-incidental effect.
- **Chaos still smears:** under chaos@CFG=2, on-beat% = **4.4% (v2) vs 6.1% (stage1)** — both wildly
  over-syncopated vs real charts (~80-90% on-beat). **v2 did NOT fix chaos.** And it regressed quality
  slightly (onset_F1 0.757→0.744, crit_adj 0.938→0.906).

**Why the feature barely matters even when engaged:** its marginal predictive value is low. The off-beat
AUC it recovered was only 0.66, and that signal **largely overlaps with the coarse onset (dim13) the model
already uses** — so the conv can hold weight on it but it adds little *new* information. Most of real
off-beat placement is **not audio-onset-determined at all**; it's groove/pattern (charter style). And the
chaos knob is a **global conditioning scalar** amplified by CFG — the model renders it as a global
off-grid shift, and a weak local feature doesn't change that global mechanism.

**CONCLUSION (well-evidenced, two independent retrains): chaos/syncopation is NOT a feature problem.**
Stage 1 (chroma) didn't fix chaos; H4 (high-res onset) didn't either, even with full incentive. The
audio off-beat signal that exists is weak and redundant with the onset the model already has. The real
levers are **(a) the conditioning mechanism** — how the chaos scalar is injected (e.g. a per-frame
chaos×onset *gate* that ties off-beat placement to the local feature, rather than a global additive bias),
and **(b) the objective** — frame-wise CE never rewards tasteful syncopation, which is exactly why the
Stage-2 **taste critic** is the thing that actually tracks musicality. This vindicates H6-revised
("fixing chaos needs the conditioning mechanism/objective, not decode or more features").

**Disposition:** gen_highres / gen_highres_v2 are **parked** (no quality win, slight regression). The
high-res feature + its cache (`cache/samples_v3`) stay available but are not adopted as the base. Redirect
the chaos/Hard effort to objective/conditioning work (Stage 2c critic-guided fine-tune; or a chaos-gating
conditioning redesign), NOT more feature retrains.
