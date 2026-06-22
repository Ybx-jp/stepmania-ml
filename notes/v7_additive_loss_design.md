# v6 FAILED (additive trade) + v7 loss design — FOR REVIEW (do not retrain yet)

## v6 result (eval_v6_additive.py, 39 chaotic songs, coherent conditioning, real density)
```
  source           q-rate  8th-rate  16th-rate  density  | shares q/8th/16th
  REAL              0.167    0.117      0.051     0.335   | 50/35/15%
  gen_highres_v4    0.156    0.163      0.017     0.335   | 46/49/ 5%
  gen_highres_v6    0.166    0.009      0.160     0.335   | 50/ 3/48%   <- 8ths NUKED, 16ths 3x overshoot
```
v6 (ranking loss λ=1) stripped the 8th groove and flooded 16ths — the exact "trade away the groove"
failure the additive principle warns against. Backbone (quarter-rate) survived; everything else broke.

## Why it failed (mechanism — important for the redesign)
1. **Aggressive 16th-recall pressure floods the WHOLE 16th phase.** The loss boosts real-16th-NOTE logits,
   but the onset head can't precisely localize 16ths (AUC 0.67), so under strong pressure it raises ALL
   16th-phase frames (empties too) → 48% placed vs real 15%. There's a PRECISION WALL: lifting 16th recall
   trades against 16th precision.
2. **At FIXED density the budget is zero-sum**, so the flood of 16ths displaced the 8ths (and the lowest-
   ranked real-8ths, since empty-8th vs real-8th aren't cleanly separable either). → 8th collapse.
3. **Selection metric `s16err` was doubly blind:** (a) only measured 16th SHARE (no 8th guard), (b) averaged
   over mostly-LOW-chaos val songs (real16≈0, trivially matched), drowning the catastrophic overshoot on the
   few chaotic songs that matter. It reported 0.127→0.088 ("improving") while the model collapsed. This is
   the val_f16 lesson again: metric must capture the FULL target AND be weighted to the regime that matters.

## The deeper lesson (the user's additive insight, formalized)
At fixed density with imperfect per-phase signal, matching real's phase mix REQUIRES a trade — you cannot add
16ths without dropping something. **Real chaos avoids this by raising DENSITY** (more notes total; 16ths added
on top of preserved quarters+8ths — corr(chaos,density)=+0.63, quarter/8th RATES preserved). So the fix is
not "rank 16ths higher at fixed density" — it is "ADD 16ths by coupling density to chaos," plus a GENTLE
16th-recall lift. The honest residual ceiling: AUC-0.67 16th localization caps placement precision (a feature
problem, separate) — v7 can fix the RATE/quantity additively, not perfect placement.

## v7 recipe (FOR REVIEW)
1. **Additive selection metric (the clear fix):** per-song L1 over ALL THREE phase RATES vs real
   (|q|+|8th|+|16th|), restricted/weighted to chaotic songs (real 16th-share ≥ 5%). Penalizes 8th collapse
   AND 16th overshoot; dominated by the regime that matters. (This is eval_v6_additive.py's metric, used for
   selection.) Minimize.
2. **Gentle, balanced 16th-recall lift:** v6's λ=1 was ~5–20× too strong. Options to weigh:
   - (a) tamed ranking loss, λ sweep 0.05–0.3, NOT stacked on v4's heavy w16=15 (the two double-pushed);
   - (b) just retune v4's per-phase BCE weights (modest w16, keep w8) to nudge 16th recall toward real;
   - (c) a direct per-phase RATE-matching loss (penalize expected placed-rate deviation from real) — most
     direct on the target, but a distributional/threshold-aware loss (more to implement).
3. **Protect 8ths:** ensure real-8th recall stays ~real (don't let the 16th lift displace them) — either by
   not over-weighting 16ths (2b) or a symmetric "real-8th > empty" term (2a).
4. **Density coupled to chaos at GENERATION (the additive mechanism):** generate high-chaos with
   --target_density on the real chaos→density curve (~0.22 low → ~0.34 high) so 16ths are ADDED, not traded.

## REFINED (user direction 06-22) — reweighted BCE + the MUSICAL-LINKAGE feature path
Decisions:
- **Loss = reweighted BCE** (confirmed): modest w16, keep w8, paired with the additive per-phase-rate
  selection metric. (Ranking loss dropped — v6 proved it untameable.)
- **Density coupling is "too flat"** (user): a per-song scalar controls how MANY 16ths, never WHICH — it
  can't fix the AUC-0.67 *localization* wall. The real lever is **musical linkage**: tie 16th placement to
  audible cues. This is the mountaintop; decode/fine-tune is "fine-tuning at best."
- **Sequencing change:** do NOT run reweighted-BCE v7 standalone (it fixes rate, inherits the placement
  wall, and a feature change needs a retrain anyway). PROBE features first, then fold reweighted-BCE into
  the ONE feature retrain.

Candidate musical-linkage features (map of user's cues):
- **perc_hr** — HPSS-PERCUSSIVE high-res onset (drum fills/runs = most 16th sources). NB H8 found HPSS
  onsets redundant at COARSE res; fine-res percussive-only is untested — the probe settles it.
- **harm_hr** — HPSS-HARMONIC high-res onset; **chroma_flux** — melodic note attacks (synth/lead accents).
- **BPM** — currently IMPLICIT (baked into grid hop); add as explicit conditioning scalar so the model can
  thin 16ths at high BPM like real charters do.
- Pushback noted: AUC 0.67 may be ARCHITECTURE (per-frame conv can't see a 16th-RUN), not features — the
  probe distinguishes (better features lift AUC => features; flat => architecture/audio-ambiguity).

Plan:
1. **Feature probe (cheap, no retrain) — `diag_16th_features.py`:** does perc_hr / harm_hr / chroma_flux
   beat mixed_hr (AUC ~0.67) at 16th localization? + BPM vs real-16th-rate correlation.
2. **Gate:** features lift AUC -> ONE retrain with winning feature(s) + reweighted-BCE + additive metric +
   density-coupled generation. Flat -> architecture pivot (sequence-aware onset head) or accept the ceiling.
3. **Eval on PLACEMENT** (16th-AUC / onset-F1-at-16th vs real 16ths), not just rate, + playtest.
Expectation: even with features, placement has a real ceiling (audio is ambiguous — many valid chartings);
aim to MEANINGFULLY raise AUC so 16ths land on audible events often enough to feel intentional.

## PROBE RESULT (diag_16th_features.py) — musical-cue FEATURES REFUTED
20 chaotic songs, 16th-localization AUC (real-16th-note vs no-note at 16th frames):
```
  mixed_hr  0.586 (current)   perc_hr 0.581   harm_hr 0.579   perc+harm 0.586
  BPM vs real 16th-rate: Spearman -0.229  (predicted sign, weak)
```
- **HPSS percussive/harmonic high-res onsets do NOT beat the mixed onset** (all ~0.58, within noise). The
  drums/synth-cue hypothesis (via HPSS) is refuted — separating sources adds no 16th-placement signal.
- Raw-onset AUC (~0.59) is BELOW the model's learned p_on AUC (0.67): the model already extracts more from
  context than any single raw onset cue → little headroom for new onset features; the bottleneck is how the
  model USES signal, not the raw cue.
- BPM weak (−0.23). chroma_flux (melodic) UNTESTED (chroma_* segfaults here, numba); harm_hr is a proxy and
  didn't help, so melodic-onset is unlikely to either — small open gap.
- **Gate result: FLAT → not a feature problem.** 16th placement is near an audio-ambiguity ceiling (many
  valid chartings of the same fill — partly charter style) and/or an architecture limit (per-frame conv head
  can't model a 16th RUN). Probe rules OUT hand-crafted features; can't fully separate ambiguity vs arch.

## Decision (post-probe)
Don't engineer features. Proceed with **reweighted-BCE additive retrain** (the loss already chosen) as the
realistic, concrete win: gets the 8th/16th BALANCE real-like (additive — quarters+8ths preserved, 16ths
added via density coupling), with 16th PLACEMENT at the model's existing ~0.67 quality (decent, not perfect —
the audio ceiling). Honest expectation: this fixes the AMOUNT/feel, not perfect placement. Longer bet if
placement still feels off in playtest: a sequence-aware onset head (architecture), NOT more features.

## Open questions for the user's review
- Loss primitive: tamed ranking (2a) vs reweighted BCE (2b) vs direct rate-matching (2c)? (I lean 2b first —
  simplest, lowest-risk; v6 showed the ranking loss is hard to tame.)
- λ / weight sweep: single value or a small grid? (Sweep, given v6's extreme sensitivity.)
- Density coupling: decode-only (--target_density) for now, or also teach the model chaos→density in training
  (bigger change)?
- Accept the AUC-0.67 placement ceiling for v7 (fix RATE only), and treat placement precision as the next,
  separate feature pass? (I think yes — one change at a time.)
