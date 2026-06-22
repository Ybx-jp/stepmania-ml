# Scope: v6 retrain — fix 16th UNDER-COMMITMENT (8th-for-16th substitution)

Follows the coherent fair test ([[playtest_log]] 06-22, [[chaos_conditioning_findings]]). Applies the
[[experiment-design]] discipline: precise problem, deployment-matched metric, fair eval, one variable,
confounds explicitly out of scope.

## Problem (precise, from the FAIR test — not a rigged one)
Under COHERENT high-chaos conditioning (`--match_radar` on real high-chaos Hard songs + real density), the
model substitutes 8ths where real charts commit to 16ths:
- GEN: 8th 49–65% / 16th 0–19%
- REAL: 8th 35–43% / 16th 7–28%
Backbone is FINE (quarter/frame survival 0.83) — NOT the defect. Decode is exhausted — NOT the lever.
The defect lives in the **onset head** (which frames get notes; 8th frame t%4==2 vs 16th frame t%4∈{1,3}).

Strong prior on mechanism (already measured): **knows-but-loses.** The model can identify real-16th frames
(16th AUC 0.742) but assigns them lower onset probability than 8ths (p_on@16th 0.415 vs p_on@8th ~0.55),
so under the per-song density threshold the 8ths win the budget. v4's 16th-recall weighting (w16=15) raised
p_on@16th 0.169→0.415 but not enough to beat 8ths. The objective still REWARDS the 8th substitution (gets
recall credit for the easy 8th; the coarse onset feature resolves 8ths, 16ths need the high-res feature).

## OUT of scope (avoid confounds — one change at a time)
- Hands/quads filter relaxation (separate data change + retrain; clean attribution).
- Per-section chaos conditioning (the local-control follow-on).
- 16th PLACEMENT quality ("awkward 16ths") — separate model issue, after commitment.
- Radar-conditioning + heuristic tuning — the user's explicit "circle back AFTER the retrain."

## Step 0 — confirm the mechanism (DONE, `diag_16th_commit.py`, 39 coherent-conditioned songs)
**Result = KNOWS-BUT-LOSES (objective lever), with a placement ceiling.** Per-song threshold, coherent
conditioning: recall of real-8th notes 0.741 vs real-16th notes **0.065** (16ths catastrophically dropped);
p_on @ real-16th-notes 0.475 vs real-8th-notes 0.612 (under-scored ~0.14) vs no-note-16th 0.430; AUC @ 16th
frames **0.671** (> chance → it DOES see them, moderately). The model identifies real 16ths but ranks them
below 8ths, so the threshold drops them. → **OBJECTIVE lever** (raise 16th commitment RELATIVE to 8ths).
(The script's auto-label "CANT-SEE" was an over-strict conjunction; AUC 0.67 is decisively > chance — read
the numbers, don't trust the auto-verdict: [[experiment-design]] rule 1/8.)
- **Placement ceiling:** AUC 0.67 (moderate) → the objective fix improves QUANTITY (how many real 16ths get
  placed; the "8th bias") but caps PRECISION (where they land; the "awkward 16ths") → placement is a FEATURE
  follow-on, separate from v6.
- **Data sub-check INCONCLUSIVE:** the secondary mis-measured simultaneity (note-starts, not sustained hold
  occupancy → "0 rejected", contradicting the known ~55% hands rejection). Not a finding; out of v6 scope.

## Step 1 — v6 retrain (ONE primary change, chosen by Step 0)
Leading candidate (if knows-but-loses): an onset objective that stops rewarding 8th-substitution — make the
8th-vs-16th choice CONDITIONAL on the high-res audio cue and on the chaos condition (chaos should move
probability mass 8th→16th, not raise both — the sweep showed it currently raises both). Concretely, candidates
to pick ONE of after Step 0: (a) precision-side penalty on 8ths placed where the high-res feature supports a
16th subdivision; (b) a relative 16th-vs-8th margin loss at busy beats; (c) condition the phase weighting on
the chart's chaos. Keep the v4 high-res feature; warm-start from v4.
- **Selection / early-stop metric (apply the val_f16 lesson — DEPLOYMENT-MATCHED + target-sensitive):**
  per-song **16th-share error vs real** = |gen_16th_share − real_16th_share| at the song's own density
  threshold, under the SAME (coherent) conditioning used at generation. NOT val_total (blind), NOT a global-
  threshold F1 (mismatched to per-song decode — the exact v5 mistake).

## Step 2 — fair eval (deployment-matched, the same fair test that found the defect)
Re-run the COHERENT GEN-vs-REAL 8th/16th comparison (`--match_radar`, real density) on real high-chaos Hard
songs. **Success = GEN 16th-share approaches REAL** (close the ~12pp gap; GEN 16th 0–19% → toward 7–28%)
WITHOUT regressing: backbone survival stays ≥~0.8, onset_F1 and crit_adj hold, 8th share drops toward real.
Guard against the failure mode: don't just flood 16ths everywhere (would tank backbone + precision) — the
win is SUBSTITUTING 16ths for 8ths at the right beats, measured per-song against real.

## Then (circle back, post-retrain)
Radar-conditioning + heuristic tuning (user: "should work fine when tuned"); then 16th placement quality;
then hands-filter relaxation as its own change. See [[chaos_mechanism_plan]], [[constraint_relaxation_roadmap]].
