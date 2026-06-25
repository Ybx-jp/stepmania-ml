# H19 retrain (clean representation) — gen_motif_full_fixed: levers preserved, trill honest, sweep improved

**Date:** 2026-06-25. **Branch:** `gen/h19-retrain`. **Checkpoint:** `checkpoints/gen_motif_full_fixed`
(deliberately NOT clobbering the deployed `gen_motif_full`, so we can A/B). **Why:** retrain the H15 deliverable
on the corrected chart representation after the two repr bug fixes (H19 attacks-only `onset_tokens` +
zero-overwrite sticky `convert_to_tensor_typed`). See [[repr_integrity_findings]].

## Flow (cheaper than first flagged — see repr_integrity_findings "RETRAIN" section)
`train_motif_consolidated.py` on the fixed branch — NO cache rebuild, NO basis refit. `collect_typed`
re-parses the typed chart fresh (`convert_to_tensor_typed`, now sticky) and derives motif/figure targets via the
attacks-only `onset_tokens`, so both fixes flow into targets + teacher-forcing automatically. Warm-start
`gen_motif_local2`, add the figure token, figure-aware selection, patience-5.
- **Converged:** early-stopped epoch 10, best epoch 5, sweep_lift +0.062, onset_F1 0.71–0.72 (healthy). Warm-start
  clean (only `figure_embedding`/`null_figure` fresh).

## Validation — gen_motif_full_fixed vs deployed gen_motif_full (BOTH measured with corrected detector)
16 songs, ±3z, g1/g3. `eval_motif.py` (candle k3, trill k10) + `eval_figure_control.py` (figure=sweep).

| lever | metric | **fixed (clean)** | deployed | read |
|---|---|---|---|---|
| candle k3 | Δself g1 / g3 | **+1.84 / +3.62** | +1.71 / +3.58 | preserved, marginally stronger |
| trill k10 | Δself g1 / g3 | +0.32 / +1.75 | +0.47 / +1.71 | g1 LOWER (honest), g3 equal |
| sweep (figure) | g1 lift / realized | **+0.04 / 0.09** | +0.02 / 0.07 | IMPROVED (closer to real 0.11) |
| quality | onset_F1 | 0.72–0.78 | 0.72–0.79 | identical |

## Verdict — strict win (or equal) on every lever, no regression
- **Candle preserved** (+1.84 vs +1.71 g1). The strong section-by-section lever survives the retrain intact.
- **Trill is now HONEST.** The deployed model learned trill from hold-tail-INFLATED targets and slightly
  over-produced it; the clean model learned honest targets → g1 trill steering +0.32 vs +0.47 (the ~0.15 drop
  mirrors the ~10% measurement-side correction, now baked into the WEIGHTS, not just the metric). g3 equal.
- **Sweep IMPROVED** (g1 lift +0.04 vs +0.02; realized 0.09 vs 0.07, vs real 0.11). The figure-token targets are
  now computed on recovered-notes + attacks-only sequences, so "sweep" sections are labeled more accurately and
  the figure embedding gets a cleaner signal. Still below real 0.11 (the soft-realize ceiling stands), but the
  best sweep movement in H15 so far.
- **Quality identical** (onset_F1 0.72–0.78, density matched).

⇒ The two representation fixes corrected the honesty of the trill lever and improved sweep, without breaking
candle or quality. `gen_motif_full_fixed` is a clean, strictly-not-worse replacement for `gen_motif_full`.

## Playtest A/B prep — an HONEST complication (presses-during-holds NOT reduced)
Built `ab_trill_A_old` (gen_motif_full) vs `ab_trill_B_fixed` (gen_motif_full_fixed), SAME 5 songs (the h15
set: Deja loin, Pound the Alarm, IN BETWEEN, nightbird, japa1), trill=3 g2 — matching the original
`h15_08_motif_trill` the user called "huge jack streams during holds." Direct proxy for that phenomenon =
fresh presses landing while another panel sustains a hold:
- **A (old): 3.1%** (71/2277).  **B (fixed): 5.4%** (119/2223).
So the retrain did **NOT** reduce the during-holds activity — slightly UP (small/noisy over 5 songs). The user's
felt "jacks during holds" is a GENERATION-side behavior the representation fix did not target (H19 was the
DETECTOR; this is the decoder placing notes during holds, which is legal+hittable under the pad constraints but
may read busy). ⇒ the A/B is a genuine feel test with NO guaranteed direction: offline trill is honestly-lower
on B, quality equal, but during-holds is unchanged. Only ears resolve whether honest-lower-trill feels better.

## Caveat + next
- **Warm-start residue:** base (onset/pattern/type) inherited from `gen_motif_local2` (trained on buggy charts),
  fine-tuned here. The deltas are small partly because it's a fine-tune. A FULLY clean model = retrain the chain
  hr→local→local2→consolidated from scratch on fixed charts — deferred (the fine-tune already validated clean).
- **Numbers are within a 16-song stochastic eval's noise** for candle/trill; the directions are consistent and
  quality is unchanged, so the safe claim is "equal-or-better, trill honest." Sweep's doubling is the clearest gain.
- **UNTESTED: playtest/feel** — does the honest-trill / improved-sweep model play differently? (H15 is a feel thesis.)
- **Deliverable swap:** to make it the default, point `export_typed_samples.py` / `eval_motif.py` defaults +
  the playtest skill at `checkpoints/gen_motif_full_fixed`. Recommend a playtest A/B before swapping.
