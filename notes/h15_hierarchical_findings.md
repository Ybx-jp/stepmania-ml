# H15 Hierarchical pick-then-realize (discrete figure tokens) — findings

Branch `gen/h15-motif-hierarchical` (off `gen/h15-motif`). The banked follow-up to the local-motif work: the
continuous per-section motif rescued candle/trill but NOT jack↔sweep. Hypothesis: a DISCRETE per-section figure
token isolates a figure the continuous projection cannot.

## GATE (no train) — `diag_figure_labels.py`, 600 real charts, section=64
Why the continuous knob fails on sweep, made concrete:
```
trill           35.6%   mean knob-0 +0.23
sweep/staircase 15.5%   mean knob-0 -0.47
step            14.9%   mean knob-0 -0.67
jump/bracket    14.3%   mean knob-0 +0.38
candle/cross    11.2%   mean knob-0 -0.60
jack             8.6%   mean knob-0 +2.04   <-- knob-0 is really a JACK detector
```
- **knob-0 'jack↔sweep' ENTANGLES figures:** jack sits at +2.04, but sweep (−0.47) mushes with step (−0.67)
  and candle (−0.60) on the negative side. The PCA axis can't isolate sweep → that's WHY the continuous local
  vector couldn't steer it. A DISCRETE figure label cleanly separates it.
- Sweep is learnably present (15.5%), within-chart variety high (4.0 distinct classes/chart), section
  dominance a clear plurality (0.53). **VERDICT: build.** Metric must measure realized figure-LABEL fraction
  (Rule 1), not knob-0 (which conflates).

## BUILD
- `typed_model.py`: added `figure_embedding` (NUM_FIGURE_CLASSES=7) + `null_figure`, threaded a discrete
  per-frame figure-token schedule (B,T) through `_cond`/`_decode`/`generate`. DECODER-only (onset decoupled,
  like motif). Zero-init embedding → warm-start no-op. `motif_codebook.py`: `FIGURE_CLASSES`, `figure_token`,
  `figure_token_schedule`. 32/32 gen tests pass (+ existing motif test).
- `train_motif_figure.py`: warm gen_motif_local2, ADD figure on top of the continuous motif (one new variable,
  Rule 11), self-condition on section figure, style off, patience-3. **Early-stopped epoch 4 (best epoch 1, val
  1.0938 ≈ local2 1.092)** — figure barely moved val CE.

## RESULT — `eval_figure_control.py` (12 songs; realized figure-label fraction, base=figure-null)
| target | g=1 base→set (REAL) | lift g=1 | lift g=3 |
|--------|---------------------|----------|----------|
| **sweep**  | 0.05 → **0.13** (0.11) | **+0.08** | +0.01 |
| jack   | 0.13 → 0.10 (0.08) | −0.03 | −0.09 |
| candle | 0.07 → 0.10 (0.06) | +0.03 | +0.03 |
| trill  | 0.41 → 0.40 (0.38) | −0.01 | +0.09 |
Quality intact (F1 0.72, density 0.186 at g=1).

- **WIN (qualified): the discrete sweep token gives the FIRST positive sweep movement in the H15 arc** —
  0.05→0.13, above the real rate, specific and quality-safe. Validates the discrete-pick direction for the axis
  the continuous knob provably couldn't isolate.
- **Modest, and very likely UNDERSTATED (Rule 2/9):** early-stop used `val_total` (CE), which is BLIND to
  figure control, so the embedding barely trained (best = epoch 1); figure sits ON TOP of the continuous motif
  (redundant for prediction); and the "realize" is a soft per-frame bias, not an enforced staircase — so a
  long-range figure (sweep) lifts but can't dominate, and CFG g=3 washes it out instead of amplifying.

## OPTION 1 TESTED → REFUTED (2026-06-24): the modest lift is the CEILING, not an artifact
`train_motif_figure_standalone.py` (warm gen_motif_hr, figure-ONLY conditioning, FIGURE-AWARE selection =
per-epoch generation-time sweep-control lift not val CE, quality guard, patience-5). Best epoch 3, early-stop
epoch 8. Full eval (`gen_motif_figure_solo`, 12 songs): figure=sweep lift **+0.07 @g1 (0.03→0.09)** — IDENTICAL
to the on-top-of-local2 version (+0.08). Both of option 1's sub-fixes tested, neither moved the needle:
- **Standalone (drop continuous motif)** → +0.07, unchanged ⇒ REDUNDANCY (Reason A) was NOT the bottleneck.
- **Figure-aware selection** → the per-epoch control metric bounced ~+0.03 in noise, never climbed ⇒ SELECTION
  (Reason B) was NOT the bottleneck.
**CONCLUSION: a per-section token biases sweep FREQUENCY a little (0.03→0.09, real 0.11) but cannot ENFORCE the
L→D→U→R staircase SEQUENCE. Soft per-frame conditioning has a low ceiling on long-range coordinated figures.
The real lever is a STRUCTURED "realize" (option 2), not more/better figure training.**
METHOD NOTE (Rule 11 misstep, user-caught): standalone discarded gen_motif_local2's WORKING candle/trill
representation for no product benefit, on the redundancy hunch — which the result then refuted. It survives only
as a NEGATIVE CONTROL. Should have kept local2 as the base or run both side-by-side. `gen_motif_figure_solo` is
NOT a deliverable (no candle/trill).

## NEXT (decision point)
1. **Structured "realize" (option 2, the real lever):** commit to a figure for a window, render its coherent
   panel sequence (template / sequence-level head), instead of a soft per-frame bias. The only thing left that
   can enforce a staircase. Bigger build.
2. **Consolidate + accept (product):** add figure on top of gen_motif_local2 (KEEP candle/trill), accept the
   modest sweep nudge (~+0.08) — one model with all working levers; sweep stays a known weak axis.
3. **Stop:** jack↔sweep remains the documented holdout; ship the candle/trill section lever (gen_motif_local2).
Carry-forward deliverable = `gen_motif_local2` (candle/trill); `gen_motif_figure` adds a modest sweep nudge.

## CONSOLIDATED DELIVERABLE — `gen_motif_full` (chosen path: consolidate + accept, 2026-06-24)
`train_motif_consolidated.py` = the CORRECTED option 1 (user-caught): warm-start gen_motif_local2 (KEEP the
working candle/trill continuous-motif representation), ADD the figure token, train both with the FIGURE-AWARE
selection (per-epoch sweep-control lift, not figure-blind CE). train_pat held at ~0.95 (local2 level — rep
preserved); best epoch 5, early-stop epoch 10. ONE model with ALL levers, all quality-safe (onset_F1 0.72–0.74,
density matched):
| lever        | mechanism                | Δ / lift (g=1 / g=3)        |
|--------------|--------------------------|-----------------------------|
| candle/cross | continuous motif (knob 3)| **+1.72 / +3.88** (strong)  |
| jack↔trill   | continuous motif (knob 10)| **+0.73 / +1.10** (preserved)|
| sweep        | discrete figure token    | **frac +0.05 → real 0.11**  |
| jack↔sweep   | continuous knob 0        | +0.03 / +0.15 (stuck — sweep comes from the figure token, not the knob) |
**Carry-forward deliverable = `gen_motif_full`** (radar + section candle/trill + modest figure sweep nudge).
Sweep accepted at the soft-conditioning ceiling; the structured-realize fix (option 2) remains the only path to
a strong sweep lever if ever wanted. `gen_motif_figure_solo` = negative control (not a deliverable).
