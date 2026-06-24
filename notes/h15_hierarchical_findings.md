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

## NEXT (decision point)
1. **Re-train figure properly (cheap):** STANDALONE figure (drop the continuous motif so figure must carry the
   signal) + MORE epochs + a FIGURE-AWARE selection metric (sweep-realization on a held set, not val CE — Rule
   2). Likely the modest result is a training/selection artifact, not the ceiling.
2. **Stronger "realize" (bigger):** the soft per-frame token can't enforce a staircase; a structured/template
   decode (commit to a figure for a window, realize its panel sequence) is the real fix for long-range figures.
3. Carry-forward if pursued: `gen_motif_figure`. The candle/trill levers (gen_motif_local2) are unaffected.
