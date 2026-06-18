# Per-Difficulty Onset Threshold (Stage 3 fix)

*Run: 2026-06-18 (seed 42, 96 val songs). Eval-only on the factorized checkpoint — no retraining.*

Fixes the Stage 3 difficulty-fidelity caveat: a single global onset threshold ignored
that each difficulty has its own density, smearing the signal the [[difficulty critic]]
reads. Per-difficulty real densities span a wide range, so one cutoff can't fit all:

| difficulty | real density | per-diff τ | (global τ) |
|---|---|---|---|
| Beginner | 0.089 | 0.804 | 0.853 |
| Easy | 0.163 | 0.912 | 0.853 |
| Medium | 0.238 | 0.835 | 0.853 |
| Hard | 0.303 | 0.840 | 0.853 |

## Results

| strategy | onset_F1 | prec | rec | density | crit_exact | crit_adj | crit_mae |
|---|---|---|---|---|---|---|---|
| global threshold | 0.756 | 0.761 | 0.778 | 0.201 | 0.281 | 0.771 | 1.010 |
| **per-difficulty** | 0.756 | 0.759 | 0.776 | 0.201 | **0.406** | **0.802** | **0.854** |
| Bernoulli (per-frame) | 0.660 | 0.535 | 0.886 | 0.329 | **0.552** | **0.917** | **0.531** |

Real density 0.202.

## Conclusions

1. **Per-difficulty thresholds are a free win.** Same onset_F1 (0.756) as the global
   threshold, but better difficulty fidelity: crit_exact 0.281 → **0.406**, crit_adj
   0.771 → 0.802, MAE 1.010 → 0.854. Use them as the default decode. The fix cost
   nothing (no retraining, no onset-quality loss).
2. **[[Bernoulli sampling]] is the best for difficulty fidelity** — crit_adj **0.917**,
   exact **0.552**, MAE 0.531 — because sampling each frame at the difficulty-conditioned
   onset head's own probability preserves the per-difficulty *shape* that any single
   threshold flattens. But it costs onset precision (0.76 → 0.54, F1 0.76 → 0.66).
3. **Bernoulli over-places (density 0.329 vs 0.202), revealing an onset-calibration
   issue.** If the onset head were well-calibrated, Bernoulli density would match real.
   The over-placement is the expected side effect of the [[pos_weight]]=3.7 in the onset
   BCE, which inflates predicted P(onset). So the calibration distortion we flagged for
   weighted losses is showing up here, in the onset head specifically.

## Recommendation / next

- **Default to per-difficulty thresholds** (strict improvement over global).
- The best-of-both path: **recalibrate the onset head** (temperature-scale its
  probabilities, or drop pos_weight and rely on focal loss) so its probabilities match
  the true onset rate — then **per-difficulty Bernoulli** should give both the high
  onset_F1 of thresholding AND the high difficulty fidelity of sampling, at correct
  density. That's a small, cheap follow-up (post-hoc calibration; no full retrain).

Code: `experiments/generation_factorized/eval_per_difficulty.py`.
