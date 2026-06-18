# Onset-Head Recalibration (Stage 3 follow-up)

*Run: 2026-06-18 (seed 42, 96 val songs). Eval-only on the factorized checkpoint — no retraining.*

The factorized onset head was over-confident (a [[pos_weight]] side effect), so
Bernoulli-sampled decoding over-placed. Fixed post-hoc with **per-difficulty Platt
scaling** (calibration — fit `p = sigmoid(a·logit + c)` per difficulty so predicted
probabilities match real frequencies).

## Calibration quality

| difficulty | real density | raw mean p | cal mean p | raw ECE | cal ECE | a / c |
|---|---|---|---|---|---|---|
| Beginner | 0.089 | 0.162 | 0.089 | 0.072 | 0.006 | 1.00 / −1.82 |
| Easy | 0.163 | 0.254 | 0.163 | 0.090 | 0.008 | 1.05 / −2.00 |
| Medium | 0.238 | 0.407 | 0.238 | 0.169 | 0.008 | 1.09 / −2.09 |
| Hard | 0.303 | 0.477 | 0.304 | 0.173 | 0.013 | 0.97 / −1.58 |

[[ECE]] (expected calibration error — gap between predicted confidence and actual
frequency) collapsed ~0.17 → ~0.01. Fitted scale a≈1.0 with bias c≈−1.9 confirms the
over-confidence was a **pure bias offset from pos_weight**, exactly as diagnosed — not
a sharpness problem.

## Decode comparison

| strategy | onset_F1 | prec | rec | density | crit_exact | crit_adj | crit_mae |
|---|---|---|---|---|---|---|---|
| per-diff threshold | **0.756** | 0.759 | 0.776 | 0.201 | 0.406 | 0.802 | 0.854 |
| Bernoulli (raw) | 0.660 | 0.535 | 0.886 | 0.329 | 0.552 | 0.917 | 0.531 |
| **Bernoulli (calibrated)** | 0.656 | 0.659 | 0.670 | **0.201** | 0.448 | **0.906** | 0.667 |
| per-diff threshold (calibrated) | 0.756 | 0.759 | 0.776 | 0.201 | 0.406 | 0.802 | 0.854 |

Real density 0.202.

## Conclusions

1. **Calibration did its job.** It fixed Bernoulli's over-placement (density 0.329 → 0.201,
   matching real) and lifted Bernoulli precision 0.535 → 0.659 (fewer spurious notes) — at
   honest density. onset_F1 held (~0.66). The onset head now outputs trustworthy probabilities.
2. **Calibration is ranking-invariant, so threshold decoding is unchanged.** per-diff
   threshold == per-diff threshold (calibrated): Platt is monotonic, so the top-density frames
   it selects are identical. Calibration only matters for *probability-based* decoding
   (Bernoulli, density-matched cutoffs), which is where it helped.
3. **There's a real F1-vs-fidelity frontier, not a single best decode:**
   - **per-diff threshold** — best onset_F1 (0.756) and precision/recall: deterministically
     picks the top-probability frames, maximizing agreement with the reference chart.
   - **calibrated Bernoulli** — best difficulty fidelity (crit_adj 0.906, exact 0.448) at
     correct density: stochastic sampling preserves each difficulty's onset *character*, which
     the critic rewards, but it stochastically misses some high-prob frames (lower F1).
   These optimize different things — reproducing *the* reference chart vs reading as the right
   difficulty. Calibration removed the confound (over-placement); the remaining gap is inherent
   to deterministic-vs-stochastic decoding.

## Recommendation

- **Apply the per-difficulty Platt params at inference** (the onset head is now calibrated).
- **Default decode = calibrated Bernoulli**: for real generation, faithful difficulty + correct
  density matters more than matching one specific reference chart (many charts are valid for a
  song). Use **per-difficulty threshold** when maximizing onset_F1 against a reference is the goal.
- Possible best-of-both follow-up (not done): a **hybrid decode** — deterministically place the
  highest-confidence onsets, Bernoulli-sample the uncertain middle band — to get threshold's F1
  with sampling's difficulty character.

Code: `experiments/generation_factorized/onset_calibration.py`. The model's `generate()` now
takes `onset_logit_scale`/`onset_logit_bias` for post-hoc calibration.
