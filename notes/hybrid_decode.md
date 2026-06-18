# Hybrid Onset Decode (Stage 3 follow-up)

*Run: 2026-06-18 (seed 42, 96 val songs). Eval-only on the calibrated factorized checkpoint — no retraining.*

Tests whether a hybrid decode beats the F1-vs-fidelity frontier from `onset_calibration.md`.
Using calibrated per-difficulty onset probabilities and the density-matching threshold τ_d,
each frame: `p > τ_d+m` → onset, `p < τ_d-m` → no onset, else Bernoulli(p). Margin `m`
sweeps m=0 (pure threshold) → large m (pure Bernoulli). Injected via `generate(onset_override=...)`.

## Results

| margin m | onset_F1 | prec | rec | density | crit_exact | crit_adj | crit_mae |
|---|---|---|---|---|---|---|---|
| 0.00 (=threshold) | **0.757** | 0.759 | 0.776 | 0.201 | 0.417 | 0.812 | 0.823 |
| 0.05 | 0.756 | 0.757 | 0.776 | 0.202 | 0.354 | 0.823 | 0.865 |
| 0.10 | 0.744 | 0.744 | 0.764 | 0.204 | 0.406 | 0.781 | 0.885 |
| **0.20** | 0.721 | 0.717 | 0.744 | 0.205 | 0.479 | 0.865 | 0.688 |
| 0.50 (~Bernoulli) | 0.662 | 0.665 | 0.676 | 0.201 | 0.500 | 0.896 | 0.615 |

Real density 0.202 — held correct across all margins (calibration + density-matched τ).

## Conclusions

1. **The hybrid is a clean, tunable dial along the frontier — not a free lunch.** onset_F1
   declines monotonically with m (0.757 → 0.662) while difficulty fidelity broadly improves
   (crit_adj 0.812 → 0.896, MAE 0.823 → 0.615). No single m matches threshold's F1 AND
   Bernoulli's fidelity at once — the frontier is real, and the margin just lets you pick where
   to sit on it. Density stays correct throughout.
2. **m ≈ 0.20 is the best-balance operating point.** It keeps most of threshold's onset_F1
   (0.721 vs 0.757) while gaining most of Bernoulli's fidelity (crit_adj 0.865 vs 0.812;
   crit_exact 0.479 vs 0.417; MAE 0.688 vs 0.823). A ~3.6-pt onset_F1 cost for a clear
   difficulty-fidelity gain.
3. Fidelity at small m (0.05, 0.10) is noisy — the critic metric wobbles at 96 songs. The
   endpoints (threshold, Bernoulli) and m=0.20 are the trustworthy reference points.

## Recommendation

The decode question is settled — pick by goal, all at correct density on the calibrated head:
- **max onset_F1** (reproduce a reference chart): per-difficulty threshold (m=0).
- **balanced default**: hybrid m≈0.2.
- **max difficulty realism**: calibrated Bernoulli (large m).

Decode tuning has hit diminishing returns. Bigger remaining levers are upstream (model/training),
not decode: see `stage3_roadmap.md` — focal-loss retraining of the onset head, more capacity /
longer training, KV-cache for full-1440 generation.

Code: `experiments/generation_factorized/hybrid_decode.py`. `generate()` now accepts
`onset_override` (B,T bool) so any onset policy can be computed externally and injected.
