# Ordinal vs Classification Head — Findings

*Run: 2026-06-17 (seed 42, 20 epochs, identical splits across variants). Test set: 871 samples.*

## Setup

6-way comparison: `{standard, contrastive} × {classification, ordinal (scalar), ordinal_multi}`.

- **classification**: softmax + CrossEntropy (baseline)
- **ordinal (scalar)**: proportional-odds cumulative link — single scalar score minus learned thresholds, BCE on cumulative logits
- **ordinal_multi**: independent cumulative logits (K-1 outputs), BCE with monotone target encoding — removes the scalar bottleneck

Hypothesis (as originally framed): ordinal head reduces adjacent-class errors, target <15% adjacent-misclassification rate.

## Results (test set)

| Variant | Adj.Misclass% | Accuracy | Macro F1 | MAE |
|---|---|---|---|---|
| **standard_ordinal_multi** | **16.5%** | **0.829** | **0.835** | **0.177** |
| contrastive_classification | 17.7% | 0.814 | 0.819 | 0.195 |
| standard_classification | 18.3% | 0.812 | 0.819 | 0.194 |
| contrastive_ordinal_multi | 35.0% | 0.626 | 0.603 | 0.400 |
| standard_ordinal (scalar) | 49.5% | 0.442 | 0.306 | 0.623 |
| contrastive_ordinal (scalar) | 48.6% | 0.441 | 0.307 | 0.636 |

Plots: `outputs/ordinal_experiment/{metric_comparison,adjacent_misclass_rate,confusion_matrices_2x2}.png`

## Conclusions

1. **Winner: `standard_ordinal_multi`** — best on every metric (82.9% acc, 0.835 macro F1, lowest MAE and adjacent-error rate). New Phase 1 best, beating the prior contrastive baseline (~81.4% val).
2. **The scalar proportional-odds head collapses** (~44% acc, ~49% adjacent errors) in both standard and contrastive settings. Early-stopping fired at epoch 2 — it never trains. The single-scalar bottleneck + tight init is the culprit, not ordinal modeling itself. **Do not use the scalar ordinal head.**
3. **Ordinal structure helps only without the bottleneck.** `ordinal_multi` modestly beats classification (16.5% vs 18.3% adjacent, +1.7pp acc, +1.6pp F1). The original "ordinal beats classification" hypothesis is true *only* for the multi-output formulation.
4. **Contrastive added nothing here** — flat for classification, and it actively hurt `ordinal_multi` (35% vs 16.5%).
5. **Nobody cleared the <15% adjacent-error bar**, but `ordinal_multi` came closest at 16.5%. The threshold was aspirational; treat 16.5% as the current ceiling for this data/architecture.

> Note: `compare.py`'s printed "HYPOTHESIS VERDICT" only contrasts classification vs the *scalar* ordinal head (the original 2×2 framing), so it reports "classification wins by ~31%." That verdict is stale now that `ordinal_multi` is in the mix — read the table, not the verdict line.

## Decision

Phase 1 is closed. Best model = `standard_ordinal_multi` (`checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt`).
Graduating to Phase 2: generative chart creation via an **autoregressive transformer** conditioned on audio features + target difficulty, reusing the Phase 1 audio encoder backbone. See [[phase2-generative-plan]].
