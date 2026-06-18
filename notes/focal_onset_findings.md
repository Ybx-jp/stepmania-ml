# Focal-Loss Onset Retraining (Stage 3, upstream lever)

*Run: 2026-06-18 (seed 42). Retrained the factorized model with focal loss on the onset head instead of BCE+pos_weight. 20 epochs, same setup otherwise. Eval: 64–96 val songs.*

Hypothesis (from the decode work): BCE+pos_weight made the onset head over-confident, which
we patched post-hoc with Platt scaling and worked around with decode tradeoffs. **Focal loss**
fights the empty-frame imbalance by down-weighting easy/confident frames rather than blanket-
inflating positives — so it should train a head that's well-calibrated out of the box and lift
the whole F1-vs-fidelity frontier, instead of just sliding along it.

## Headline: the frontier lifted

| model + decode | onset_F1 | crit_adj | crit_exact | crit_mae | onset ROC-AUC |
|---|---|---|---|---|---|
| **focal + per-diff threshold** | **0.748** | **0.927** | 0.458 | 0.615 | 0.945 |
| BCE + per-diff threshold (prior best F1) | 0.756 | 0.802 | 0.406 | 0.823 | 0.950 |
| BCE + calibrated Bernoulli (prior best fidelity) | 0.656 | 0.906 | 0.448 | 0.667 | — |
| focal + calibrated Bernoulli | 0.648 | 0.958 | 0.656 | 0.396 | — |

**focal + per-difficulty threshold gives both** the high onset_F1 of thresholding (0.748 ≈ BCE's
0.756) **and** difficulty fidelity (crit_adj 0.927) that under BCE required sacrificing F1 with
Bernoulli sampling. It strictly dominates BCE+threshold (same F1, +0.12 crit_adj, MAE 0.823→0.615)
for a hair of onset ROC-AUC (0.950→0.945). This is the best operating point in the project.

## Calibration (focal vs BCE, per difficulty)

| difficulty | real density | BCE raw mean p | focal raw mean p |
|---|---|---|---|
| Beginner | 0.089 | 0.162 | 0.139 |
| Easy | 0.163 | 0.254 | 0.202 |
| Medium | 0.238 | 0.407 | 0.279 |
| Hard | 0.303 | 0.477 | 0.316 |

Focal's marginal (overall predicted onset rate) is far closer to the true density — so
**per-difficulty threshold on raw focal probs needs no post-hoc Platt** (perdiff_thresh ==
perdiff_thresh_cal exactly, since thresholding is ranking-invariant). Note focal's Platt fit was
a≈2.8, c≈0.6 — focal made the head *under-dispersed* (probabilities compressed), the opposite of
BCE's over-confidence; sharpening (a>1) is only needed to make Bernoulli work well (raw focal
Bernoulli F1 was low, 0.457 → 0.648 after calibration).

## Conclusions

1. **Focal loss is the better onset objective** — train with it. It lifts the frontier so the
   simple, F1-optimal **per-difficulty threshold** decode also reads as the right difficulty, with
   no post-hoc calibration. Recommended default: **focal model + per-difficulty threshold**.
2. **For maximum difficulty realism**, focal + calibrated Bernoulli (crit_adj 0.958, exact 0.656,
   MAE 0.396) at correct density — calibration (a≈2.8) is needed to sharpen focal's compressed
   probabilities before sampling.
3. Onset ROC-AUC essentially unchanged (0.95): focal didn't improve raw onset *ranking*, it
   improved *calibration/marginal*, which is what made the threshold decode difficulty-faithful.

## Status / remaining levers

Generation is in good shape: onset_F1 0.75 at correct density with strong difficulty fidelity
(crit_adj 0.93), playable .sm export, classifier-critic eval. Decode and onset-objective are now
settled. Remaining upstream levers (see `stage3_roadmap.md`): more capacity / longer training,
KV-cache for full-1440 generation, panel-head improvements (pattern/stream quality).

Reproduce: `python experiments/generation_factorized/train_factorized.py --data_dir data/ \
  --audio_dir data/ --onset_loss focal --epochs 20 --warmup_freeze 3 --batch_size 8`
