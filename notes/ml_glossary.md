# ML Glossary

Plain-English glosses for the ML jargon used in this project. Maintained by the
`ml-gloss` skill + hook: each term gets a plain explanation the first time it
comes up, then lands here so it isn't re-explained. New terms get appended at the
bottom under the next date heading.

Format per entry: **term** — plain meaning *(how it shows up here)*.

---

## Seeded 2026-06-17

- **teacher forcing / teacher-forced** — during training (or a clean-context eval), feeding the model the *real* previous step at each timestep instead of its own guess, so one early mistake doesn't snowball. Gives the model's best-case view of "what comes next."
- **autoregressive (AR)** — generating a sequence one item at a time, where each new item is conditioned on the items produced so far (the model reads its own output back in).
- **exposure bias / AR drift** — the failure where a model trained with teacher forcing only ever saw *real* history, so at generation time its own imperfect output drifts into territory it never trained on (here: it emits an empty step, then the all-empty context it now sees pushes it to keep emitting empty → collapse).
- **onset** — in this rhythm-game context, a timestep where a step actually occurs (any arrow pressed), regardless of which arrow.
- **onset posteriors** — the model's predicted probability, per timestep, that a step occurs there. "Posterior" = a probability after the model has looked at the inputs.
- **onset-threshold decoding** — generating by placing a step wherever the predicted onset probability exceeds a cutoff (threshold), instead of sampling. Lets you control *how many* notes directly via the cutoff.
- **cross-entropy (CE)** — the standard loss for classification: penalizes the model by how surprised it is at the correct answer (low when it confidently predicts the truth).
- **softmax** — turns a vector of raw scores (logits) into probabilities that sum to 1.
- **logits** — the raw, pre-softmax output scores of a model, one per class.
- **class weighting / weighted CE** — multiplying the loss for rare classes so the model doesn't ignore them; here used to stop the dominant "empty" step from swamping training.
- **focal loss** — an alternative to class weighting that automatically focuses training on hard, uncertain examples and down-weights easy confident ones — fights class imbalance without blanket-inflating the rare class's probability everywhere.
- **temperature / top-k sampling** — knobs for random generation. Temperature scales how adventurous the sampling is (higher = more random); top-k restricts each pick to the k most likely options.
- **ROC-AUC** — a 0–1 score for how well a probability ranks positives above negatives, independent of any threshold; 0.5 = random, 1.0 = perfect. Good for "does the model *know where*, regardless of cutoff."
- **PR-AUC** — like ROC-AUC but on the precision/recall curve; more informative when positives are rare (compare it to the base rate, not to 0.5).
- **precision / recall / F1** — precision = of the steps it placed, how many were right; recall = of the real steps, how many it found; F1 = their harmonic mean (one number balancing both).
- **macro F1** — F1 averaged equally across classes, so small classes count as much as big ones (the project's main classifier metric).
- **ordinal regression / cumulative link (proportional odds)** — modeling an *ordered* target (Beginner<Easy<Medium<Hard) by predicting "is it past each boundary," instead of treating the classes as unordered. "Proportional odds" is the constrained variant with one shared score; "multi-output" relaxes that.
- **warm-start / freeze / fine-tune** — warm-start = initialize part of a new model with weights from an already-trained one; freeze = hold those weights fixed for a while; fine-tune = later let them keep learning.
- **self-attention / cross-attention / causal mask** — attention = each position looks at other positions and pulls in relevant info. Self-attention = within one sequence; cross-attention = one sequence (the chart) attends to another (the audio); causal mask = forbids looking at future positions so generation stays left-to-right.
- **KV-cache** — a speed trick for autoregressive generation: store each step's attention keys/values so you don't recompute the whole prefix every new token (turns O(T²) decoding into O(T)).

## Added 2026-06-18

- **BCE / binary cross-entropy** — cross-entropy for a yes/no (binary) target; the loss for a single probability like "is there a step here or not."
- **pos_weight** — a multiplier on the positive class in BCE that offsets a lopsided yes/no split (here ~4×, since only ~20% of frames have a step), so the rare "yes" isn't ignored.
- **Bernoulli sampling** — deciding a yes/no outcome by flipping a weighted coin: emit "yes" with probability equal to the model's predicted probability for that frame.
- **non-causal** — attention/processing allowed to look at the whole sequence (past and future), not just earlier positions; valid here for the onset head because the full song audio is known up front (opposite of a [[causal mask]]).
- **factorized head** — splitting one combined prediction into two simpler sub-predictions trained separately (here: onset = "is there a step?" then panel = "which arrows?"), so each can be controlled and calibrated on its own.
