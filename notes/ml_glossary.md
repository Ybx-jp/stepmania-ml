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
- **calibration** — making a model's predicted probabilities match real frequencies: if it says 30% across many frames, ~30% should actually be onsets. A model can rank well yet be mis-calibrated (here: over-confident).
- **Platt scaling** — post-hoc calibration that fits a small logistic `sigmoid(a·logit + c)` on held-out data to remap raw scores to honest probabilities, without retraining the model (a = sharpness, c = bias shift).
- **temperature scaling** — the simplest calibration: divide logits by one learned scalar T before softmax/sigmoid (a special case of Platt with only a scale). Higher T = softer/less confident.
- **ECE / expected calibration error** — a single 0–1 score for mis-calibration: bin predictions by confidence and average |predicted confidence − actual frequency| across bins. 0 = perfectly calibrated.
- **greedy decoding** — at each step pick the single highest-probability option (argmax). Reproduces the most likely sequence but collapses to repetitive/degenerate output (here: always Left, jacks). Opposite of sampling.
- **repetition penalty** — at decode, lower the probability of recently-chosen tokens so generation stops repeating itself (here: discourage placing the same arrow pattern as the previous note).
- **conditioning** — feeding a control signal into a generative model so its output obeys it (here: difficulty, and proposed: target groove radar, reference style). The model learns the mapping from control → output during training.
- **classifier-free guidance (CFG)** — a trick to make conditioning *stronger*: train the model both with and without the control signal (randomly dropping it), then at generation push the output away from the unconditioned prediction toward the conditioned one (`out = uncond + g·(cond − uncond)`, g>1). Without it, weak conditioning is often ignored.
- **latent / embedding** — a learned fixed-size vector that compresses something (here: a whole reference chart) into a point the model can condition on. "Latent" = not directly observed, derived by the encoder.
- **bottleneck** — deliberately squeezing information through a small representation (here: pooling a whole chart to one vector) so the model can only keep the gist (global feel) and not memorize/copy the details.
- **mean-pooling** — collapsing a sequence (B,L,d) to one vector per item (B,d) by averaging over time; the simplest way to summarize variable-length input into a fixed-size latent.
- **autoencoder-style conditioning** — at training, the conditioning reference IS the target itself (encode the target → condition on it → reconstruct it). The bottleneck stops trivial copying, forcing the encoder to learn transferable style; at inference you swap in a *different* reference.

## Added 2026-06-22

- **jack** — domain (rhythm-game) term: consecutive presses on the SAME single arrow/panel. At 16th spacing it's one foot hammering fast = physically brutal; real charters break runs by alternating panels (the H13 exertion finding).
- **out-of-distribution (OOD)** — an input or conditioning target the model never saw in training; behavior there is unreliable. Here: cranking one groove-radar dim while pinning the correlated ones requests a combo absent from real data.
- **manifold** — the lower-dimensional surface that real data actually occupies inside the full space. The 5-dim groove radar really lives on a ~2-dim surface (one intensity axis + freeze); "on-manifold" = realistic combo, "off-manifold" = a combo real charts avoid.
- **covariance / correlation matrix** — how dimensions move together across the data. Covariance is the raw co-variation; correlation normalizes it to [−1, 1] (here: stream↔chaos r=0.80 means they rise together). The basis for the [[manifold]].
- **Mahalanobis distance** — distance from a point to a distribution that accounts for its spread and correlations (roughly "how many typical std-devs away, along the data's own axes"), unlike plain straight-line distance. Our "how unusual is this groove combo" meter.
- **Gaussian conditional / conditional expectation E[·|·]** — fit a multivariate normal, fix some variables, and read off the expected values of the rest (`E[free | fixed]`). Used to auto-fill the radar dims the user didn't set, coherently with the ones they did.
- **projection (onto the manifold / covariance ellipsoid)** — snapping an off-[[manifold]] point back to the nearest plausible on-manifold point (shrink along the [[Mahalanobis distance]] ray) so contradictory requests still cohere.
- **rank / low-rank** — the number of genuinely independent axes of variation. "Rank-2" = despite 5 named radar knobs, there are effectively only ~2 free directions (intensity + freeze); the rest are correlated.
- **k-nearest-neighbors (k-NN)** — find the k closest real examples to a query point; here, the nearest real charts to a target groove, to check whether that combo actually exists.
- **PCA / factor** — finding the few orthogonal axes that capture most of the variation in correlated data (an alternative way to turn the 5 coupled radar dims into ~2 independent steering coordinates).
- **manifold hypothesis** — the idea that real high-dimensional data (here, "good charts") actually clusters on a much lower-dimensional surface ([[manifold]]) inside the full space; generating well = staying on that surface.
- **feasible region** — the set of conditioning settings that all yield good output; here the in-distribution core where radar/motif/audio constraints are simultaneously satisfiable. Its boundary is where charts stop being coherent.
- **product of experts** — combining several soft constraints by requiring all of them at once (the good region = where every "expert" — radar ellipsoid, motif subspace, audio feasibility — is satisfied), rather than averaging them.
- **geometric deep learning (GDL)** — the subfield about building networks that respect a domain's symmetries/structure (graphs, groups, manifolds) via equivariance; distinct from merely *having* a latent [[manifold hypothesis|manifold]].
- **equivariance / equivariant** — a network property where transforming the input produces the correspondingly-transformed output (e.g. mirror the pad L↔R → the chart mirrors too), so the symmetry is baked into the architecture instead of patched in afterward.
- **symmetry group (dihedral / Klein-four)** — the set of transforms that leave a structure's "rules" unchanged; the dance pad's L↔R (and U↔D) mirrors form such a group, giving the 4 panels a small dihedral-like symmetry.

## Added 2026-06-27

- **ablation** — a controlled "remove (or swap out) one component and measure the drop" experiment, to attribute how much that component actually contributes; here, comparing the learned pattern head against a foot-physics-only generator on identical onsets isolates what the learned head adds.
