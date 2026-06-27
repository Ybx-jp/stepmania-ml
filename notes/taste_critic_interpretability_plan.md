# Plan: activation + saliency maps for the taste critic — "what is it measuring?"

*2026-06-26. Motivated by [[taste_critic_transfer_findings]]: the critic transfers to the current decoder and
independently rated the chaos-conditioning redesign as more musical — but it's a near-binary SEPARATOR, and we
have only INDIRECT evidence of WHAT it keys on (training negatives were `panels` = arrow scramble + `shift` =
audio misalignment; the chaos isolation suggests it penalizes off-grid flooding). Goal: pin down the actual
features driving P(real), so (a) we can trust it as the best-of-N inner judge and (b) it becomes a defensible
"here's what taste looks like to the model" artifact for the writeup. Runs under the `experiment-design` skill —
saliency on a saturated classifier is a textbook way to produce a confidently-wrong map.*

## The critic (hook points)
`LateFusionClassifier` (`src/models/classifier.py`): `audio_encoder` → `chart_encoder` → `fusion_module` →
`backbone` (Conv1DBackbone) → `pooling` → `classifier_head` (2-way real/fake). Inputs as scored:
audio `(B,T,23)` + binary chart `(B,T,4)` + mask. Checkpoint `checkpoints/realism_critic/best_val.pt`.

## THE PITFALL THIS PLAN IS BUILT AROUND (experiment-design Rule 1 + 11)
The critic is **near-binary / saturated** (60–77% of scores on the 0/1 rails — [[taste_critic_transfer_findings]]).
**Gradient of the PROBABILITY through a saturated softmax ≈ 0** on exactly the confident examples → a naive
`∂P(real)/∂input` saliency map is near-zero noise where the critic is most decided. Two consequences baked into
the method below:
1. **Attribute on the LOGIT MARGIN (pre-softmax `z_real − z_fake`), never on P(real).** The logit keeps dynamic
   range when the probability has saturated.
2. **Use INTEGRATED GRADIENTS** (path integral from a baseline), not a single raw gradient — IG is the standard
   fix for saturation and gives a completeness guarantee (attributions sum to the logit gap vs the baseline).
   Baseline = empty chart (all-zero panels) for the chart channel; mean/zero audio for the audio channel.

## VALIDATION GATE — run FIRST, believe nothing until it passes (experiment-design Rule 7, the highest-leverage habit)
Before interpreting any map on real cases, prove the method localizes a KNOWN, INJECTED defect — using the
critic's OWN training negative types so we're testing the cue it was actually trained on:
- **panels-local:** take a high-P(real) REAL chart, scramble panels (keep onset count) in ONLY frames `[a,b]`,
  leave the rest intact. Compute IG. **PASS = saliency mass concentrates in `[a,b]`** and `Δlogit` is driven by
  that window. (Mirrors `to_binary`/`panels` corruption from `stage2a_critic_findings`.)
- **shift-local:** roll the chart vs audio by k frames in ONLY a window. PASS = saliency localizes to the
  misaligned window on the AUDIO branch.
- If the maps DON'T localize a known-injected corruption, the saliency is untrustworthy → STOP, fix the method
  (baseline choice, IG steps, logit-vs-prob) before any real-case interpretation. Report this gate result first.

## What to actually build
1. **IG saliency** on the logit margin w.r.t. (a) chart `(T,4)` → per-frame × per-panel attribution; (b) audio
   `(T,23)` → per-frame attribution (sum over the 23 features, and per-feature for the top frames). Sum the chart
   attribution to a per-frame "taste-saliency" time series.
2. **Activation maps:** forward hooks on `audio_encoder`, `chart_encoder`, `fusion_module`, and each
   `Conv1DBackbone` block → capture per-channel activations over time. Compare REAL vs fake: which channels/frames
   fire differently; is there a channel that tracks off-grid (16th-phase) frames or misalignment?
3. **Occlusion cross-check** (cheap, saturation-proof sanity): zero out a sliding frame-window, measure Δlogit.
   Agreement between occlusion-Δlogit and IG is independent corroboration (don't trust a single attribution method).

## Application — connect to the chaos finding
Run the maps on the matched quartet from the isolation: **REAL vs MEANPIN-chaos vs MANIFOLD-chaos vs BASE**, same
songs. Hypotheses to confirm/refute (stated so a result can overturn them):
- **H1:** the "fake" signal concentrates on **off-grid / 16th-offbeat frames** (phase grid `t%4 ∈ {1,3}`, see
  conditioning-mechanics §6). Test: corr(per-frame fake-saliency, off-beat indicator). If MEANPIN's low score is
  driven by saliency on its off-grid flood, that's the mechanism of "stopped flooding = more tasteful," confirmed
  at the input level — and it nails WHAT the critic measures.
- **H2:** the audio branch carries **alignment** weight (the `shift` cue): fake-saliency on audio peaks where
  notes are off-onset. 
- **Null to rule out (Rule 11 / the v1 failure mode):** the critic keys on a **shared global fingerprint**
  (density, a constant offset) rather than localized musical structure — in which case saliency is diffuse/flat
  and the validation gate above would have already failed to localize.

## Discipline checklist (experiment-design)
- **Match deployment (Rule 2):** score exactly as the critic is used — `audio[:, :23]`, binary chart, mask=ones;
  same code path as `score()` in the eval harness.
- **Stratify (Rule 12):** bucket every analysis by difficulty AND by confidence regime (saturated-high / MID /
  saturated-low). The **mid-confidence** examples are where attribution is most informative; don't pool them with
  the rails.
- **Ground in the artifact (Rule 8):** overlay the per-frame saliency on the actual arrows/audio and eyeball
  whether high-saliency frames are musically meaningful (off-beat arrows, misaligned notes) — a clean heatmap is
  not a true explanation until it matches something you can see/hear.
- **State what would change the conclusion (Rule 10):** if IG and occlusion DISAGREE, or the validation gate is
  weak, treat all maps as suggestive only.

## Deliverables
`experiments/realism_critic/critic_saliency.py` (IG + occlusion + activation hooks; reuses the critic loader from
`eval_taste_current.py`), saliency-overlay PNGs into `outputs/`, and a `notes/taste_critic_saliency_findings.md`
with the validation-gate result FIRST, then the REAL/MEANPIN/MANIFOLD/BASE comparison and the H1/H2 verdicts.

## Why this matters beyond curiosity
- **best-of-N gate:** if the critic keys on a coherent musical property (off-grid penalty, alignment), reranking
  by it is principled; if it keys on a fingerprint/density artifact, reranking would select for the artifact.
  This decides whether the [[geometry_feasible_region]] §V2 sweep can trust it as the inner judge.
- **writeup/marketing:** "here is a saliency map of what the taste model looks at" is the concrete,
  show-don't-tell version of the evaluation thesis (`marketing/PLAN.md` beat 7b).

## Threads
- [[taste_critic_transfer_findings]] (the transfer + chaos isolation this explains), `stage2a_critic_findings.md`
  (the panels/shift training negatives = the cues to look for), [[geometry_feasible_region]] §V2 (best-of-N gate),
  conditioning-mechanics §6 (the phase grid for the off-grid hypothesis).
