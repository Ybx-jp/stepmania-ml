# Phase 3 — joint generative chart model (placement distribution) — DESIGN

Follows the [[sequence_aware_onset_plan]] dead-end: placement is sequence-determined (AUC 0.935 given good
context) but NOT cheaply exploitable (AR explodes; refinement can't bootstrap from v4's anti-correlated C0).
The remaining path is a different PARADIGM. This note captures the design space + the de-risk gates.

## Two distinct problems (keep separate)
1. **AMBIGUITY (objective):** the same audio has many valid chartings; point-target CE punishes valid-but-
   different placements -> learns the blurry average -> awkward. Addressed by "train placement DISTRIBUTION".
2. **RELATIONAL coherence + GENERATION (architecture):** placement is sequence-determined, but AR generation
   is unstable and refinement can't bootstrap. Addressed by graph/structured-attention + JOINT generation.
You need BOTH: a graph model with point CE still mushes; a point model with a distribution objective is
representation-limited.

## Objective axis — "placement distribution"
Options to stop punishing valid alternatives:
- **multiple-charting supervision** — reward any of several human charts of the same song.
- **critic / adversarial reward** — the taste critic rewards ANY real-like chart, not one GT (we have it).
- **latent generative** (VAE/diffusion) — captures the distribution implicitly.
GATE RESULTS:
- **OBJECTIVE GATE PASSED** (`diag_critic_placement.py`, 39 chaotic-Hard): critic P(real) REAL 0.844 vs
  v4-gen 0.043 (+0.80) vs **shuf16 0.524** (+0.32; 16ths moved to random 16th frames, amount/backbone/panels
  kept). 92% REAL>gen, 92% REAL>shuf16. **The critic SEES placement quality** (incl. isolated 16th-shuffle)
  -> critic-as-objective is VIABLE. (Caveat: v4-gen 0.043 conflates placement w/ other deficits; the clean
  placement signal is shuf16 +0.32 — solid, moderate.)
- **DATA GATE** (`diag_multichart_count.py`): 484 titles (~9% of 5308) appear in >=2 distinct packs =
  potential independent chartings (415x2, 57x3, 10x4, 2x5). The distribution data EXISTS. TODO: verify they
  DIFFER (independent re-chartings, not copies) before relying on it.

## Architecture axis — graph / message passing / attention (assessed)
- **Message passing for the global-local gap?** In principle yes, but plain SELF-ATTENTION already does it
  (every position attends to every other). Our gap is the small-receptive-field CONV onset head + point
  objective, NOT inability to propagate. So message passing isn't a magic bridge — full attention is the
  simpler version. Don't default to a GNN.
- **Graph attention transformer to steer placement?** Right SHAPE, but for a REGULAR grid (time x 4 panels +
  metric hierarchy) frame it as a STRUCTURED TRANSFORMER: encode structure as positional encodings + attention
  biases (beat-phase, measure-phase, section id, panel relations, relative time). Captures "relate to the
  downbeat / same beat last measure / adjacent panel" = the graph, with a simpler proven arch. A true GNN
  earns its keep ONLY with designed sparse long-range semantic edges (note->section-anchor, ->repeated-phrase,
  ->audio-onset).
- **The key lever (from the refinement result):** generate JOINTLY, not autoregressively — diffusion
  (iteratively denoise the whole chart from noise) or mask-and-predict (BERT-style: mask cells, predict from
  the rest, iterate). Stable (no AR explosion) AND doesn't depend on a good first pass (diffusion-from-noise
  learns a reverse process to the chart manifold) -> sidesteps BOTH failures (AR explosion + refinement
  bootstrap). NOT green-field: our generator is already a transformer w/ cross-attn to audio; this EVOLVES it
  (AR->joint, +structure, +critic).

## Convergent design
A JOINT GENERATIVE chart model (diffusion or mask-predict over the chart) that is: (a) STRUCTURE-AWARE
(metric/panel positional encodings; optional sparse semantic edges), (b) CONDITIONED on audio + groove/
difficulty (steering), (c) trained with a DISTRIBUTION-aware objective (denoising + taste-critic reward).
Addresses all three failures: ambiguity->distribution/critic; global-local->full attention+structure;
AR instability->joint generation.

## Honest cost / risk
Major build (weeks), real uncertainty: discrete-chart diffusion is finicky; ~3800 charts is smallish for a
generative model; "produces MUSICAL placement" is unproven; critic-guided training can be GAMED (cap
iterations + keep playtest in the loop). Phase 3 = a research project, not a fine-tune. Current best playable
stays gen_highres_v4.

## De-risk ladder (cheap gates first — [[experiment-design]])
- [x] Objective gate: critic sees placement -> PASSED.
- [x] Data gate: multi-charting pool exists (484) -> PASSED (verify-differ TODO).
- [ ] Multi-charting DIVERGENCE check: do the 484 independent chartings actually DIFFER (esp. in 16th
  placement)? quantifies the ambiguity + validates distribution data. Cheap.
- [ ] Generation prototype: tiny mask-predict/diffusion on ONSET placement only — does joint denoising-from-
  noise beat refinement-from-bad-C0 (0.666) toward the 0.935 ceiling? The core paradigm test.
- [ ] Then scope the full conditioned, critic-trained generative model.
