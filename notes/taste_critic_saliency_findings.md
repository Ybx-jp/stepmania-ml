# Taste-critic saliency — findings

*2026-06-26, branch `taste-critic-saliency`. Executing [[taste_critic_interpretability_plan]] under the
experiment-design skill (saliency on a saturated classifier = easy to get a confidently-wrong map). Harness:
`experiments/realism_critic/critic_saliency.py`. This note grows as phases complete; the validation gate is first.*

## PHASE A — the validation gate: PASSED (for perturbation/repair attribution)
Goal: before interpreting ANY real chart, prove the attribution method localizes a KNOWN injected defect
(panels-scramble = the critic's own `panels` training negative), on high-margin REAL charts.

**Results (6 highest-margin REAL val charts, logit margin z_real−z_fake):**
1. **Cue exists, strongly.** A whole-chart panels-scramble (keep onset count, permute which panels) drops the
   margin by **+9.85** (min +9.59, max +10.16). The critic robustly keys on **arrow-choice configuration** —
   confirming directly what `stage2a_critic_findings` inferred from the training setup.
2. **The cue is GLOBAL, not frame-local.** A 32-frame (~1/24th) local scramble barely moved the margin (Δ≈0);
   a contiguous 50% window dropped it ~+7. Consistent with the critic's mean/attention pooling over the
   sequence — a small local defect is washed out by the pooled score.
3. **Perturbation/repair attribution is VALIDATED; gradient-IG-from-empty is NOT.**
   - **Block-repair** (restore a block corrupt→clean, measure margin recovery — saturation-proof, on the logit):
     localizes the injected 50% window at **mean 251×** (several are "perfect" = zero recovery outside the
     window, capped at 999; partials 7.5–8.4×).
   - **Integrated gradients from an empty-chart baseline**: only **1.4×**. Principled reason, not a bug: an empty
     baseline makes IG measure note **presence vs absence**, but the critic's cue is **which panels at fixed
     count** (configuration). IG-from-empty highlights all note frames, not the mis-chosen ones.
   - **Decision:** Phase B uses **perturbation/repair (occlusion-style) saliency**, which asks the critic's own
     question (change the arrows, watch the margin). (A clean-baseline IG would localize trivially — `input−base`
     is zero outside the changed window by construction — so it's not an independent check; perturbation is.)

**Method takeaways for any future critic probe here:** (a) attribute on the **logit margin**, not P(real)
(saturation); (b) for a critic whose signal is CONFIGURATION at fixed density, use **perturbation** attribution,
not gradient-from-empty; (c) size the perturbation to the critic's scale — it pools globally, so frame-local
defects are invisible.

## PHASE B — what the critic measures on real generations (PLANNED, not yet run)
Using the validated perturbation/repair saliency, run on the matched quartet from the chaos isolation
(REAL / MEANPIN-chaos / MANIFOLD-chaos / BASE, same songs). Hypotheses (overturnable):
- **H1 (off-grid):** the critic's "fake" signal concentrates on off-grid 16th-offbeat frames (phase grid
  `t%4 ∈ {1,3}`). Operationalize as a **per-frame scramble-saliency** (scramble each block's panels at fixed
  count, measure Δmargin) and correlate with the off-beat indicator. If MEANPIN's low score is driven by
  saliency on its off-grid flood, that explains the chaos finding at the input level.
- **H2 (alignment):** audio-branch perturbation (local shift) carries weight where notes are off-onset.
- **Null to rule out:** the score is a global density/fingerprint artifact (then per-frame saliency is diffuse).

## Threads
[[taste_critic_transfer_findings]] (the chaos isolation Phase B explains), [[taste_critic_interpretability_plan]]
(the full plan), `stage2a_critic_findings.md` (the panels/shift training cues), conditioning-mechanics §6 (phase grid).
