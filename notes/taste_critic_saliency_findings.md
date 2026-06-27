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

## PHASE B — what the critic measures on real generations: H1 CONFIRMED (off-grid FLOODING is the fake evidence)
`critic_saliency_phaseB.py`, the matched quartet from the chaos isolation (REAL / BASE / MANIFOLD-chaos /
MEANPIN-chaos, same model, n=12 songs). Primary tool = **phase ablation** (remove all off-grid vs all on-grid
notes, measure margin change — the validated perturbation family, no clean reference needed).

| rung | logit margin | off-grid note frac | Δm (remove off-grid) | Δm (remove on-grid) |
|---|---|---|---|---|
| REAL | +1.91 | **0.02** | **−0.73** | −5.05 |
| BASE | −2.36 | 0.00 | +0.00 | −0.41 |
| MANIFOLD | −1.90 | 0.00 | +0.00 | −0.86 |
| MEANPIN | −5.27 | **0.85** | **+2.55** | −0.01 |

- **Only the mean-pin flood is off-grid** (off-grid note fraction 0.85 vs ~0 for REAL/BASE/MANIFOLD) — the
  conditioning-mechanics prediction, confirmed: mean-pin smears onto 16th-offbeats, the manifold stays on-grid.
- **The off-grid flood IS the fake evidence.** Removing MEANPIN's off-grid notes RAISES its margin by **+2.55**
  (recovers ~half the gap to BASE); removing its on-grid notes does nothing (−0.01).
- **The on-grid backbone is what "real" rests on:** removing REAL's on-grid notes tanks its margin **−5.05**.
- **cross-chart corr(margin, off-grid fraction) = −0.50** — more off-grid → more fake.
- **NUANCE (n=12 surfaced it): the critic is NOT off-grid-phobic.** REAL carries a few off-grid notes (frac
  0.02) and removing them HURTS (−0.73) — sparse syncopation is *tasteful*. So the learned rule is "coherent
  on-grid backbone + tasteful sparse off-beats, penalize off-grid FLOODING," not "off-grid = bad." This is the
  mechanistic, input-level version of the README's "felt chaos peaks mid-range, not at the flood."

**This explains the chaos isolation ([[taste_critic_transfer_findings]]) at the input level:** mean-pin scores low
*because* the critic's fake evidence sits on the off-grid flood it produces; the manifold scores higher *because*
it refuses to flood (stays on-grid). The taste critic and the conditioning redesign agree on the same axis.

- **H2 (alignment)** and the **block-scramble spatial map** were not decisive here: off-grid content is
  concentrated in a single rung (MEANPIN), so within-chart block contrast is near-zero (the map reported n/a for
  the on-grid rungs). The phase ablation was the decisive tool. H2 (audio-shift saliency) is left for a follow-up
  on deliberately-misaligned charts.
- **Null ruled out:** the signal is localized to a musically-meaningful axis (on/off-grid), not a diffuse
  density/fingerprint artifact (a diffuse-artifact critic would show ~0 ablation contrast).

## PHASE C — activation maps (NOT yet run)
The other half of the user's ask: forward-hook the `audio_encoder`/`chart_encoder`/`fusion`/`Conv1DBackbone` and
look for a channel that tracks on/off-grid or alignment — to corroborate the input-level H1 finding at the
representation level. Deferred; the input-level evidence above already answers the headline question.

## Threads
[[taste_critic_transfer_findings]] (the chaos isolation Phase B explains), [[taste_critic_interpretability_plan]]
(the full plan), `stage2a_critic_findings.md` (the panels/shift training cues), conditioning-mechanics §6 (phase grid).
