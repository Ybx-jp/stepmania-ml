# HANDOFF — taste-critic re-validation + interpretability arc (ACTIVE); fatigue governor SHIPPED

**Written 2026-06-27 for the next Claude.** Two arcs are current. The **fatigue/stamina governor is SHIPPED**
(PR #41 MERGED to `main`). The **active thread** is the **taste-critic re-validation + interpretability** work,
which lives on `release/v0.1.0-prep` and is staged for `main` via **PR #42 (OPEN)**. Read this, then the memory
(`~/.claude/projects/.../memory/taste-critic-transfer.md` = active state; `fatigue-governor.md` = the shipped prior
arc). Env: conda `stepmania-chart-gen` (interpreter `/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`).

---

## 1. WHERE WE ARE (one paragraph)
The generator (`checkpoints/gen_motif_full_fixed/best_val.pt`, 42-dim highres) ships with a complete decode-time
**governor** (per-note foot model + per-region stamina + breathing arc; PR #41 merged, default `fatigue_penalty=2`,
stamina/breathe opt-in). This session re-validated the **taste critic** against that current decode stack and then
opened up *what it measures*: (a) the critic's REAL>BASE>CHAOS ranking **transfers** to the current decoder (an
old-machinery control reproduced the historical numbers exactly), but it is a **near-binary separator, not a graded
scorer**; (b) holding the model fixed, the **manifold chaos conditioning** scores far above the old **mean-pin**
flood — so the chaos tastefulness gain is the *conditioning redesign, not the model upgrade*; (c) an
**interpretability** probe (validation-gate-first) shows the critic's "fake" evidence is **off-grid FLOODING** —
removing a bad chart's off-grid notes recovers ~half its score, the on-grid backbone is what "real" rests on, and
sparse off-grid syncopation is *tasteful* (not penalized). Folded into README + marketing. All on
`release/v0.1.0-prep`, staged for main via PR #42.

## 2. IMMEDIATE NEXT (open decisions — do NOT act without the user)
1. **Merge PR #42?** (`release/v0.1.0-prep` → `main`, 18 commits = v0.1.0 release prep + taste-critic
   re-validation + chaos isolation + interpretability arc). This is the live gate to ship everything to main. It is
   a behavior-neutral docs/eval release (no generator default changes beyond what #41 already shipped).
2. **best-of-N viability (the v2 region-map inner judge):** before the critic can rank generated candidates, two
   gates remain — (a) **recalibrate its near-binary low end** (it's a strong separator, weak grader), and (b) a
   **by-ear** check that high-P(real) generations actually play better than low-P(real) for the same setting
   (experiment-design Rule 8). See `notes/taste_critic_transfer_findings.md` "Next gate".
3. **Governor follow-up (still pending from the prior arc):** re-playtest the floored arc endings and pick the
   breathe default (**1.2 vs 1.8**); sets at `~/sm-generated/chaos_stamina_g25_breathe{12,18}`. Pending entry at top
   of `notes/playtest_log.md`.

## 3. THE TASTE-CRITIC ARC (this session — the active work)
Critic = `checkpoints/realism_critic/best_val.pt` (the v2 corrupted-real critic; `LateFusionClassifier`, mean_max
pooling, fusion_dim 256). Always attribute on the **logit margin** `z_real−z_fake`, never the saturated P(real).
- **Transfer** (`experiments/realism_critic/eval_taste_current.py`, mirrors `scripts/generate.py`): REAL 0.823 >
  BASE 0.269 > CHAOS 0.228 (n=64); control = original `eval_taste.py` reproduced 0.823/0.290/0.003 exactly.
  corr(P(real),density) = −0.09 (density-OOD worry cleared). Near-binary (only 14–30% of scores in the middle).
- **Chaos isolation** (`eval_chaos_mechanism.py`, model held fixed): MANIFOLD 0.228 vs MEANPIN 0.028 (73%/song);
  the old mean-pin request still scores ~0.028 on the new model ⇒ the conditioning redesign, not the model.
- **Interpretability** (`critic_saliency.py` gate → `critic_saliency_phaseB.py` → `critic_interpretability.ipynb`):
  - **Phase A gate PASSED + chose the method.** Whole-chart panels-scramble drops the margin +9.85 (cue = arrow
    CONFIGURATION, GLOBAL — a 1/24th local scramble is invisible). **Perturbation/repair attribution localizes a
    known defect (~251×); gradient-IG-from-EMPTY does NOT (1.4×)** — empty baseline measures note PRESENCE, not
    configuration. So Phase B uses perturbation, not IG.
  - **Phase B = H1 CONFIRMED.** Off-grid note frac: only MEANPIN (0.85) vs ~0; removing MEANPIN off-grid notes
    RAISES margin +2.55, removing REAL on-grid backbone tanks it −5.05; corr(margin, off-grid frac) = −0.50. NOT
    off-grid-phobic (REAL's sparse off-grid notes tasteful, removing hurts −0.73).
  - **Phase C** = executed notebook (figures embedded; `outputs/critic_interp_fig{1..5}.png`). Confound caught:
    raw corr(activation, off-grid INDICATOR) is confounded by metric-grid periodicity → use NOTE-conditioned
    contrast (channel #121 fires +10.3 higher on off-grid notes). Phase C corroborative; H1 ablation is causal.
- Findings: `notes/taste_critic_transfer_findings.md`, `notes/taste_critic_saliency_findings.md`. Plan:
  `notes/taste_critic_interpretability_plan.md`. v2 prereq marked DONE in `notes/geometry_feasible_region.md`.

## 4. THE FATIGUE GOVERNOR (prior arc — SHIPPED, pointers only)
PR #41 MERGED 2026-06-26 (`origin/main` head `37257d1`). Decode-time governor in
`LayeredTypedChartGenerator.generate`: per-NOTE foot model (`fatigue_penalty`, default 2), per-REGION stamina
(`stamina_ceiling`, opt-in), breathing arc (`stamina_breathe`, opt-in). The math is the **conditioning-mechanics
skill §8**; spec + every failure in `notes/foot_fatigue_design.md`; vouched per-knob ranges in
`notes/governor_release_region.md`. Diags: `experiments/generation_typed/diag_*.py`. RELEASE CENTER:
`fatigue_penalty=2, jack_penalty=0`, stamina/breathe OFF by default. Mandatory playability is code-enforced via
`enforce_playability`. The governor memory file has the full detail + the failure log.

## 5. BRANCH / PR / PROTECTION STATE
- **PR #41** (`gen/foot-fatigue-stage2` → main) **MERGED** — governor on `origin/main` `37257d1`.
- **PR #43** (`taste-critic-saliency` → `release/v0.1.0-prep`) **MERGED** — interp arc on the release branch.
- **PR #42** (`release/v0.1.0-prep` → `main`) **OPEN** — 18 commits, the live gate to ship the v0.1.0 release +
  taste-critic work to main. `origin/release/v0.1.0-prep` head `db164e8`.
- **`main` protected** by branch ruleset **`protect-main`** (id 18199761): require-PR (0 approvals → solo self-merge
  OK), require conversation resolution, block force-push (`non_fast_forward`), block deletion; **auto-merge OFF**
  on the repo; no bypass actors. Edit: `gh api repos/Ybx-jp/stepmania-ml/rulesets/18199761`. (Solo-repo note:
  do NOT set required approvals ≥1 while single-identity — you can't approve your own PR, it would lock merges.)

## 6. OPEN THREADS (priority; none block PR #42)
1. **best-of-N reranking** = the v2 region-map inner judge — gated on critic low-end recalibration + the by-ear
   pass (§2.2). The interpretability result (critic keys on a coherent musical axis, off-grid flooding) supports
   that reranking-by-P(real) is *principled*, not an artifact.
2. **V2 — MAP THE REAL REGION OF GOOD SETTINGS** (`notes/geometry_feasible_region.md` §V2): joint, song-conditional
   feasible set across ALL conditioning + governor knobs, by sweep × best-of-N × difficulty+taste critics. The
   critic re-validation was its prerequisite (DONE).
3. **H-onset-perc-bias** (onset head under-places on melodic-only sections) — a feature/retrain thread, not a knob.
4. **Model UNDER-JUMPS** (6% vs real 31%) — separate density/air thread; don't tune the governor to it.
5. **GDL/equivariance** (pad L↔R / Klein-four symmetry) — v2 redesign / paper angle, parked.

## 7. DISCIPLINE (hard-won; the two skills are load-bearing)
- **conditioning-mechanics skill** = the exact decode math; replicate `scripts/generate.py` for any probe that
  sets/measures a conditioning or governor knob. **experiment-design skill** = attribution HARNESS→DATA→MODEL;
  validate the method on a known answer BEFORE trusting it (the saliency gate is the canonical example — it
  rejected IG-from-empty and caught the periodic-indicator confound).
- **Match the metric to the property's resolution** (stamina = redistribution, blind to the mean; saliency on a
  saturated critic needs the logit, not P). **One change at a time; isolate with a toggle; validate on a number.**
- **ml-gloss hook active** (gloss ML jargon on first use → `notes/ml_glossary.md`). Keep `playtest_log.md` =
  subjective play-feel only; experiment results get their own `notes/*_findings.md`.
