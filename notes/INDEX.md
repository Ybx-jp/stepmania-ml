# Notes Index

Map of `notes/`, organized by phase and thread. Findings notes (`*_findings.md`) hold offline/quantitative
results; plans/roadmaps hold forward design; `playtest_log.md` is the subjective play-feel ledger. Arcs are
roughly chronological; the newest (motif → governor → release) is at the bottom.

> **📚 NARRATIVE BRIEFS — start here for the claims audit.** [`briefs/`](briefs/README.md) holds one
> quote-grounded narrative brief per arc, each with an **"audit hooks"** table tracing README claims to
> verbatim evidence + a verb-precision check (measured / vouched / mapped). Built to replace the distrusted
> first-pass `readme-0.1.0-audit.md` (now untracked). The README re-audit reconciles against the briefs;
> the source notes below stay flat (the briefs are an overlay, so no `see notes/X` pointer broke).

**CURRENT STATE (2026-06-26):** deployed model = **`gen_motif_full_fixed`** (42-dim highres, the H19 clean
retrain). Two arcs shipped on top of it: (1) the **H15 motif arc** — steerable section-level candle/trill
figure levers (jack↔sweep is the lone dead axis); (2) the **biomechanical governor** — per-note foot fatigue
(default `fatigue_penalty=2`), per-region stamina + breathing difficulty arc (opt-in), playtest-confirmed "a
tasteful edit, not a rewrite." A Phase-3 mask-predict prototype is de-risked but parked; the deployed track is
the staged AR highres generator. **Now in v0.1.0 release prep** on branch `release/v0.1.0-prep` (LICENSE,
pyproject, README claims audit, bring-your-own-audio `scripts/generate.py`, CHANGELOG). Next-session pointer =
`HANDOFF.md`.

## Phase 1 — difficulty classifier (closed)
- `ordinal_experiment_findings.md` — ordinal vs classification head; `standard_ordinal_multi` won. Phase 1 closed.

## Phase 2 — generative foundation
- `generation_baselines_findings.md` — Stage 1 dumb baselines; onset-alignment is the hard axis (floor F1≈0.05).
- `generation_transformer_findings.md` — Stage 2 AR transformer; sampling unlocks it (greedy collapses).
- `density_calibration.md` — Stage 2 core weakness: knows WHERE notes go (ROC-AUC 0.81) but AR free-run collapses.
- `factorized_head_findings.md` — Stage 3 factorized onset-then-panel head (the big F1 jump 0.30→0.76).
- `focal_onset_findings.md` — focal onset loss lifted the whole frontier (best F1/fidelity).
- `per_difficulty_threshold.md`, `onset_calibration.md`, `hybrid_decode.md` — onset decode/calibration levers.
- `kv_cache.md` — O(T) decode for full-length generation.
- `stage3_roadmap.md` — remaining Stage 3+ ideas (capacity, eval hardening, scope).

## Phase 2.5 — step types / typed model (holds)
- `step_types.md` — typed per-panel representation (tap/hold/tail/roll); rolls absent from data.
- `typed_model_findings.md` — layered typed generator (onset→pattern→type); hold over-gen fixed.
- `hold_aware_decode.md` — per-panel hold automaton (orphans 56%→3%).
- `hold_cross_decode.md` — no_cross_during_hold decode fix (bipedal one-foot-during-hold).
- `playable_samples.md` — playable StepMania folder export.

## Conditioning (control knobs)
- `conditioning_step1_pattern_prefs.md` — decode-time pattern prefs + crossover constraint.
- `conditioning_step2_radar.md` — trained groove-radar profile conditioning.
- `conditioning_step2b_cfg.md` — classifier-free guidance at inference (amplify the steer).
- `conditioning_step3_style.md` — reference-chart style embedding.
- `conditioning_match_findings.md` — matching source groove: radar beats style.
- `numeric_difficulty_conditioning_plan.md` — numeric-difficulty conditioning lessons + plan.

## Musical features (H1/H4/H5 — choreography axis)
- `stage1_musical_features_findings.md` — chroma+HPSS+metric-phase retrain; playtest WIN, metrics blind.
- `feature_retrain_plan.md` — the musical-feature plan (chroma/HPSS/metric phase).
- `h4_offbeat_signal_findings.md` — the off-beat/16th signal is a RESOLUTION problem → high-res onset feature.
- `choreography_metrics_findings.md`, `groove_periodicity_findings.md` — geometry/groove spot-check metrics.

## Stage 2 — realism / taste critic
- `stage2_realism_critic_plan.md` — plan: critic for "taste" + best-of-N + critic-guided fine-tune.
- `stage2a_critic_findings.md` — v1 learned generator-fingerprint (failed gate); v2 corrupted-real critic =
  VALID taste metric (REAL>BASE>CHAOS) + best-of-N reranking.

## Decode / playtest fixes
- `style_decoding.md` — sample the pattern head (greedy = always-Left).
- `playtest_log.md` — **the subjective play-feel ledger** (standing hypotheses H1–H20, newest on top).
- `h11_transitions_findings.md` — AR pattern head drifts through section boundaries (exposure bias).

## CHAOS / 16th-PLACEMENT ARC
Read roughly in this order:
- `chaos_mechanism_plan.md` — keystone synthesis: chaos = where resolution + data + objective converge.
- `chaos_conditioning_findings.md` — chaos-radar conditioning WORKS post-high-res (16th amount 0.3%→26%).
- `selfsim_chaos_findings.md` — self-similarity feature REFUTED for section chaos (R²≈0.06).
- `phase_aware_threshold_findings.md` — decode: calib (variable per-song chaos) vs alloc (smearing quota).
- `chaos_retrain_scope.md` — v6 scope (Step-0: 16th under-commitment = knows-but-loses).
- `v7_additive_loss_design.md` — v6 failed (additive trade); v7 reweighted-BCE additive design + result.
- `chaos_placement_ceiling_SUPERSEDED.md` — ⚠️ **SUPERSEDED** (its "audio-ambiguity ceiling" was refuted by
  the corrected sequence probe; kept as record).
- `sequence_aware_onset_plan.md` — placement is SEQUENCE-determined (AUC 0.935); AR explodes. Critic-guided
  refinement idea recorded.

## Phase 3 — joint generative paradigm (de-risked, parked)
- `phase3_generative_design.md` — the forward plan: joint generative model (diffusion/mask-predict, structured
  transformer, critic objective). Objective + data gates PASSED.
- `phase3_prototype_findings.md` — first build of the mask-predict paradigm (`diag_maskpredict_proto.py`):
  onset mask-and-predict. The de-risk prototype; deployed track stayed the staged AR generator.

## Manifold conditioning + guidance tuning
- `radar_manifold_findings.md` — the 5 radar dims are ~rank-2; manifold-aware steering (conditional-fill +
  ellipsoid project) + source-chart-free density. `src/generation/radar_manifold.py`. (Manifold now SHIPPED
  as `cache/radar_manifold.npz` for dataset-free generation.)
- `h14_guidance_sweep_findings.md` — guidance is per-axis; g=5 overshoots chaos into the H4 smear.
- `h16_harmonic_findings.md` — "harmonic guidance" FALSIFIED vs sampling noise; sweet spot = a KNEE.

## H15 — MOTIF ARC (shipped) — the *which-figures* control axis
- `h15_motif_handoff.md` — the plan: train the pattern head on groove-correlated motifs (the "vibe" lever).
- `h15_motif_findings.md` — ★ Phase-0 gate PASS: note-pattern motifs separate groove≫shuffled; radar pins R²0.61
  of motif usage with ~39% signature-figure residual the radar can't see → motif CODEBOOK lever.
- `h15_local_motif_plan.md` — per-section (incremental sectional) motif conditioning plan.
- `h15_hierarchical_findings.md` — hierarchical pick-then-realize via discrete figure tokens (isolates SWEEP
  that the continuous projection can't).
- `h15_set_characterization.md` — offline characterization of the exported playtest sets (realized figures).
- `repr_integrity_findings.md` — pre-retrain representation audit: found + fixed a note-dropping converter bug.
- `h19_retrain_findings.md` — the clean retrain → **`gen_motif_full_fixed`** (deployed): levers preserved, trill
  honest, sweep improved.
- `note_patterns_and_motifs.md` — **the consolidated home** for note-pattern/motif framing: the pattern
  vocabulary + reference links, the figure codebook, the motif control surface's intent + PARTIAL status, and
  the H20 coverage gap.

## Biomechanical governor (foot fatigue / stamina / breathing arc) — shipped
- `h13_exertion_findings.md` — H13: does the model represent physical exertion / fast-jack cost? (the seed).
- `foot_exertion_findings.md` — the soft, BPM-aware jack governor (decode-time, graded) — the Stage-0 version.
- `foot_fatigue_design.md` — the full per-foot fatigue model spec + every failure (Stages 1–3: per-note foot,
  per-region stamina, breathing difficulty arc). The governor's design bible.
- `governor_release_region.md` — vouched-for per-knob RANGES + shipping defaults. ⚠️ a vouched table, NOT a
  "mapped region" (see [[claim-precision]]); the joint region is a v2 item.

## Geometry / feasible region (parked → v2)
- `difficulty_corner_findings.md` — offline release gate: the EASY difficulty corner is healthy.
- `geometry_feasible_region.md` — the "feasible region of good settings" geometry framing (radar ellipsoid ×
  motif subspace, audio as a constraint); the geometric-DL / map-the-region thread, parked for v2.

## Roadmaps / standing plans
- `augmentation_roadmap.md` — on-the-fly augmentation ideas (mirror, etc.).
- `constraint_relaxation_roadmap.md` — when to relax max-2/variable-BPM/finer-res (data-layer v2).

## Meta
- `HANDOFF.md` — **the next-session handoff** (current = governor complete + v0.1.0 release prep).
- `ml_glossary.md` — ML jargon glossary (gloss-on-first-use; maintained by the ml-gloss skill).
- `archive/` — superseded scratch.
