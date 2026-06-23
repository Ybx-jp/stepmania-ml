# Notes Index

Map of `notes/`, organized by phase and thread. Findings notes (`*_findings.md`) hold offline/quantitative
results; plans/roadmaps hold forward design; `playtest_log.md` is the subjective play-feel ledger. Newest
arc (chaos / 16th-placement) is at the bottom and is where the project currently stands.

**CURRENT STATE (2026-06-22):** best playable model = **gen_highres_v4**. Chaos *amount* is a working knob
(calib / radar conditioning); chaos *placement* is bounded in the audio→chart paradigm (see the chaos arc).
Forward direction = **Phase 3** ([[phase3_generative_design]]), gated by two PASSED de-risks.

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
- `playtest_log.md` — **the subjective play-feel ledger** (709L; standing hypotheses H1–H11, newest on top).

## H11 — transitions
- `h11_transitions_findings.md` — AR pattern head drifts through section boundaries (exposure bias).

## CHAOS / 16th-PLACEMENT ARC (current, mostly branch `gen/16th-commit-retrain`)
Read roughly in this order:
- `chaos_mechanism_plan.md` — keystone synthesis: chaos = where resolution + data + objective converge.
- `chaos_conditioning_findings.md` — chaos-radar conditioning WORKS post-high-res (16th amount 0.3%→26%).
- `selfsim_chaos_findings.md` — self-similarity feature REFUTED for section chaos (R²≈0.06).
- `phase_aware_threshold_findings.md` — decode: calib (variable per-song chaos) vs alloc (smearing quota).
- `chaos_retrain_scope.md` — v6 scope (Step-0: 16th under-commitment = knows-but-loses).
- `v7_additive_loss_design.md` — v6 failed (additive trade); v7 reweighted-BCE additive design + result.
- `chaos_placement_ceiling_SUPERSEDED.md` — ⚠️ **SUPERSEDED** (its "audio-ambiguity ceiling" conclusion was
  refuted by the corrected sequence probe; kept as record).
- `sequence_aware_onset_plan.md` — placement is SEQUENCE-determined (AUC 0.935); AR explodes, refinement can't
  bootstrap from v4's anti-correlated C0 → not cheaply exploitable. Critic-guided refinement idea recorded.
- `phase3_generative_design.md` — **the forward plan**: joint generative model (diffusion/mask-predict,
  structured transformer, critic objective). Objective gate PASSED (critic sees placement), data gate PASSED
  (484 multi-charted songs).

## Roadmaps / standing plans
- `augmentation_roadmap.md` — on-the-fly augmentation ideas (mirror, etc.).
- `constraint_relaxation_roadmap.md` — when to relax max-2/variable-BPM/finer-res (data-layer v2).

## Meta
- `ml_glossary.md` — ML jargon glossary (gloss-on-first-use; maintained by the ml-gloss skill).
- `archive/` — superseded scratch (e.g. `next_steps_2-23.md`).

---
### Standing cleanup note
The chaos-arc notes (8 files) are kept granular for provenance; could optionally fold into one arc-summary
later. (Done 06-22: superseded ceiling note renamed; stale next-steps archived.)
