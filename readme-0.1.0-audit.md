# README claims audit — v0.1.0

Traces every significant claim in `README.md` back to evidence (a `notes/*` findings file, source
code, or a test). Done as part of the 0.1.0 release prep so nothing ships that the project's own
record doesn't support. Audited against the README at branch `release/v0.1.0-prep`.

**Legend:** ✅ verified · ✏️ stale/imprecise → corrected in this pass · ⚠️ accurate but worth a note ·
🔧 broken reference → fixed.

## Numeric claims

| # | Claim (README) | Evidence | Status |
|---|---|---|---|
| 1 | Floor onset F1 **0.053** (audio-blind baseline) | `notes/generation_baselines_findings.md` | ✅ |
| 2 | Phase 1: **82.9%** test acc, **0.835** macro F1 | `notes/ordinal_experiment_findings.md` | ✅ |
| 3 | Phase 2 AR (sampled) onset F1 **0.300**, "5.7× the floor" | `notes/generation_transformer_findings.md` (0.300); 0.300/0.053 = 5.7× (derived) | ✅ |
| 4 | "greedy collapses to empty" | `notes/generation_transformer_findings.md`, `notes/focal_onset_findings.md` | ✅ |
| 5 | Factorized focal head onset F1 **0.748**, crit-adj **0.927** | `notes/focal_onset_findings.md` (`focal + per-diff threshold` row) | ✅ |
| 6 | Phase 2.5 layered typed **~0.77**, "holds at the real rate" | `notes/hold_aware_decode.md`, `notes/typed_model_findings.md` | ✅ |
| 7 | Taste critic AUC **0.964**; **REAL 0.823 > BASE 0.290 > CHAOS 0.003** | `notes/stage2a_critic_findings.md` | ✅ |
| 8 | Per-difficulty Platt scaling: onset ECE **~0.17 → ~0.01** | `notes/onset_calibration.md` (Hard: raw 0.173 → cal 0.013) | ✅ |
| 9 | KV-cache **bit-identical**; **33.4s → 3.6s (9.2×)** at 1440 frames | `notes/kv_cache.md` (0/600 timesteps differ; 9.2× at T=1440) + test `test_kv_cache_matches_noncached` | ⚠️ the 3.6s is a **batch of 4** — README omits that |
| 10 | Rolls **0 of 675** charts (not generated) | `notes/step_types.md` | ✅ |
| 11 | "full 1440-frame (~2 min)" generation | `notes/kv_cache.md`, `notes/focal_onset_findings.md` | ✅ |

## Capability / architecture claims

| # | Claim (README) | Evidence | Status |
|---|---|---|---|
| 12 | Factorized head: audio-driven, **non-causal** onset predictor (density immune to AR drift) | `src/generation/factorized.py`, `notes/focal_onset_findings.md`, `notes/hybrid_decode.md` | ✅ |
| 13 | Classifier **warm-starts** the generator's audio encoder; reused as **difficulty critic** | `src/generation/evaluation.py`, `notes/typed_model_findings.md` | ✅ |
| 14 | **Taste critic** reuses the Phase-1 backbone; trained on **corrupted-real** negatives (v1 learned the generator fingerprint, scored backwards) | `experiments/realism_critic/`, `notes/stage2a_critic_findings.md` | ✅ |
| 15 | Controllability: groove-radar conditioning + **CFG** + reference-chart **style transfer** | `src/generation/typed_model.py` | ✅ |
| 16 | Output writes back to `.sm`, re-parses with valid hold spans | `src/generation/sm_writer.py` + `tests/test_generation.py` | ✅ |
| 17 | Decode-time biomechanical **governor** (per-note foot, per-region stamina, breathing arc); "a tasteful edit, not a rewrite" | `src/generation/typed_model.py`, `notes/foot_fatigue_design.md`, `notes/playtest_log.md` | ✅ (added this pass) |

## Qualitative / honesty claims (recalibrated this pass)

| # | Claim (was → now) | Evidence | Status |
|---|---|---|---|
| 18 | "no learned song-structure/climax awareness" → **the model tracks structure, accents, intensity; the open frontier is global whole-song phrase planning** | `notes/playtest_log.md` (H18; "in character with the song"; "comprehend global/local structures"; lines 892–893: flat-density was likely a *decode artifact*) | ✏️ corrected |
| 19 | "the chaos knob still smears off-grid" → **chaos is in-distribution-bounded; musical on-manifold, degrades only past the manifold boundary; 16th-share isn't the dial** | `notes/playtest_log.md` (H18); `notes/radar_manifold_findings.md`, `notes/geometry_feasible_region.md` (Gaussian conditional-fill + ellipsoid projection = deployed path) | ✏️ corrected |
| 20 | Best-of-N reranking listed as a command, **not** a results claim | README "Run it"; guardrail in `marketing/PLAN.md` §6 (reranking built, not playtest-validated) | ✅ (tagged "not yet playtest-validated") |

## Broken / stale references (commands & paths)

| # | Reference (README) | Problem | Fix | Status |
|---|---|---|---|---|
| 21 | Generate example: `--radar "chaos=0.9,air=0.85" --guidance 1.4` | `--radar` is **disabled** and now **errors out** (mean-pin = off-manifold smear). A new user copy-pasting this hits an error. | Use the manifold path: `--style "chaos=q0.9" --guidance 1.5` (exporter help: "pair with `--guidance ~1.5`"; the aggressive validated recipe is `--style "chaos=q0.99" --guidance 3.0`) | 🔧 fixed this pass |
| 22 | "Run it" Phase-2 training surfaces the `gen_stage1`-era lineage | Accurate and the scripts run, but the **deployed** checkpoint is `gen_motif_full_fixed` (42-dim highres + governor), which the Run-it section doesn't surface | Deferred — fold into the HF model-release step (model-card lineage reconciliation), not a 0.1.0 code blocker | ⚠️ noted |
| 23 | All referenced `src/generation/*` files and `experiments/*` scripts | — | All 12 referenced paths exist (verified) | ✅ |

## Net

All headline numbers trace to a findings file; the KV-cache bit-identity claim is backed by a real
test. Three corrections were made to the live README this pass (song structure, chaos framing, the
broken `--radar` command); two items (the batch-of-4 KV-cache nuance and the `gen_motif_full_fixed`
lineage gap) are flagged but deferred to the HF model-release step, where the model card gets its own
lineage reconciliation.
