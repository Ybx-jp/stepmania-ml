# HANDOFF — ACTIVE OPEN: seq-onset fork (strategic); quality-attribution thread CLOSED NULL (3 instruments)

**Written 2026-07-01 for the next Claude.** The quality-feature-attribution thread (incl. its graded-critic
follow-up) is now fully CLOSED; the ONE open ML thread is the seq-onset fork.

## CLOSED THIS SESSION — quality-feature attribution (NULL, three instruments)
*"which audio features drive per-song generator QUALITY under the canonical defaults?"* → **no interpretable audio
feature drives within-difficulty quality** (only axis = coarse difficulty/density). Triangulated on THREE
independent quality instruments (`notes/quality_feature_attribution_findings.md`, lineage
`quality-feature-attribution-arc.md`, [[quality-feature-attribution]]): (1) the deployed realism critic (SATURATES —
~94% of canonical Hard gens railed to "fake", so the deficit = the human chart's score); (2) the validated
choreography distance-to-real (`trans_KL`+`hold_burst`; the one lead was a truncation artifact, refuted
pre-registered); (3) a purpose-built GRADED critic (`experiments/realism_critic/train_graded_critic.py` →
`checkpoints/realism_critic_graded`; retrained non-saturating — gen 0.1–0.9 band 0%→44% — STILL family-wise p=0.12).
- **Reusable WINS:** choreography distance-to-real AND the graded critic are both NON-SATURATING quality
  instruments — use either over the near-binary deployed critic for any fixed-difficulty quality question. A
  "recalibrated critic" via monotonic rescale is a DEAD END (identical ranks); the graded RETRAIN was required.
- Probes (import the harness, match deployment): `probe_quality_features.py` (critic; `--critic` swaps the graded
  checkpoint; holds shared `load_val_dataset`/`build_songs`/`canonical_gen_typed`), `probe_quality_choreo.py`,
  `probe_holdburst_dynamics.py`. Docs landed via **PR #55** (verify state `gh pr view 55`).

## ACTIVE OPEN THREAD — seq-onset fork (A): ALIVE but UNDERTUNED, now STRATEGIC (unchanged since 2026-06-29)
Full state in lineage `seq-onset-arc.md` + `notes/onset_placement_findings.md`. Short version: 16th placement is a
chart-PRIOR not in audio (wall CLOSED NEGATIVE 4 ways); the BUILD re-opened cheap (M1a: conv readout on the FROZEN
decoder's `h` = 0.892 ≡ ceiling); M1b-3 broke the DENSITY drift (scheduled sampling). **THE DECODE SURFACE IS
HEAD-SPECIFIC** (`conditioning-mechanics` §8): for the causal seq head, tau→per-song ADAPTIVE, the 16th-unlock polarity
FLIPS to a down-weight, rests need an EXPLICIT valve. A head-appropriate surface (`seqonset_decode.py`) drains the flood
to a real-aligned backbone that pauses; playtest **"better, still very linear."** The fork is STRATEGIC (right investment
this stage?), not "is it viable." Sharpest untested lead: the head may lean on a HOLD-RELEASE phantom note instead of
genuinely resting. **Do NOT re-derive the "BANKED/placement-hollow" bank — the user overturned it twice (valid catches).**

**Deployed model UNCHANGED across both threads; conditioning, fatigue, stamina INTACT** — every probe was a diagnostic
READ on a frozen decoder. The shipped generator is exactly as last played.

Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); **NOT `conda run`** (it buffers child stdout → empty logs
if killed/polled). Deployed generation ~10 s/chart; the 954-file val PARSE is ~4 min (unavoidable startup); do NOT call
`val_ds.warm_cache()` (it eagerly extracts the whole val set, ~30 min CPU — use lazy `val_ds[i]`).

**READ-FIRST (in order):** for the ACTIVE thread → `notes/quality_feature_attribution_findings.md` → lineage
`.claude/skills/experiment-design/experiment_lineage/quality-feature-attribution-arc.md` → `taste-critic-transfer` memory
(the near-binary caveat + the graded-retrain fork). For the seq-onset thread → `notes/onset_placement_findings.md` →
`seq-onset-arc.md` → `conditioning-mechanics` §8. Load-bearing skills: **experiment-design** (Rules 11 dynamic-range /
12 stratify / 7-9 fair-version-before-committing — this thread is a clean worked example: 3 would-be false "drivers"
each caught), **generation-defaults**, **conditioning-mechanics** §8.

---

## 0. INFRASTRUCTURE LANDED 2026-06-30 (a detour from the seq-onset thread; deployed model UNCHANGED)
A single-source decode refactor — the seq-onset thread above is still THE active ML thread, this is orthogonal
plumbing. The trigger: `scripts/generate.py` (the PUBLIC CLI) had drifted to a *different, un-played* regime
(`pattern_temperature=0.7`, stamina/breathe OFF, no `onset_phase_calib` 16th-unlock). Now:
- `src/generation/decode_defaults.py` = `CANONICAL_DECODE` palette dict + `apply_phase_calib`/`parse_phase_calib`.
- `src/generation/decode_harness.py` = `conditioned_p_onset` (the deployed onset→tau path), `compute_tau`,
  `phase_shares`, `load_generator`, `make_feature_extractor`, `DEPLOYED_CHECKPOINT`, `MODEL_ARCH`.
- `src/utils/data_splits.py` = `discover_chart_files` + `split_chart_files`.
- `generate.py` + `export_typed_samples.py` are DOGFOODED through the harness (a NEW probe should import it, not
  hand-roll the tau pipeline). Also fixed a latent §3 bug in generate.py (tau used a different radar than decode).
- **Verified end-to-end BYTE-IDENTICAL:** migrated `buffered_sectional.py` produces md5-matched charts vs the
  pre-refactor version. See memory `decode-harness-single-source` + the `generation-defaults` skill (updated).
- ⚠️ `--style` now accepts repeated flags AND comma lists (`chaos=high,freeze=low`); but a raw float like
  `chaos=0.7` still gets manifold-PROJECTED (chaos>~0.47 is OOD for Hard → shrunk, and it drags fixed dims like
  freeze back toward the mean). Use on-manifold levels/quantiles to make a fixed dim stick.

## 1. WHERE WE ARE
Deployed model = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor (canonical
block below). No model change this session (the quality thread was diagnostic; the graded critic is a SEPARATE
evaluator model, not the generator). The one OPEN thread = seq-onset (§2b); its tooling (unchanged since 2026-06-29):
- `seqonset_decode.py` — the head-appropriate surface: `build_rest_env` (audio `p_onset` energy envelope = the rest
  valve), `selfcal_tau` (binary-search best-tracking per-song density calibration; defeats the cliff).
- `probe_seqonset_placement.py` (M1b-4, the AUC bracket + pure-TF ceiling head `cache/seqonset_tfceiling_head.pt`),
  `probe_seqonset_critic.py` (M1b-5, density-matched `onset_override` taste-critic A/B), `probe_seqonset_phase.py`
  (M1b-6, phase shares), `probe_seqonset_cond.py` (M1b-7, manifold conditioning), `probe_seqonset_phasepen.py` (M1b-8,
  phase-rebalance), `probe_seqonset_rest.py` (M1b-9, rest structure), `export_seqonset_ab.py` (the by-ear A/B export).
- `probe_seqonset_rollout.py`'s `rollout()` gained opt-in `collect_logits` / `radar` / `phase_pen` / `rest_env`.
The SS head is saved at `cache/seqonset_ss_head.pt`.

## 2a. CLOSED — quality-feature attribution + the graded-critic retrain (lineage `quality-feature-attribution-arc.md`)
Done this session; NULL across three instruments (see the header). The graded critic
(`checkpoints/realism_critic_graded`, `train_graded_critic.py`) exists as a reusable non-saturating instrument but
is NOISY per-chart (ladder monotonicity ~0.35). No open action.

## 2b. ACTIVE OPEN THREAD — the seq-onset STRATEGIC fork + technical leads (lineage `seq-onset-arc.md`)
The binding decision is the user's: **is the seq-onset path the right investment now?** If pursued, the technical
leads in priority order:
1. **Test the HOLD-RELEASE hypothesis** — does the seq head genuinely rest, or use hold-release phantom notes to avoid
   collapse? (Cheap diagnostic: correlate the rests with hold tails.) This decides whether the rest valve is real.
2. **A radar-DIRECT / phase-aware conditioned RETRAIN** — the faithful version of the user's manifold fix (feed groove
   to the seq head as a DIRECT input + train phase-aware so it stops over-firing 16ths at the source). `/autotune`
   first. This is the path to the better-than-audio 16th advantage (M1a 0.89) that free-run doesn't yet reach.
3. **Finer rest texture** (the rests are currently too long/clustered vs real) + tame the density cliff.

**Also-open, INDEPENDENT nearest-shippable** (no longer a fork-A fallback — they stand on their own, governor stack
verified intact): (1) perc-gate `harm_calib` re-A/B (gate on `perc_onset` dim-35 absence, not dim-0 energy — fixes the
HSL piano-solo gate-targeting); (2) 1/16-jack OOD — measure japa1 1/16-jack run-length (`calib_foot_fatigue.py`) BEFORE
a `fatigue_penalty` 2→3 A/B.

## 3. AWAITING USER
The fair-surface A/B is installed at `~/sm-generated/seqonset_ab_fair` (Challenge=audio, Edit=seq fair-surface, +
original). The user PLAYED it: "it's better! still very linear" + the hold-release hypothesis. No open by-ear gate;
the pending decision is the STRATEGIC fork (pursue seq-onset now vs the nearest-shippable vs something else).

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift. Durable mirror
of `generation-defaults` §1; update both (and re-run the validator) on any deliberate change. **This section is
permanent — keep it in every HANDOFF rewrite.** (Unchanged this session.)

<!-- CANONICAL-EXPORT-DEFAULTS:START (do NOT hand-edit values; re-run tools/check_export_defaults.py after a change) -->
```
checkpoint = checkpoints/gen_motif_full_fixed/best_val.pt
features = highres
type_temperature = 0.4
pattern_temperature = 1.0
repetition_penalty = 1.0
max_jack_run = 2
jack_penalty = 0.0
fatigue_penalty = 2.0
fatigue_free = 6.0
stamina_ceiling = 50.0
stamina_tau = 8.0
stamina_scale = 15.0
stamina_breathe = 1.2
onset_phase_calib = 0.0,1.0
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## 4. RESOLVED THIS SESSION (don't re-derive)
- **The decode surface is HEAD-SPECIFIC** — never test a new onset head on the deployed (audio-tuned) palette: tau,
  the 16th-unlock, and rests all break/invert (§ header + conditioning-mechanics §8). This is the session's keeper.
- **The "BANKED / placement-hollow" conclusion was PREMATURE** — three metrics on ONE under-tuned config is not
  robustness (Rule 11), and I skipped the fair version (Rule 7). The user caught it. The path is alive + undertuned.
- **The 16th-flood is NOT chaos conditioning** — `radar=None` throughout (verified). It's the head's free-run fixpoint.
- **Inference-time manifold conditioning is a ~3% echo on the seq head** (it reads `h`, not radar directly) — the
  faithful fix is a radar-DIRECT head, not a decode knob.
- **`onset_override` A/B + `enforce_playability`** is the Rule-14-clean way to compare onset sources (skips stamina for
  both arms = controlled). **`install_to_stepmania` rmtrees a same-named group** — build under `outputs/...`, NOT under
  `~/sm-generated`.
- **Self-cal tau must be a BINARY SEARCH best-tracking realized density** — a quantile-of-realized-logits iteration
  DIVERGES on the cliff (collapsed a song to empty).

## 5. BRANCH / PR STATE
- **Infra refactor (§0), 2026-06-30 — a STACKED branch chain (merge bottom-up, or PR each onto its parent):**
  `refactor/canonical-decode-single-source` → `refactor/harness-tier2-loaders` → `refactor/harness-tier3-evaldata`
  (this refresh's HANDOFF/docs commit sits on the tier-3 tip). All pushed. **Verify live PR state via `gh pr view`.**
- Seq-onset ML thread: docs branch **`docs/seq-onset-placement-banked`** (name predates the correction; PR **#51**
  carries the corrected framing). Prior seq-onset work merged via **PR #50** (merge commit `0d30ee2`). **Verify
  live state: `gh pr view <n>` / `git log origin/main`** (CLAUDE.md Documentation Discipline). `main` protected by `protect-main`.
- Gitignored (not committed): `cache/seqonset_ss_head.pt`, `cache/seqonset_tfceiling_head.pt`, `cache/seqctx_frozenh_*`,
  `outputs/seqonset_ab*`.

## 6. INFRA / PERF NOTES
- Deployed generation ~10 s/chart (B=1 AR, RTX 3060); the seq rollout is lighter. The dataset re-parse (4452 files) is
  the slow part of `probe_seqonset_critic`/`export_seqonset_ab` (~min); the cache-based probes skip it.
- `cache/samples_v3` = the deployed highres feature cache; the `seqctx_frozenh_*` caches are a FIXED split (re-extract
  for a fresh subset — [[dataset-cache-footgun]]).
- `/autotune` (never run) — the right tool BEFORE any seq-onset retrain (batch/AMP/bucketing/Optuna).

## 7. DISCIPLINE (this session is the cautionary tale)
- **Rule 7 (run the fair version before blaming the model) + Rule 11 (isolate the variable / confirm dynamic range):**
  I committed "placement-hollow" from ONE under-tuned config; the fair version (head-appropriate decode surface) showed
  the path is alive. The user caught both. When a result indicts a NEW component, suspect YOUR setup first (HARNESS→DATA→MODEL).
- **By-ear is the binding gate** (Rule 8) — it named the flood AND the "still linear"/hold-release leads the metrics missed.
- **Match the verb to the evidence** ([[claim-precision]]): "alive + undertuned" ≠ "viable/shippable"; "audio-parity
  backbone" ≠ "good". One change at a time; `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.
