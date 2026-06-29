# HANDOFF — seq-aware-onset CLOSED NEGATIVE (wall airtight 4 ways); open strategic fork: AR-retrain vs bank

**Written 2026-06-29 for the next Claude.** This session SCOPED the sequence-aware onset retrain, then **bounded it
dead** with two cheap gates (no model change): (1) the M0 own-output **matched refiner** gate (train a note-context
net ON deployed-C0 context) and (2) the user's **analysis-by-synthesis** critic (a likelihood `P(audio|chart)`).
Both NEGATIVE with positive controls FIRED → the 0.87 placement signal is a chart-structural PRIOR, NOT in the audio
in any direction. The retrain question now reduces to ONE strategic fork (below). Env: conda `stepmania-chart-gen`
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`).

**READ-FIRST (in order):** `notes/sequence_aware_onset_plan.md` (the SCOPING + M0 RESULT + ANALYSIS-BY-SYNTHESIS +
4-WAY CONVERGENCE sections — the whole story) → lineage `.claude/skills/experiment-design/experiment_lineage/seq-onset-arc.md`
(hypothesis chain + the methodology wins) → `onset-phrasing-calibrator-arc.md` (the parent active arc, with the
nearest-shippable pending items). Load-bearing skills: **experiment-design** (Rule 11 positive-control — it's the
hero of this session), **conditioning-mechanics** §8 (the WHEN↔WHERE isolation note, now "CLOSED NEGATIVE 4 ways"),
**generation-defaults** (the canonical decode/export config).

---

## 1. WHERE WE ARE
Deployed model UNCHANGED: `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor.
This session was DIAGNOSTIC + cheap-gate only — no `generate()`/export default changed (canonical block below intact).
New tooling (committed): `experiments/generation_typed/gen_train_c0.py` (deployed-C0 generator: fresh-extract → 4 GPU
shards → merge), `probe_seqcontext_matched.py` (M0 matched gate), `probe_recon_audio.py` (v1 regression, VOID),
`probe_recon_critic.py` (the analysis-by-synthesis contrastive critic). New caches (gitignored): `cache/trainfresh_cache.npz`
(800 fresh train charts), `cache/seqctx_trainc0_cache.npz` (800 audio+real+deployed-C0), `cache/seqctx_c0_cache.npz` (28 eval).

## 2. THE ACTIVE THREAD — seq-aware onset, CLOSED NEGATIVE (lineage `seq-onset-arc.md`)
The 0.87 teacher-forced note-context placement signal is a chart PRIOR, NOT recoverable from audio — confirmed FOUR
independent ways, every positive control FIRED:
| direction | result | vs |
|---|---|---|
| forward audio→16th onset | 0.65 | — |
| M0 seq refiner MATCHED (train-on-deployed-C0) | **0.672** | ceiling both_real 0.871 (fired) |
| analysis-by-synthesis critic, real vs CORRUPTED-placement (density held) | **0.570** | ~chance (N=28) |
| analysis-by-synthesis critic, real vs DEPLOYED-C0 | **0.468** | control real-vs-mismatch-song 0.815 (fired) |

⇒ the cheap own-output REFINER is DEAD (C0 carries no placement beyond audio even when trained to read it); the
ANALYSIS-BY-SYNTHESIS likelihood is DEAD for fine placement (audio is coarse/density-compatible only). In
`P(chart|audio) ∝ P(audio|chart)·P(chart)` all placement is the PRIOR → only a chart sequence model can author it.

## 3. AWAITING USER — the strategic fork (binding) + nearest-shippable
- **(STRATEGIC FORK, binding):** **(A) causal-AR head retrain** — the ONLY remaining path to 0.87 (a real Phase-2.6
  build; AR drift is the risk, scheduled-sampling + audio-anchor dampened but did not tame it in the 06-22 de-risk;
  run `/autotune` before any retrain) vs **(B) BANK the bound** (a thesis-grade negative: placement is a chart-prior
  unreachable from audio) and redirect to the nearest-shippable below. The user has NOT chosen yet.
- **(Nearest-shippable, from the parent onset-phrasing arc — unchanged this session):** (1) **perc-gate harm_calib
  re-A/B** — wire `--quiet_feat perc` into `export_typed_samples.py`'s `_sparse_harm_offset` (mirror the probe;
  still dim-0 energy), regen `~/sm-generated/harmcalib_ON` for HSL, user plays: do the 1/16s land IN the piano solo?
  (2) **1/16-jack OOD** — measure japa1 1/16-jack run-length vs real (`calib_foot_fatigue.py`) BEFORE a `fatigue_penalty`
  2→3 A/B (by ear). Log play-feel → `notes/playtest_log.md`; quantitative → a `notes/*_findings.md`.

## 4. RESOLVED THIS SESSION (don't re-derive)
- **Own-output iterative refiner = the 06-22 transfer gate in disguise** — looked re-openable (deployed C0 is
  neutral 0.667 vs v4's anti-correlated 0.456), but the MATCHED train-on-C0 test (M0) lands at 0.672 → the wall is
  C0-INDEPENDENT (any audio-only-placed C0 lacks the signal). This CLOSED 06-28's last residual confound.
- **Analysis-by-synthesis: three probe iterations, two false starts caught by controls** — v1 regression VOID
  (binary chart has no amplitude → worse than predict-mean); v2 critic CONFOUNDED (learned chart-coherence shortcut,
  control at chance 0.517); v3/v4 clean (mismatch-only training forced the audio path; control 0.815, measurement
  0.47–0.57). The mismatch-song positive control was the hero — it separated "audio-grounded" from "chart-prior."

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's
argparse defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift.
This is the durable mirror of the `generation-defaults` skill §1; update both (and re-run the validator) on any
deliberate change. **This section is permanent — keep it in every HANDOFF rewrite.** (Unchanged this session.)

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
onset_phase_calib = 0,1.0
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## 5. BRANCH / PR STATE
- This refresh's docs are on branch **`docs/seq-onset-closed-negative`** (off `main`). Prior: **PR #47**
  (`gen/full-governor-cond-grid` → `main`) and **PR #48** (`release/v0.1.0-prep` → `main`). **Do NOT trust any
  merge/open state written here — verify live: `gh pr view <n>` / `git log origin/main`** (CLAUDE.md "Documentation
  Discipline"). `main` is protected by `protect-main`.
- New cached artifacts are gitignored (not committed): `cache/trainfresh_cache.npz`, `cache/seqctx_trainc0_cache.npz`.

## 6. INFRA / PERF NOTES (cost real time — know these)
- **Deployed-C0 generation throughput:** ~10 s/chart, ~98 frames/s (B=1 AR) on the RTX 3060. 4 PROCESS shards ≈
  1.8× (GPU saturates at 99% with 4 streams — compute-bound, not launch-bound; more shards won't help). **Batched
  `generate()` is FORBIDDEN for C0** — `onset_threshold`/`bpm` are batch-scalars; batching mis-applies one song's
  tau/bpm to all → a confounded C0. Use `gen_train_c0.py`'s shard pattern.
- **Stale-cache footgun (re-bit):** `cache/seqctx_train_cache.npz` (800 rows) is STALE vs the current split (786
  songs / 3820 charts) — row j ≠ valid_samples[j]. Always re-extract FRESH from the current split ([[dataset-cache-footgun]]).
- **autotune skill** (never run): the right tool BEFORE the AR-head retrain (batch/AMP/length-bucketing/Optuna).

## 7. DISCIPLINE (load-bearing — this session proved it twice)
- **experiment-design Rule 11 (confirm the metric can MOVE / the positive control fires):** the analysis-by-synthesis
  v1 (predict-mean floor) and v2 (mismatch-song at chance) NULLs were both CAUGHT as VOID/CONFOUNDED by their
  controls — without them I'd have reported false findings. ALWAYS pair a null with a fired positive control.
- **Rule 0** (grep notes + lineage + skills BEFORE a probe — the refiner was a re-disguised 06-22 gate);
  **HARNESS→DATA→MODEL**; **by-ear is the binding gate** (Rule 8) for the shippable items.
- **One change at a time;** `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`; arc → lineage.
