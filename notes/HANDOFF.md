# HANDOFF — onset-phrasing arc: by-ear gate SPLIT (fix the gate); seq-aware-onset re-open RESOLVED (wall stands)

**Written 2026-06-29 for the next Claude.** This session was DIAGNOSTIC + decode-probe only — **no model change**.
It (1) got the **by-ear verdict** on the sparse-harm calibrator (SPLIT: japa1 ✓, HSL ✗ = a gate-targeting bug),
(2) let the user reprioritize onto **boundary-snap/structure** and REFRAMED it with two cheap probes (not a clean
gap), and (3) re-opened + **RESOLVED the sequence-aware-onset question** (the wall stands — the latent placement
signal needs a retrain, not a decode lever). Env: conda `stepmania-chart-gen`
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`).

**READ-FIRST (in order):** `.claude/skills/experiment-design/experiment_lineage/onset-phrasing-calibrator-arc.md`
(the active arc) + `seq-onset-arc.md` (the resolved re-open) → then `notes/phrasing_coherence_findings.md`
(by-ear split + boundary reframe), `notes/sequence_aware_onset_plan.md` (the re-open RESULT). Load-bearing skills:
**generation-defaults** (the ONE canonical decode/export config), **conditioning-mechanics** §6 (`onset_logit_offset`
+ the new gate-feature caveat) / §8 (the new WHEN↔WHERE ISOLATION note), **experiment-design** (Rule 0/11 + lineage).

---

## 1. WHERE WE ARE
Deployed model unchanged: `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor.
This session added DIAGNOSTICS only: `probe_phrasing_coherence.py` gained `--quiet_feat {energy,perc}`; new
probes `probe_boundary_snap.py`, `probe_figure_snap.py`, `probe_seqcontext_c0.py`. No `generate()`/export default
changed (the canonical block below is intact). Headline architectural fact (now in cond-mech §8): the decode is
strictly one-way — onset `p_onset` precomputed (audio-only) → stamina thins (ceiling-only) → pattern "where" →
fatigue. **The pattern's "where" never feeds the onset "when"; stamina is the only bridge (suppress-only).**

## 2. THE ACTIVE THREAD — onset phrase calibrator (lineage `onset-phrasing-calibrator-arc.md`)
**Step-2 by-ear gate came back SPLIT.** japa1 PASS ("fun, expressive, not a smear job, well choreographed" → the
sparse-harm mechanism is sound). HSL MEH + the tell: **"the 1/16s came AFTER the piano solo concluded"** = the
pre-registered gate-targeting failure. Cause (code): the `--harm_calib` quiet gate keys on **dim-0 total energy**,
but a piano solo is energy-LOUD + perc-ABSENT → the gate ≈0 during the solo and dumps 35% of its boost on a LOUD
drum section. **FIX = gate on `perc_onset` (dim 35) absence** — already in the probe (`--quiet_feat perc`), NOT yet
in the exporter (`_sparse_harm_offset` still uses dim-0).

## 3. AWAITING USER / OPEN FORKS (none blocked)
- **(A) perc-gate re-A/B (the binding by-ear gate).** TODO: wire `--quiet_feat` into `export_typed_samples.py`'s
  `_sparse_harm_offset` (mirror the probe), regenerate `~/sm-generated/harmcalib_ON` for HSL, user plays: do the
  1/16s now land IN the piano solo? → reads well → STEP 3 (LEARN the per-frame offset); still wrong → retune.
  (Caveat: HSL's harm channel is weak/flat, so the hand-gate is blunt there regardless; the GLOBAL
  `onset_phase_calib` already nails the solo by ear — a different lever.)
- **(B) 1/16-jack OOD (user: "penalize harder, it's definitely OOD").** It's the FATIGUE system. MEASURE japa1's
  1/16-jack run-length vs real (`calib_foot_fatigue.py`) BEFORE tuning, then a `fatigue_penalty` 2→3 A/B (by ear).

## 4. RESOLVED THIS SESSION (don't re-derive)
- **Boundary-snap is NOT a clean targetable gap** (`phrasing_coherence_findings.md` reframe): the REALIZED density
  step tracks real (`probe_boundary_snap.py`; the old "2× wide" was a posterior-envelope + 3-song artifact), and
  real charts barely snap figure-character at Foote boundaries either (`probe_figure_snap.py`, median +0.10, 3/8 neg).
- **Sequence-aware onset re-open — WALL STANDS** (`seq-onset-arc.md`, `probe_seqcontext_c0.py`): controls FIRED
  (audio 0.656≈floor, both_real 0.871≈ceiling); **deployed-C0 context = 0.667 ≈ audio** → the deployed chart
  carries no placement signal beyond audio (its onsets are audio-only-placed). Converges with 06-22's
  train-on-v4-C0 (0.666). The 0.87 signal needs a RETRAIN (sequence-aware head), not a decode lever.

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
- Branch **`gen/full-governor-cond-grid`**; this session's work is in **PR #47** (→ `main`). Related prior PRs:
  **#41** (governor), **#42** (`release/v0.1.0-prep` → `main`). **Do NOT trust any merge/open state written here —
  verify live: `gh pr view <n>` / `git log origin/main`** (per CLAUDE.md "Documentation Discipline"). `main` is
  protected by `protect-main`.
- New cached artifacts (gitignored, not committed): `cache/seqctx_c0_cache.npz` (28 deployed-C0 charts),
  `cache/seqctx_train_cache.npz` (800 real train songs) — so `probe_seqcontext_c0.py` re-runs are instant.

## 6. INFRA / PERF NOTES (cost the session real time — know these)
- **`probe_seqcontext_c0.py` parallelism:** the first cut extracted train features SERIALLY through the stale
  index-cache → a 1-hour 1-core hang. FIXED: parallel `DataLoader(num_workers=4)` + own npz cache + `cache_dir=None`
  (footgun-safe). Use this pattern for any new many-song probe.
- **Batched generation is NOT cleanly supported:** `generate()` treats `onset_threshold` AND `bpm` as batch-scalars
  but each song needs its own (the governor is BPM-coupled). Correct cross-song batching needs `generate()` to
  vectorize those per-song — a deferred, must-be-tested change to the deployed decode path.
- **autotune skill** (user has never run it): the right tool for a future generator TRAINING run (batch/AMP/length-
  bucketing/Optuna), NOT for diagnostics; won't touch a running job. Run it before any retrain (e.g. the seq-aware head).

## 7. DISCIPLINE (load-bearing)
- **experiment-design Rule 11 (confirm the metric can MOVE):** the seq-onset probe's FIRST run trained on 20 songs
  → both_real collapsed to 0.497 (positive control DEAD) → caught + re-run at 800 before reporting. Always check the
  positive control fired before believing a null.
- **Rule 0** (grep notes + lineage + skills BEFORE a probe); **HARNESS→DATA→MODEL**; **by-ear is the binding gate**
  (Rule 8) — the boundary-snap metrics kept coming back ambiguous; ground structure claims in ears.
- **One change at a time;** `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`; arc → lineage.
- **Run `/refresh`** at the next thread/session checkpoint.
