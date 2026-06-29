# HANDOFF — onset-phrasing arc; sparse-harm calibrator WIRED, awaiting the by-ear gate

**Written 2026-06-28 for the next Claude.** This session re-homed the "chart lags / under-places at musical events"
problem from a mis-attributed PATTERN-head story onto its true ONSET-head home, built the phrasing-coherence
diagnostic, and validated + wired a **sparse-harm-in-quiet phrase calibrator** (a decode-time onset offset). It
also hardened the infra: a canonical-defaults skill, a dataset cache-bug fix, and an experiment-lineage system +
a `refresh` skill. Env: conda `stepmania-chart-gen` (`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`).

**READ-FIRST (in order):** the arc map
`.claude/skills/experiment-design/experiment_lineage/onset-phrasing-calibrator-arc.md` → then
`notes/phrasing_coherence_findings.md`, `notes/h11_rerun_findings.md`, `notes/arc_lag_findings.md` → the memory
files. Load-bearing skills: **generation-defaults** (the ONE canonical decode/export config), **conditioning-
mechanics** §0/§6/§8 (onset vs pattern responsibilities + the new `onset_logit_offset`), **experiment-design**
(Rule 0 + the lineage directive), **refresh** (run it at the next checkpoint to update all this).

---

## 1. WHERE WE ARE
Deployed model unchanged: `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor.
This session's model-facing change is **decode-time only**: `generate()` gained a per-frame `onset_logit_offset`
(B,T) — the hook the phrase calibrator rides on. The headline conceptual correction (user): **phrasing = WHEN
notes fire = the ONSET head's job; the PATTERN head only decides WHICH panels, never count** (conditioning-
mechanics §0/§8). So the felt "under-placement / lag at musical events" is an onset-head matter.

## 2. THE ACTIVE THREAD — onset phrase calibrator (lineage file has the full chain)
Arc: canonical-defaults consolidation → `arc_lag` (HSL cold-start = AR pattern head; breathe arc zero-phase,
exonerated) → `h11_rerun` (the governor's transition responsiveness is ONSET-SIDE phrasing, not pattern-head
choreography — **attribution twice-corrected** via the governor-off + density-dropped ablations) → phrasing-
coherence diagnostic (4 axes vs MUSICAL EVENTS, objective = the model's OWN coherence, NOT real-chart fidelity) →
the sharpest gap = **quiet-phrase HARMONIC under-placement** (HSL piano solo) → **sparse-harm-in-quiet calibrator**
validated (Step 1, posterior: gain~10 flips corr_harm positive on all 3 songs, held global density = redistribution)
→ **wired into `generate()`/exporter `--harm_calib` + A/B installed (Step 2)**.

## 3. AWAITING USER — the by-ear gate (THE binding question)
A/B installed at **`~/sm-generated/harmcalib_{OFF,ON}`** (HSL + japa1, Hard; ON = `--harm_calib 10`). Global
density held between arms (redistribution, not inflation); critic still Hard. **Question: does HSL's piano solo /
quiet passages now get SENSIBLE, well-reading notes, or does it OVER-allocate (busy/cluttered quiet sections)?**
Log the verdict in `playtest_log.md`. It decides the fork:
- **reads well** → STEP 3: *learn* the per-frame offset from audio (the actual calibrator), and generalize to the
  other diagnostic axes (boundary-snap, clean-tail).
- **over-allocates / wrong feel** → retune first (lower gain; or fix the GATE — the melodic sections that matter on
  HSL may not be the *quietest* frames, so the energy-quiet gate may target the wrong window). Cheap to iterate on
  `probe_phrasing_coherence.py --harm_gain` before any learning.

## 4. INFRA SHIPPED THIS SESSION (know these before the next probe)
- **`generation-defaults` skill** = the ONE canonical config; `export_typed_samples.py`'s BARE defaults now ARE the
  deployed stack (checkpoint→gen_motif_full_fixed, features→highres, onset_phase_calib→"0,1.0", full governor,
  pattern_temp 1.0). `generate()` bare defaults are unplayable/governor-off — never call bare.
- **Dataset cache footgun FIXED** (`notes/cache_index_bug.md`, commit d6bde49): the sample cache was index-keyed
  (no identity check) → subset/`--match` probes read STALE features. Now identity-stamped + verified; safe again.
  (H11 was verified uncontaminated; arc_lag's kneeso-secondary was the only casualty, already discarded.)
- **Experiment-lineage system** (`.claude/skills/experiment-design/experiment_lineage/`) + the **`refresh` skill**
  (runs the whole memory/INDEX/skills/lineage/HANDOFF refresh cycle — invoke at the next checkpoint).

## 5. BRANCH / PR STATE
- This session's work is on **`gen/full-governor-cond-grid`** (renamed from claude/full-governor-cond-grid).
  Committed, **NOT pushed, NO PR yet**. Commits: `26200f2` (canonical-defaults + H11 + diagnostic), `d6bde49`
  (cache fix), `fca697f` (Step 1), `9580b1b` (Step 2), `6258b48` (lineage system + cond-mech refresh), `2617c21`
  (refresh skill), + this handoff.
- Prior arcs: governor SHIPPED (PR #41 merged to `main`). **PR #42** (`release/v0.1.0-prep` → `main`) status not
  touched — CHECK before assuming. `main` protected by ruleset `protect-main`. The 06-27 `pattern_temperature`
  playtest RESOLVED (temp ~1.0 reads coherent → now the deployed default).

## 6. OTHER OPEN THREADS (none block the calibrator)
- **6 experiment-lineage STUBS to backfill** (governor / jack-heaviness / chaos / taste-critic / motif / seq-onset)
  — `experiment_lineage/INDEX.md`; the directive says fill them as those threads next get touched.
- Model UNDER-JUMPS (separate air/density thread; do NOT tune the governor to it — cond-mech §8d).
- best-of-N reranking + V2 region-map; GDL/equivariance — parked (v2/paper).

## 7. DISCIPLINE (load-bearing)
- **experiment-design Rule 0** (now also at the ARC level via the lineage files): grep `notes/` + the lineage
  file + skills BEFORE designing a probe. It saved a cycle TWICE this session (the `onset_override` pattern-head
  isolation was a KNOWN-INVALID setup; the calibrator was already an open lead in `onset_alloc_findings.md`).
  **HARNESS→DATA→MODEL**; run the fair re-test FIRST.
- **conditioning-mechanics:** replicate the canonical config (generation-defaults) for any probe; tau from the SAME
  conditioned+offset logits; governor needs `bpm`. The new `onset_logit_offset` follows the same tau-coupling rule.
- **One change at a time; coherence/play-feel is BY-EAR (Rule 8).** `playtest_log.md` = subjective only;
  quantitative → `notes/*_findings.md`; the arc narrative → the lineage file.
- **Run `/refresh`** at the next thread/session checkpoint to keep all of the above current.
