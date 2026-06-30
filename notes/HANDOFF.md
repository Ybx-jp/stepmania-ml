# HANDOFF — seq-onset fork (A) BANKED (cheap build is PLACEMENT-HOLLOW); PIVOT to nearest-shippable governor items

**Written 2026-06-29 for the next Claude.** The seq-aware-onset arc is DONE for now. Prior sessions: the wall is
CLOSED NEGATIVE (16th placement is a chart-PRIOR, not in audio — 4 ways); the BUILD re-opened cheap (M1a: a conv
readout on the FROZEN decoder's hidden state `h` = 0.892 ≡ the note-context ceiling); M1b-3 broke the DENSITY drift
(note-dropout scheduled sampling → free-run run-length 1.0 at real density). **THIS session closed the binding
placement-QUALITY question NEGATIVE and BANKED fork (A): the cheap frozen-decoder + dropout-SS build is
PLACEMENT-HOLLOW.** Density-stability ≠ musicality (experiment-design Rule 1). Confirmed three ways + by-ear:
- **M1b-4** (`probe_seqonset_placement.py`): free-run gen-time 16th-AUC **0.43–0.63 ≤ the deployed audio floor 0.751**
  (vs a re-measured pure-TF conv CEILING 0.839; realized 16th precision **0.04**). The 0.839 representation is
  contingent on REAL notes in context; the head can't bootstrap the 16th prior from its own audio-placed notes.
- **M1b-5** (`probe_seqonset_critic.py`): the realism/taste critic (the fair MUSICALITY gate — refutes "AUC vs one
  reference is too strict") ranks SEQ **0.005 ≪ deployed AUDIO 0.253** on EVERY song (REAL 0.727 ≫ shuf16 0.270,
  control fired), via a density-matched `onset_override` A/B through the DEPLOYED `generate()` (no loop surgery, Rule 14).
- **M1b-6** (user BY-EAR on `export_seqonset_ab.py` → `probe_seqonset_phase.py`): the seq charts read "bland, only
  1/16s"; measured phase shares **19/19/62%** (quarter/8th/16th) vs real **64/32/4** = a self-generated 16th-FLOOD,
  backbone collapsed, with **radar=None (NO chaos conditioning — verified)**. The head INVERTS the rhythm. The by-ear
  gate was LOAD-BEARING: AUC and the critic compressed this to "worse"; the phase-share metric + the ear NAMED it.

**Deployed model UNCHANGED; conditioning, fatigue, and stamina are confirmed INTACT and untouched** — every probe this
arc was a diagnostic READ on a frozen decoder (no model writes). The export-defaults validator is ALIGNED and all
governor knobs are live in `generate()`. The shipped generator is exactly as last played.

Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); **NOT `conda run`** (it buffers child stdout until exit
→ empty logs if the process is killed/polled).

---

## 1. WHERE WE ARE
Deployed model = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor (canonical
block below). The seq-onset work added DIAGNOSTIC probes only; no model change. New tooling this session:
`probe_seqonset_placement.py` (M1b-4 — placement-quality bracket; trains+caches a pure-TF ceiling head
`cache/seqonset_tfceiling_head.pt`), `probe_seqonset_critic.py` (M1b-5 — density-matched `onset_override` taste-critic
A/B), `probe_seqonset_phase.py` (M1b-6 — phase-share measurement), `export_seqonset_ab.py` (the by-ear A/B export);
`probe_seqonset_rollout.py`'s `rollout()` gained an opt-in `collect_logits`. New findings:
`notes/onset_placement_findings.md` (M1b-4/5/6). The M1b-3 SS head is saved at `cache/seqonset_ss_head.pt`.

## 2. THE ACTIVE THREAD — nearest-shippable governor/onset items (fork A banked)
Lineage: `seq-onset-arc.md` (now ✅ BANKED, points 12–14 = M1b-4/5/6) and its parent
`onset-phrasing-calibrator-arc.md`. With fork (A) banked, the ACTIVE work returns to the parent arc's
nearest-shippable items, both on the VERIFIED-INTACT governor stack:
1. **perc-gate `harm_calib` re-A/B (by ear).** The Step-2 sparse-harm-in-quiet calibrator came back SPLIT (japa1 ✓,
   HSL ✗: "1/16s came AFTER the piano solo") because the quiet gate keys on **dim-0 total energy** (a piano solo is
   energy-LOUD + perc-ABSENT → gate fires in the post-solo lull). FIX = gate on **`perc_onset` dim-35 absence**
   (`probe_phrasing_coherence.py --quiet_feat perc` exists; wire `--quiet_feat perc` into the exporter's
   `_sparse_harm_offset`), regen `~/sm-generated/harmcalib_ON` for HSL, user plays: do the 1/16s land IN the solo?
2. **1/16-jack OOD.** Measure japa1 1/16-jack run-length vs real (`calib_foot_fatigue.py`) BEFORE a `fatigue_penalty`
   2→3 A/B (by ear). Per-NOTE governor measured with run-length, NOT raw mass (cond-mech §9).
Play-feel → `notes/playtest_log.md` (subjective only); quantitative → a `notes/*_findings.md`.

**The banked seq-onset escalation (only if revisited, eyes open):** audio-anchor (blend the audio onset logits into
the `h`-readout so placement stays audio-grounded) / true own-output rollout-SS / a joint unfreeze-retrain. All are the
EXPENSIVE path the arc tried to avoid, and the 4-way wall predicts low odds for fine 16th placement. `/autotune`
before any retrain. The SS head + pure-TF ceiling head are cached for re-probing.

## 3. AWAITING USER
The by-ear A/B set is INSTALLED at `~/sm-generated/seqonset_ab` (4 chaotic Hard songs; each `.sm` has Challenge=AUDIO
onsets, Edit=SEQ onsets, + the original). The user PLAYED it and confirmed the seq (Edit) charts are a bland 16th-flood
— this is M1b-6, already recorded. No open by-ear gate for seq-onset. NEXT by-ear gate will be the perc-gate
`harm_calib` re-A/B once wired (item 1 above).

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

## 4. RESOLVED THIS SESSION (don't re-derive)
- **The cheap frozen-decoder seq-onset build is PLACEMENT-HOLLOW** (M1b-4/5/6). Don't re-run the M1b-3 density/run
  gate and conclude the wall is "broken" — it's broken for DENSITY only; placement is re-closed three ways + by-ear.
- **The M1b-4 first run's control FAILED for a HARNESS reason, not the model** (HARNESS→DATA→MODEL): the SS head's
  teacher-forced pass (0.736) is NOT the representation ceiling (SS training trades ~0.10 TF accuracy for
  drift-robustness). Use a PURE-TF conv head (0.839) as the ceiling. Don't bracket placement with the SS head's TF pass.
- **The seq head's 16th-flood is NOT chaos conditioning** — `radar=None` throughout (verified in the rollout cond, the
  `generate()` call, AND the audio baseline). It's the head's intrinsic free-run fixpoint (backbone collapse).
- **`onset_override` is the Rule-14-clean way to A/B a candidate onset source** (audio vs seq) through the deployed
  `generate()` without surgery — it skips stamina for BOTH arms (controlled), runs per-note fatigue + playability.
- **`install_to_stepmania` rmtrees a same-named group:** build the export OUT dir under `outputs/...` (NOT under the
  songs root `~/sm-generated`, or install deletes the source before copying). It copies to `~/sm-generated/<basename>`.

## 5. BRANCH / PR STATE
- This refresh's docs are on branch **`docs/seq-onset-placement-banked`** (off `main`). Prior seq-onset work
  (M1a+M1b+M1b-3) merged via **PR #50** (`docs/seq-onset-frozenh-m1a` → `main`, merge commit `0d30ee2`). **Do NOT
  trust any merge/open state written here — verify live: `gh pr view <n>` / `git log origin/main`** (CLAUDE.md
  Documentation Discipline). `main` is protected by `protect-main`.
- New cached artifacts are gitignored (not committed): `cache/seqonset_ss_head.pt`, `cache/seqonset_tfceiling_head.pt`,
  `cache/seqctx_frozenh_{train,val}.npz`. The `outputs/seqonset_ab` build dir is gitignored too.

## 6. INFRA / PERF NOTES (cost real time — know these)
- **Deployed generation throughput:** ~10 s/chart (B=1 AR) on the RTX 3060; the seq-onset rollout is lighter (onset
  head + pattern argmax, no type head). The dataset re-parse (4452 files) is the slow part of `probe_seqonset_critic`/
  `export_seqonset_ab` (~min); the `*_phase`/`*_placement` probes use the M1a caches and skip it.
- **Stale-cache footgun:** the `seqctx_frozenh_*` caches are keyed to a FIXED split; for a fresh subset always
  re-extract from the current split ([[dataset-cache-footgun]]). `cache/samples_v3` = the deployed highres feature cache.
- **autotune skill** (never run): the right tool BEFORE any seq-onset retrain escalation (batch/AMP/bucketing/Optuna).

## 7. DISCIPLINE (load-bearing — this arc proved it repeatedly)
- **experiment-design Rule 1 (the metric must SEE the property):** M1b-3's density/run-length gate was BLIND to
  placement — it said "wall broken" while the head sprayed 16ths on the wrong frames. The by-ear gate + phase-share
  metric (Rule 8) NAMED the failure the aggregates hid. ALWAYS pair an aggregate with an artifact-level check.
- **Rule 11 (confirm the control fires / the metric can move):** the M1b-4 control failure (SS-TF "ceiling" < floor)
  was caught and fixed (pure-TF ceiling) BEFORE interpreting — don't read a measurement against an invalid bracket.
- **Rule 14 (route through shared infra):** `onset_override` + `enforce_playability`, not a bespoke generate() path.
- **Rule 0** (grep notes + lineage + skills BEFORE a probe); **HARNESS→DATA→MODEL**; **by-ear is the binding gate**
  (Rule 8); **one change at a time;** `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.
