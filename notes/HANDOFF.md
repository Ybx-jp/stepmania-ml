# HANDOFF — ACTIVE: good-settings region (tolerance = f(song features)); diagnostic thread, no model change. seq-onset parked.

**Written 2026-07-03 for the next Claude.** A SHAREABLE-quality milestone (`--style "chaos=0.9,voltage=0.7,air=0.5,freeze=0.5" g=3.0`,
manifold snap — the user "wanted to share these with others") opened a new thread: **map per-song tolerance = f(song
features)** for good decode settings ("a formula to be derived"). This session did the ATTRIBUTION + built the offline
metric anchor; **no model or decode-default change** (diagnostic). Also earlier this session: a small PERF change
(prefetch DataLoader for cold-cache export) on a separate branch (§ BRANCH/PR).

## WHERE WE ARE
Deployed model UNCHANGED = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres). Canonical decode defaults
UNCHANGED (the v0.2.0 `footswitch=False` + `hold_stream_penalty` ship is live; block below). This thread is offline
diagnostics on top of the deployed config.

## THE ACTIVE THREAD — good-settings region (lineage `good-settings-region-arc.md`, memory [[good-settings-region]])
Goal: discover the **matrix of influential song features × their interactions with conditioning** — per-song
**tolerance** = how far a song can be cranked before it leaves the REAL high-chaos phase envelope. Settled so far:
- **The 1/4→1/16 backbone flip under cranked chaos+guidance is NOT the governor** (clean ablation, FULL≡GOV_OFF to the
  digit; `probe_backbone_phase.py`). It's onset-side: CFG-amplified chaos ~70% + 16th-unlock calib ~30%; a phase
  REALLOCATION at held density. CONFIRMS+DECOMPOSES the documented H4/H14 chaos degeneracy (Rule 0: partly re-derived).
- **Rule-5 real anchor** (`probe_real_phase_reference.py`, real Hard n=176): real charts get chaotic by ADDING density
  on a **PRESERVED, better-ANCHORED backbone** (chaos→density +0.68, on_grid 0.99→0.85, s16 bounded ≤0.15, anchoring
  0.41→0.73). → **tolerance = distance from THIS envelope** (on-grid ~0.85, anchoring ~0.73; both →0 in the smear).
  **Anchoring names the H4 defect:** real=anchored coherent runs, generated=unanchored global shift.

### OPEN FORK / binding question
1. **Real-anchored tolerance sweep RUNNING** (`probe_backbone_tolerance.py`, n=40, on_grid/anchoring vs real → feature
   Spearman): does tolerance vary across songs and is it FEATURE-predictable (the "formula")? Report stratified, not
   pooled. Data → `cache/backbone_tolerance.csv`.
2. **The graded critic FAILED as a taste stick** — it measures REALISM and floors on OOD-forced styles (the documented
   H14 limitation; cf [[taste-critic-transfer]]). Do NOT use it to score "good" at the expressive corner.
3. Deeper fix for real-like chaos is the KNOWN conditioning-mechanism (a chaos×onset gate), not decode tuning (H4 §6).

## AWAITING USER
**By-ear ratings of `~/sm-generated/taste_grid`** — 6 cells (chaos {0.2,0.5,0.9} × guidance {1.5,3.0}) × 2 songs
(NIGHT IN MOTION, Grand Chariot), everything else at the milestone spec. **Question: rate each cell 1–5 for
"fun/would-share".** These REFEREE whether the offline metric (on-grid/anchoring, "backbone retained") actually tracks
liking — the whole thread rests on it. Log to a new `notes/goodregion_findings.md` (subjective play-feel → `playtest_log.md`).

## ⚠️ DISCIPLINE — this thread is a CAUTIONARY case study (read before continuing it)
Three metric MISREADS + a pooled claim, each caught by the user's ear (lineage file details):
- **Rule 1/8:** quarter-SHARE then ±1-window quarter-REP both blind to a phase-SHIFTED spine. **DUMP THE ASCII GRID
  FIRST** — one 8-measure onset dump settled what 3 scalar iterations couldn't.
- **Rules 9+12:** I committed "the whole `chaos=0.9,g=3.0` regime is OOD" from **n=2**, POOLING a song-STRATIFIED
  phenomenon against ear evidence. It is **song-DEPENDENT** (user has played several fantastic there) — that is the SUBJECT.
- **Rule 0:** built 2 probes before checking `h4_offbeat_signal_findings.md` / `h14_guidance_sweep_findings.md`.

## PARKED — seq-onset fork (strategic, unchanged since 2026-06-29; lineage `seq-onset-arc.md`)
16th placement is a chart-PRIOR not in audio (wall CLOSED NEGATIVE 4 ways); the cheap frozen-`h` build is ALIVE but
UNDERTUNED. THE DECODE SURFACE IS HEAD-SPECIFIC (`conditioning-mechanics` §8). Playtested "better, still very linear."
Untested lead: hold-release phantom-rest. Not this session.

Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); NOT `conda run`. Deployed generation ~10 s/chart; the
954-file val PARSE is ~2–4 min (unavoidable startup); do NOT `warm_cache()` (eager, ~30 min — use lazy `val_ds[i]`).
Note: redirected probe stdout BLOCK-BUFFERS (per-song prints don't appear live) — check GPU util, not the log, for liveness.

**READ-FIRST (in order):** ACTIVE → lineage `good-settings-region-arc.md` → `notes/backbone_phase_findings.md`
(read its UPDATE block) + `notes/real_phase_reference_findings.md` → `notes/h4_offbeat_signal_findings.md` +
`notes/h14_guidance_sweep_findings.md` (the parent phenomenon). Load-bearing skills: **experiment-design** (this thread
is its cautionary case study), **conditioning-mechanics** §2/§6 (manifold `--style`, CFG on the onset path),
**generation-defaults** §1 (canonical config), **playtest** (the `taste_grid` referee).

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift. Durable mirror
of `generation-defaults` §1; update both (and re-run the validator) on any deliberate change. **This section is
permanent — keep it in every HANDOFF rewrite.**

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
hold_stream_penalty = 8.0
hold_stream_floor = 0.45
hold_stream_win = 16
footswitch = False
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## RESOLVED (don't re-derive)
- **The good-settings thread's phenomenon is documented H4/H14** — chaos = a global off-grid shift the model renders
  uniformly (no local off-beat signal; 16th under-commitment); guidance FLOODS off-beats past a cliff; two retrains
  proved it's NOT a feature problem (fix = conditioning-mechanism/objective). Don't re-derive; EXTEND (per-song cliff).
- **The backbone flip is NOT the governor** (0%, FULL≡GOV_OFF); it's CFG-amplified chaos (~70%) + 16th-calib (~30%).
- **`chaos=0.9,g=3.0` is SONG-DEPENDENT, not OOD as a regime** — some songs there are "fantastic" (ear), others (Deja
  loin) collapse to a 1/16-offset smear. The dependence is the subject.
- **The graded critic is a REALISM detector, floored on OOD-forced styles** (its 0.018 at the loved corner ≠ a taste gap).
- **Prior thread (hold-in-stream fix) SHIPPED + confirmed** (v0.2.0: `footswitch=False` + `hold_stream_penalty`).
  Details in `notes/hold_in_stream_findings.md` + lineage `hold-in-stream-arc.md` + [[hold-in-stream-fix]].

## BRANCH / PR STATE  (verify ALL live state via `gh pr view <n>` / `gh pr list` / `git log origin/main` — Documentation Discipline)
- **PERF (prefetch DataLoader):** commit `6a03753` on branch **`perf/export-prefetch-dataloader`** — a `--prefetch_workers`
  flag (cold-cache export overlap; byte-identical output) + removed the stale pattern_temperature warning + a
  generation-defaults skill note. **Committed, NOT yet pushed/PR'd** (verify via `gh pr list`).
- **THIS refresh (good-settings docs):** on branch **`docs/good-settings-region`** → PR (see `gh pr list`). New notes
  `backbone_phase_findings.md` + `real_phase_reference_findings.md`, lineage `good-settings-region-arc.md`, probes
  `probe_{goodregion_sweep,backbone_phase,real_phase_reference,backbone_tolerance}.py`. Gitignored (reproducible):
  `cache/backbone_*.csv`, `cache/goodregion_smoke.csv`, `~/sm-generated/taste_grid`.
- **Prior:** v0.2.0 shipped (PR #58 merged); hold-in-stream on `docs/hold-in-stream-fix`. `main` protected by `protect-main`.

## DISCIPLINE
Rule 0 (check notes first — it caught an H4/H14 re-derivation here). **By-ear is the binding gate** (Rule 8 — it caught
all three metric misreads; DUMP THE ARTIFACT). Stratify, don't pool (Rule 12). State findings as "under setup X,
observed Y" until the fair test / ear clears them (Rule 9). Match the verb to the evidence ([[claim-precision]]).
`playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.
