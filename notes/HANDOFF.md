# HANDOFF — ★ SHIP MODE PAUSED: taste-critic arc UN-PARKED. Active = decode-fixing the 3 defects + a taste-aligned critic.

**Written 2026-07-12 for the next Claude.** The "ship v1.0.0" directive is **PAUSED** (the taste-critic thread was
un-parked 2026-07-11; see [[ship-mode-park-research]]). v1.0.0 is DEFERRED, not cancelled; the OTHER parked paths
(GDL, seq-onset retrain, good-settings formula) stay parked. **This session SHIPPED the UNIVERSAL sub-train-length
onset window as a canonical DEFAULT (by-ear PASSED) — the first canonical-decode change of the arc.**

## WHERE WE ARE
- **Deployed model UNCHANGED:** v2 48th-grid `gen_motif_v2_48th_cont` + `--features highres_v2`, both CLIs default to it.
- **★ CANONICAL DECODE CHANGED THIS SESSION (2026-07-12):** the UNIVERSAL sub-train-length onset window is now a
  DEFAULT — `decode_defaults.UNIVERSAL_ONSET_WINDOW=3600`, both CLIs default v2 → 3600 (exporter `--onset_window`
  default 3600 gated subdiv!=4; `generate.py --onset_window auto`→3600 + `--onset_tail_hangover auto`).
  `check_export_defaults` now 27 ✓ (added `onset_window`/`onset_tail_hangover`). v1 + short-fit (<3600 frames) songs =
  byte-identical no-op; disable via `--onset_window 0`. (The 07-11 HANGOVER PAD is now subsumed — it's the end-centering
  half of the same window.)
- **Goal (user's words):** f48 raised quality VARIANCE — fix = SELECTION (a critic picks the best of N conditioning
  variants), plus DECODE-FIXING the concrete defects that make gens feel off. The 3 defects ARE the critic's negative
  target list.
- **This session (2026-07-12) SHIPPED the universal window** (`notes/universal_window_findings.md`, `playtest_log.md`
  2026-07-12; probes `probe_universal_window{,_decoded}.py`):
  - **Premise MEASURED (exp-design Rule 5/6):** v2 train-len median 3120 / p75 3648 / MAX 5128; abs-PE exposure
    collapses to 31%/13%/6% by pos 3500/4000/4320. So any song >~3500 sat its END in the under-trained abs-PE tail,
    yet `onset_window` was pinned at V2_MSL=5400 → windowing NEVER fired for T≤5400 → short songs' ends collapsed like
    long songs'.
  - **Onset probe (RIGHT population — cached VAL, human chart = ground truth, n=60/band):** single-pass fires only
    **30% of real TAIL notes** on the 3800–5128 band + tail backbone Herfindahl 0.61→0.34; W3600 restores recall +
    Herfindahl to the HUMAN value; CONTROL (<3000) byte-identical no-op; **W4320 ~no-op** (fires only T>4320, past the
    ~3500 onset) = the sharpest proof it's an abs-PE effect. Decoded: windowed tail quarter% 33–69 vs single-pass's
    collapsed 4–8%, tail jitter 0.
  - **BY-EAR A/B PASSED — windowed won on all 3 songs** ("great"/"better"/"fine"). → SHIPPED as default.
  - **NEW residual [H-winddown]** (SEPARATE, pre-existing on BOTH arms): neither winds down into a silence/outro —
    candidate = the window restores tail p_onset peaks → the stamina breathe arc thins the outro LESS (queued probe,
    do NOT build blind). Bland choreography on 2/3 = per-song CONDITIONING (user's read) → feeds the best-of-N track.
  - **PRIOR (still current):** #2 harm_calib PASSED (DENSITY-PRESERVING trade); empty-MIDDLES = global-tau allocation
    (local-tau SHELVED, Rule-13 quota); #3 free-foot-overload during a hold still PARKED (§5b).

## ★ ACTIVE THREAD — taste-critic-quality arc (lineage `experiment_lineage/taste-critic-arc.md`)
Two complementary tracks; the defects feed the critic (they ARE its negative targets):
- **(A) Decode-fix the 3 defects.** #2 harm_calib = PASSED (density-preserving trade, documented — don't stack gains
  blindly). #1 tail COLLAPSE = FIXED (hangover) + **short-song END-degeneration = FIXED + SHIPPED (universal window
  W3600, by-ear PASSED 2026-07-12)**; the empty-MIDDLES half is OPEN (density allocation, local-tau shelved); NEW
  **[H-winddown]** outro-taper lead (pre-existing, queued probe). **#3 (free-foot-overload during a hold) = NEXT-UP**
  (flipped from PARKED 2026-07-12; `footspeed_floor_findings.md §5b`).
- **(B) Phase 2: taste-align the critic (R3)** — the confirmed crux for best-of-N; a preference reward-model on the
  user's good/bad labels. Not started. The 3 defects are its negative targets. E4 (critic-as-OPTIMIZER: richer
  typed+local input + preference objective → best-of-N/rejection-loop/guided-decode/RL ladder) RECORDED, gated on E0.1
  (lineage Decisions 2026-07-12b).

## ★ NEXT ACTION (locked 2026-07-12) — the exact sequence
1. **BUILD the #3 free-foot-under-hold fix** (`footspeed_floor_findings.md §5b`), STRUCTURAL-primary per
   [[structural-over-salience]]: automaton hold-release / duration-cap (or pattern-head logit shaping) as the primary,
   `stamina_hold_bump` thinning as the ordered residual (⚠️ pipeline-ordering: compute release on PRE-thinning demand;
   gate the bump off on holds about to close). Metric `scratchpad/measure_defect.py`; probe on fast + `freeze=high`
   (Watch Out Pt.2). WHY first: #3 is the one open defect that would POLLUTE taste labels (candidate-VARYING under
   freeze=high + presence-INVISIBLE to the critic).
2. **THEN the critic track** — E0.1 best-of-N spread set (kill-switch + seeds labels). **harm_calib EXCLUDED as a
   conditioning/ranking axis for this label round** (user decision). Comparative-at-fixed-song labeling makes SHARED
   flaws (wind-down, empty-middles) cancel, so they need not be fixed first.

## ⏳ AWAITING USER — binding questions
1. **Silence-pad by-ear re-confirm.** The hangover was ear-validated with REFLECTION; the pad default is now SILENCE
   (`hangover_reflect=False`, correctness call). Offline near-identical, but re-confirm on the next long-song play.
   Installed A/B: `~/sm-generated/stamina_probe/Lick {GLOBAL (base), HANGOVER (fix)}` (distinct #TITLEs; clear
   `~/.stepmania-5.1/Cache` if StepMania shows stale titles).
2. **UNIVERSAL sub-train-length window — ✅ SHIPPED AS DEFAULT (2026-07-12; by-ear PASSED)** (`notes/universal_window_
   findings.md`, `playtest_log.md`). Premise measured: v2 train-len median 3120 / MAX 5128, abs-PE exposure 31%/13%/6%
   by pos 3500/4000/4320, yet `onset_window` was pinned at V2_MSL(5400) so short songs' ends collapsed like long ones.
   `probe_universal_window.py` (RIGHT population, human ground truth, n=60): single-pass fires **30% of real TAIL
   notes** on the 3800–5128 band; W3600 restores recall + backbone to the HUMAN value; control byte-identical no-op;
   W4320 ~no-op (proves abs-PE effect). Decoded + BY-EAR A/B: **windowed won on all 3 songs** ("great"/"better"/
   "fine"). **`decode_defaults.UNIVERSAL_ONSET_WINDOW=3600`; both CLIs default v2 → 3600; check_export_defaults 27 ✓.**
   NEW residual **[H-winddown]** (SEPARATE, pre-existing): neither arm winds down into a silence/outro — candidate =
   window restores tail p_onset peaks → stamina breathe thins the outro less (queued probe, don't build blind). Local
   tau stays shelved.

## THE v1.0.0 SHIP CHECKLIST (DEFERRED — resume when the arc lands or is set down)
Regen the personal deliverable, doc the hold-stream known-limitation, host, announce ([[marketing-track]]). The
hold-stream free-foot edge = defect #3.

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py`)
The bare `export_typed_samples.py` run reproduces what the user plays; these MUST equal its argparse defaults.
UNCHANGED this session (the hangover + window knobs are generate.py-only / probe-only, off by default, and do NOT
touch the exporter). **Permanent section — keep in every rewrite.**
<!-- CANONICAL-EXPORT-DEFAULTS:START (do NOT hand-edit values; re-run tools/check_export_defaults.py after a change) -->
```
checkpoint = checkpoints/gen_motif_v2_48th_cont/best_val.pt
features = highres_v2
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
auto_b_trip = True
triple_pref_thresh = 0.0
grid_snap = auto
grid_snap_keep_triplets = True
onset_window = 3600
onset_tail_hangover = auto
hold_stream_penalty = 8.0
hold_stream_floor = 0.45
hold_stream_win = 16
footswitch = False
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- On branch **`explore/taste-critic-quality-resolution`** (off `feat/youtube-audio-pull-trim-append`). Holds the
  session-1 work (graded critic v2, stamina probe, harm_calib), the session-2 work (hangover pad + onset-window
  internals), AND the 2026-07-12 UNIVERSAL WINDOW ship (`decode_defaults.py`/`generate.py`/exporter, the
  `probe_universal_window{,_decoded}.py`, `notes/universal_window_findings.md`) + this refresh. The 2026-07-12 commit
  ALSO includes the pre-existing `notes/grid_snap_findings.md` duple-fidelity addition (user-approved) and gitignores
  `teaching/`.
- Verify the current commit/PR via `git log` / `gh pr list` (don't trust this doc's state).
- Still untracked, NOT mine (leave alone): `.claude/commands/begin.md`.

## READ-FIRST (in order)
Memory [[ship-mode-park-research]] (un-park block) → this HANDOFF → lineage `experiment_lineage/taste-critic-arc.md`
→ `notes/universal_window_findings.md` (the shipped window) + `notes/playtest_log.md` (2026-07-12 entry = the by-ear
verdict + the RNG explanation + [H-winddown]; 2026-07-11 entries = the harm gate / hangover / empty-middles) →
memories [[taste-critic-transfer]], [[meter-4-4-grid]], [[personal-reference-charts]]. Load-bearing skills:
**conditioning-mechanics** (§6 now covers the universal window + global-tau empty-middles + hangover; §8 stamina),
**experiment-design** (Rule 5/11 wrong-population — the universal window fixed the predecessor's error; Rule 6
cheapest-first; Rule 9 don't-commit), **generation-defaults** (§1 the 3 v2 defaults incl. the window).

## DISCIPLINE
**The EAR is the deciding vote** — every offline metric is a proxy (this session: the hangover offline "confirmed the
mechanism" but the EAR both validated it AND surfaced 2 new defects the metrics missed). **Run the fair test / right
POPULATION before committing** (I tested smaller-W on the wrong population; the user caught it). **Retract cleanly** —
I committed "PERC feeds the tail" and "decoder PE" and had to retract both when the cheap probe/arithmetic overturned
them. **One change at a time. Match the verb to the evidence** ([[claim-precision]]). Ship mode is PAUSED, not off.
