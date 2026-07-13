# HANDOFF — ★ Active = taste-critic-quality arc. Decode-fix track: DEFECT #3 free-foot-under-hold FIXED + CANONICALIZED. Next = the critic best-of-N (E0.1).

**Written 2026-07-12 for the next Claude.** Ship v1.0.0 is PAUSED (taste-critic thread un-parked 2026-07-11; the OTHER
parked paths — GDL, seq-onset retrain, good-settings formula — stay parked). **This session BUILT + SHIPPED the DEFECT #3
free-foot-under-hold fix as a CANONICAL DEFAULT (by-ear "much better, maybe totally fixed").**

## WHERE WE ARE
- **Deployed model UNCHANGED:** v2 48th-grid `gen_motif_v2_48th_cont` + `--features highres_v2`, both CLIs default to it.
- **★ CANONICAL DECODE CHANGED THIS SESSION (2026-07-12):** the free-foot-under-hold force-close is now ON by default.
  `decode_defaults.CANONICAL_DECODE` gained `hold_release_run=4`, `hold_release_gap=None` (→ an 8th = `subdiv//2`),
  `hold_max_beats=6.0`. Both CLIs (`export_typed_samples.py`, `scripts/generate.py`) source them. **Fires ONLY on a real
  defect → byte-identical on clean charts.** `check_export_defaults` updated (see the canonical block below).
- **The fix (rule = the user's exact spec, `notes/footspeed_floor_findings.md §5c`):** while a hold pins one foot, an 8th
  is the fastest allowable free-foot note under it. A note FASTER than an 8th → the hold CONCLUDES ON THE CURRENT note (via
  the **precomputed non-causal onset LOOKAHEAD** `onset[:,t+1:t+gap].any()` → release on the FIRST note of the fast run so
  the freed foot travels into it). A run of `hold_release_run` 8ths → release (3-note flourish free). `hold_max_beats`
  duration cap for quiet monster holds. Code in `typed_model.generate()` `hold_aware` block (~:929). **`stamina_hold_bump`
  salience residual was NOT built — unneeded (automaton alone drives the defect to 0), validating [[structural-over-salience]].**
- **Metric (rebuilt this session):** `scratchpad/measure_defect.py` (`scratchpad/` is NOT gitignored → now committed, unlike
  the prior lost copy). Validated vs the documented anchor EXACTLY (holdfix 2 / holdbug 4). ★ **A/B installs for reference:**
  `~/sm-generated/holdrelease_{byear,v2,v3}` (v3 = the shipped lookahead fix; Challenge=fix, Edit=baseline-off).
- **Goal (user's words):** f48 raised quality VARIANCE → fix = SELECTION (a critic picks the best of N conditioning
  variants), plus DECODE-FIXING the concrete defects. #3 was the last defect that would POLLUTE taste labels
  (candidate-varying under `freeze=high` + presence-invisible to the critic) → it was built BEFORE the label matrix.

## ★ ACTIVE THREAD — taste-critic-quality arc (lineage `experiment_lineage/taste-critic-arc.md`)
Two complementary tracks; the defects feed the critic (they ARE its negative targets):
- **(A) Decode-fix the 3 defects.** #1 tail-collapse = FIXED (hangover + universal window, both shipped). #2 harm_calib =
  PARTIAL (density-preserving trade, lever OFF by default). **#3 free-foot-under-hold = ✅ FIXED + SHIPPED + CANONICAL this
  session.** empty-MIDDLES = SHELVED (global-tau allocation, Rule-13 quota). [H-winddown] outro-taper = queued probe (don't
  build blind).
- **(B) Phase 2: taste-align the critic (R3)** — the confirmed crux for best-of-N; a preference reward-model on the user's
  good/bad labels. Not started. E4 (critic-as-OPTIMIZER: richer typed+local input + preference objective → best-of-N /
  rejection-loop / guided-decode / RL ladder) RECORDED, gated on E0.1 (lineage Decisions 2026-07-12b).

## ★ NEXT ACTION (the #3 gate is cleared)
1. **E0.1 — the best-of-N SPREAD set.** Build a kill-switch/oracle check that best-of-N over conditioning variants at a
   FIXED song actually helps (if an oracle can't pick a better candidate, the whole selection vision is moot). **harm_calib
   EXCLUDED as a conditioning/ranking axis for this label round** (user decision). Comparative-at-fixed-song labeling makes
   SHARED flaws (wind-down, empty-middles) cancel, so they need not be fixed first.
2. **THEN the critic label round** (R3 preference model) → the ladder (selection → rejection-loop → guided-decode → RL),
   each rung gated (E4).

## ⏳ AWAITING USER — binding questions
1. **#3 by-ear = PASSED** ("much better, maybe totally fixed! let's ride with it for now") → canonicalized. TWO watch-items
   the user was told about, no objection raised, revisit if they surface in future play: (a) the fix SHORTENS holds (a fast
   approach releases early; Watch Out 19→25 holds, shortest ~0.5 beat) — flag if holds feel choppy; (b) a RESIDUAL pattern
   remains — a hold that OPENS on a fast note (a two-foot hold-ENTRY: hold-head then a 16th on another arrow). If (b) ever
   bugs the user, the fix is to REFUSE to open a hold when the next onset is faster than an 8th (place a tap instead).
2. **Silence-pad by-ear re-confirm (still open from a prior session):** the hangover pad default is SILENCE
   (`hangover_reflect=False`). Re-confirm on the next long-song play. Installed A/B: `~/sm-generated/stamina_probe/Lick`.

## THE v1.0.0 SHIP CHECKLIST (DEFERRED — resume when the arc lands or is set down)
Regen the personal deliverable, doc the (now-fixed) hold-stream limitation, host, announce ([[marketing-track]]).

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py`)
The bare `export_typed_samples.py` run reproduces what the user plays; these MUST equal its argparse defaults.
**Permanent section — keep in every rewrite.**
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
hold_release_run = 4
hold_release_gap = None
hold_max_beats = 6.0
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- On branch **`explore/taste-critic-quality-resolution`** (off `feat/youtube-audio-pull-trim-append`). This session's #3
  work: `src/generation/{typed_model,decode_defaults}.py`, `experiments/generation_typed/export_typed_samples.py`,
  `scripts/generate.py`, `notes/footspeed_floor_findings.md §5c`, `scratchpad/{measure_defect,sweep_measure}.py`, this
  refresh. The docs refresh lands on a `docs/...` branch → PR to `main` (protected).
- Verify the current commit/PR via `git log` / `gh pr list` (don't trust this doc's state).
- Still untracked, NOT mine (leave alone): `.claude/commands/`, `.claude/worktrees/`.

## READ-FIRST (in order)
Memory [[taste-critic-transfer]] (active thread) + [[structural-over-salience]] (the #3 principle, now VALIDATED) → this
HANDOFF → lineage `experiment_lineage/taste-critic-arc.md` (Results 2026-07-12 session 2 = the #3 fix) →
`notes/footspeed_floor_findings.md §5c` (the fix + the 4-round metric scar) → `notes/playtest_log.md`. Load-bearing skills:
**conditioning-mechanics** (§7 now covers the hold force-close lever + the metric trap), **generation-defaults** (§1 palette
incl. the 3 new hold-release knobs), **experiment-design** (Rule 8 the-ear-is-ground-truth; Rule 1 match-metric-to-property —
the persist-exclusion hid the defect 4×).

## DISCIPLINE
**The EAR is the deciding vote** — the persist-based metric reported "0 defect" FOUR TIMES while the user heard a real one;
each round the ear was right (match the metric to the FELT property, not a convenient aggregate). **The onset schedule is
PRECOMPUTED/non-causal** — a decode rule CAN look ahead at upcoming onsets (the lookahead trick that fixed #3). **One change
at a time. Match the verb to the evidence** ([[claim-precision]]). Ship mode is PAUSED, not off.
