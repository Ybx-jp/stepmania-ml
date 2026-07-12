# HANDOFF — ★ SHIP MODE PAUSED: taste-critic arc UN-PARKED. Active = decode-fixing the 3 long-song defects + a taste-aligned critic.

**Written 2026-07-11 (session 2) for the next Claude.** The "ship v1.0.0" directive is **PAUSED** (the taste-critic
thread was consciously un-parked 2026-07-11; see [[ship-mode-park-research]]). v1.0.0 is DEFERRED, not cancelled; the
OTHER parked paths (GDL, seq-onset retrain, good-settings formula) stay parked. **This session shipped ONE new
generate.py-only lever (the HANGOVER PAD, off by default) + flipped its pad default to silence; the canonical v2
export default is UNTOUCHED** (validator green).

## WHERE WE ARE
- **Deployed model UNCHANGED:** v2 48th-grid `gen_motif_v2_48th_cont` + `--features highres_v2`, both CLIs default to
  it. This session added `generate.py --onset_tail_hangover` (long-song tail fix, OFF by default) + tunable onset
  window internals (`hop_frac`/`hangover_reflect`, probe-only, default-inert). No export-path change.
- **Goal (user's words):** f48 raised quality VARIANCE — fix = SELECTION (a critic picks the best of N conditioning
  variants), plus DECODE-FIXING the concrete defects that make gens feel off. The 3 defects ARE the critic's negative
  target list.
- **This session (2026-07-11 s2) advanced the DECODE-FIX track on long songs** (`notes/playtest_log.md`, newest
  entries; probes `experiments/generation_typed/probe_{subtail_position,lick_vs_byebye,onset_window_sweep,harm_fills_middle}.py`):
  - **Defect #2 (quiet under-charge) — harm_calib gate PASSED by ear** ("did its job"). Mechanism nailed: harm_calib
    is DENSITY-PRESERVING → it TRADES (fills melodic by STEALING from percussive: HARM-TOTAL +40% lull / −13% out-of-
    gate), does not add. TOTAL+PERC compete for one budget.
  - **Defect #1 (sub-16th tail) → a LENGTH-GATED long-song defect** (both long songs, neither short; harm_calib
    EXONERATED). = the onset-head sliding-window TRAILING EDGE (song-end at the final window's under-trained high-
    local-PE, no later window to heal → quarter-backbone COLLAPSE; user's hypothesis, confirmed by the window
    arithmetic). **FIX SHIPPED: the HANGOVER PAD** (`--onset_tail_hangover auto`); **BY-EAR "definitely better."**
  - **Empty MIDDLES ("long empty sections + scattered notes", both long songs) = a SEPARATE bug: global-tau
    ALLOCATION starves onset-poor regions** (NOT windows, NOT harm_calib — both refuted offline). local/windowed tau
    fixes it offline (maxgap 371→188) but is the Rule-13 quota anti-pattern → **user SHELVED it.**

## ★ ACTIVE THREAD — taste-critic-quality arc (lineage `experiment_lineage/taste-critic-arc.md`)
Two complementary tracks; the defects feed the critic (they ARE its negative targets):
- **(A) Decode-fix the 3 defects.** #2 harm_calib = PASSED (density-preserving trade, documented — don't stack gains
  blindly). #1 tail COLLAPSE = FIXED (hangover, ear-confirmed); the empty-MIDDLES half is OPEN (density allocation,
  local-tau shelved). #3 (free-foot-overload during a hold) still PARKED.
- **(B) Phase 2: taste-align the critic (R3)** — the confirmed crux for best-of-N; a preference reward-model on the
  user's good/bad labels. Not started. The 3 defects are its negative targets.

## ⏳ AWAITING USER — binding questions
1. **Silence-pad by-ear re-confirm.** The hangover was ear-validated with REFLECTION; the pad default is now SILENCE
   (`hangover_reflect=False`, correctness call). Offline near-identical, but re-confirm on the next long-song play.
   Installed A/B: `~/sm-generated/stamina_probe/Lick {GLOBAL (base), HANGOVER (fix)}` (distinct #TITLEs; clear
   `~/.stepmania-5.1/Cache` if StepMania shows stale titles).
2. **The user's OPEN priority = a UNIVERSAL sub-train-length window.** Short songs (T≤5400) get NO windowing today, so
   a ~4800-frame song's end sits at the under-trained ABS-PE tail → the user believes a ~80%-train-length window
   applied to ALL songs fixes broad short-song END-degeneration (seen even pre-windowing). **UNTESTED** — my
   `onset_window_sweep` tested smaller-W on Lick's MIDDLES (WRONG population, exp-design Rule 5/11). NEXT window work =
   test smaller-W applied universally on the SHORT val-set songs that show end-degeneration. NOT local tau (shelved).

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
  session-1 work (graded critic v2, stamina probe, harm_calib) AND this session-2 work (the hangover pad +
  tunable onset-window internals in `typed_model.py`/`decode_harness.py`/`generate.py`, the 4 new probes, these docs).
- A `docs/...` refresh branch carries THIS refresh commit (open a PR to `main`; verify number via `gh pr list`).
- Pre-existing untracked/unstaged NOT mine (leave alone): `.claude/commands/`, `teaching/`, and the pre-session
  modification to `notes/grid_snap_findings.md`.

## READ-FIRST (in order)
Memory [[ship-mode-park-research]] (un-park block) → this HANDOFF → lineage `experiment_lineage/taste-critic-arc.md`
→ `notes/playtest_log.md` (2026-07-11 entries — the harm gate, the length-defect chain, the hangover fix, the
empty-middles diagnosis, the user decisions) → memories [[taste-critic-transfer]], [[meter-4-4-grid]],
[[personal-reference-charts]]. Load-bearing skills: **conditioning-mechanics** (§6 now covers the global-tau empty-
middles + the onset sliding-window/hangover; §6 harm gates; §8 stamina), **experiment-design** (Rule 5/11 wrong-
population, Rule 13 global-quota, Rule 9 don't-commit — all exemplified this session), **generation-defaults**.

## DISCIPLINE
**The EAR is the deciding vote** — every offline metric is a proxy (this session: the hangover offline "confirmed the
mechanism" but the EAR both validated it AND surfaced 2 new defects the metrics missed). **Run the fair test / right
POPULATION before committing** (I tested smaller-W on the wrong population; the user caught it). **Retract cleanly** —
I committed "PERC feeds the tail" and "decoder PE" and had to retract both when the cheap probe/arithmetic overturned
them. **One change at a time. Match the verb to the evidence** ([[claim-precision]]). Ship mode is PAUSED, not off.
