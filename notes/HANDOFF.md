# HANDOFF — ★ SHIP MODE PAUSED: the TASTE-CRITIC arc is UN-PARKED. Active work = a taste-aligned critic for best-of-N.

**Written 2026-07-11 for the next Claude.** The standing "ship v1.0.0" directive is **PAUSED** — on 2026-07-11 the
user CONSCIOUSLY UN-PARKED the taste-critic thread ("un-park now" = the tripwire override; see
[[ship-mode-park-research]], now marked partially-superseded). v1.0.0 is DEFERRED, not cancelled; the OTHER parked
paths (GDL, seq-onset retrain, good-settings formula) stay parked. **No deployed-model change this session** — the
canonical v2 default is untouched; all new code is experimental + off-by-default.

## WHERE WE ARE
- **Goal (user's words):** f48 raised quality VARIANCE — "banger when the conditioning is right, but I can't predict
  the right knobs per song." Fix = **SELECTION** (generate N conditioning variants, a critic picks the best), NOT
  PREDICTION (audio→knobs is the [[good-settings-region]] clean-negative). The critic is the linchpin.
- **The diagnostic that framed it** (`notes/taste_critic_v2_findings.md`): the taste critic (a) rates the user's OWN
  hand-made charts mostly "fake" (P(real) 0.32 vs train-real 0.82); (b) was 16th-grid so it CAN'T see f48 output
  (grid wall, user-predicted); (c) graded objective helps but isn't enough. → a 3-part arc: R1 see-f48, R2 grade,
  R3 taste-align.
- **✅ E1.1 DONE — R1 cleared.** `experiments/realism_critic/train_graded_critic_v2.py` →
  `checkpoints/realism_critic_graded_v2/best_val.pt` (48th grid, warm-started from the binary critic, + a sub-16th
  JITTER corruption axis). Jitter ladder monotone **0.98** → it grades the sub-16th placement the 16th critic
  couldn't see.
- **✅ E1.2 DONE — mixed; R3 (taste) is the confirmed crux.** `scratchpad/critic_catches_defects.py`: the critic
  CATCHES the tail/placement defect (tail scored −1.59 below body, within-song), but is presence-blind to subtle
  arm quality and **rates generations above the user's human charts on 2/3 songs**. E1.1 fixed the grid, NOT taste.
- **STAMINA long-song detour (user-prioritized) CLOSED — hypothesis REFUTED.** The breathing arc's whole-song
  z-normalization IS length-mis-scoped (corr(len,ceiling-divergence)+0.83) but does NOT bite the chart (fair
  density test + by-ear: OFF was WORST, GLOBAL best-or-tied). `notes/stamina_longsong_findings.md`. New non-breaking
  `stamina_breathe_local_win` lever kept as a MILD partial fix for defect #2 (NOT default).
- **PLAYTEST (2026-07-11, `notes/playtest_log.md`) enumerated 3 real defects = the critic's target list:**
  **#1** spurious sub-16ths, worst near the END, consistent across songs (critic SEES it); **#2** quiet/harmonic
  sections under-choreographed (decode leads: harm_calib + local-z); **#3** very-high foot-speed during a hold,
  worst on fast songs (presence-BLIND → the parked free-foot-overload gate).

## ★ ACTIVE THREAD — taste-critic-quality arc (lineage `experiment_lineage/taste-critic-arc.md`)
Two complementary tracks; the defects feed the critic (they ARE its negative targets):
- **(A) Decode-fix the 3 defects** to raise base quality. STARTED with **defect #2** (user "leaning decode fixes",
  flagged harm_calib). Added `generate.py --harm_quiet_feat total|perc` (perc = cond-mech §6 dim-35 gate). Offline
  (`scratchpad/harm_offline.py`): both gates land + orthogonal (TOTAL +40% density in silent lulls; PERC +17% in
  busy-harmonic drum-sparse sections; the two target DIFFERENT sections, Jaccard 0.20).
- **(B) Phase 2: taste-align the critic (R3)** — the confirmed crux for best-of-N; a preference reward-model on the
  user's good/bad labels (mine `playtest_log.md` + structured pairs). Not started.

## ⏳ AWAITING USER — the binding gate
**By-ear verdict on the Bye Bye 4-arm harm_calib A/B**, installed at `~/sm-generated/stamina_probe/` (group has
Bye Bye {OFF, GLOBAL=base, LOCAL, HARM-TOTAL, HARM-PERC} + Switch/Calling stamina arms). **Question:** in the bare
spots, does HARM-TOTAL (silent-lull fill) or HARM-PERC (harmonic-passage fill) land the choreography — or does gain
10 over-boost junk? Log the verdict to `notes/playtest_log.md` (newest on top). Then: lock a #2 fix (gate+gain, fold
`local-z` in) and move to defect #1 (sub-16th tail probe) or #3 (free-foot gate) — OR, if neither gate lands, #2 is
an onset-head/retrain matter → it goes to the reward-model's target list, and we pivot to track (B).

## THE v1.0.0 SHIP CHECKLIST (DEFERRED — resume when the arc lands or is set down)
Was one mechanical step from a cut (regen the personal deliverable, doc the hold-stream known-limitation, host,
announce; [[marketing-track]]). The hold-stream free-foot edge is the same as defect #3 above — the arc may fix it.

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py` = 25 ✓ this session)
The bare `export_typed_samples.py` run reproduces what the user plays; these MUST equal its argparse defaults.
UNCHANGED this session (the new `generate.py` flags `--stamina_breathe_local_win`/`--harm_quiet_feat` are
generate.py-only, off by default, and do NOT touch the exporter). **Permanent section — keep in every rewrite.**
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
- On branch **`explore/taste-critic-quality-resolution`** (off `feat/youtube-audio-pull-trim-append`). Holds this
  session's work: the v2 graded-critic trainer, the stamina probe, the non-breaking `generate.py`/`typed_model.py`
  levers, and these docs. NOT yet PR'd (mid-investigation, experimental code, awaiting the by-ear gate) — open a PR
  only when the arc reaches a shippable conclusion.
- Pre-existing untracked/unstaged NOT mine (leave alone): `.claude/commands/`, `teaching/`,
  `notes/grid_snap_findings.md` (modified before this session).

## READ-FIRST (in order)
Memory [[ship-mode-park-research]] (the un-park block) → this HANDOFF → lineage
`experiment_lineage/taste-critic-arc.md` → `notes/taste_critic_v2_findings.md` + `notes/stamina_longsong_findings.md`
→ `notes/playtest_log.md` (2026-07-11 entry) → memories [[taste-critic-transfer]], [[meter-4-4-grid]]. Load-bearing
skills: **conditioning-mechanics** (§6 harm gates, §7 hold-stream, §8 stamina), **experiment-design** (Rule 7/9 —
the stamina detour is the exemplar), **generation-defaults**.

## DISCIPLINE
**The EAR is the deciding vote** — every offline metric (ceiling divergence, ladder AUC, critic margin) is a proxy;
the stamina detour is the cautionary tale (a +0.83 cheap-probe correlation that the fair test + ears REFUTED). **Run
the fair test before committing a conclusion** (Rule 7/9, necessary≠sufficient). **Match the verb to the evidence**
([[claim-precision]] — the user corrected GLOBAL→LOCAL on the bridge). **One change at a time.** Ship mode is PAUSED,
not off — don't wander into the OTHER parked paths.
