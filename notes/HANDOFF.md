# HANDOFF — ★ SHIP MODE: cut v1.0.0 (deploy-swap v2 + guide). Research PARKED. Don't wander.

**Written 2026-07-06 for the next Claude. THE PROJECT IS IN SHIP MODE** (user decision — memory
[[ship-mode-park-research]]): clean up, cut **v1.0.0** (tags `0.1.0`/`0.2.0` already exist), host, announce, move on.
The v2 48th-grid model is "a full step better in expressiveness… already pretty incredible." The remaining gap ("onset
allocation undertuned") is the **note-context PLACEMENT CEILING** — a RETRAIN problem, not a decode-tune — so more
knob-tuning is diminishing returns. **Do NOT initiate the parked research** (good-settings tolerance formula,
GDL/meter-equivariance retrain, seq-onset retrain); redirect tangents to the release checklist. Un-park ONLY on the
user's literal phrase **"the times have changed."**

**LATEST session (2026-07-07) — PUBLIC CLI now speaks v2 + `.sm` header/BGCHANGES passthrough** (commit `cf0b820`;
[[ship-mode-park-research]]): `scripts/generate.py` (the bring-your-own-SONG single-file CLI) gained `--features
highres_v2` — auto-selects the v2 checkpoint (`gen_motif_v2_48th_cont`), the 48th feature extractor, the 5504-frame
context, `V2_MSL`, the 48-row `sm_writer`, and the v2 decode flags (`--no_fast_jump` default-on, `--min_onset_gap`,
`--grid_snap`, `--auto_b_trip`), all derived from one `subdiv` (`fspec.extractor.config.timesteps_per_beat`) so
audio/notes can't drift. **Its DEFAULT stays highres/v1 (v1 byte-identical) — this is v2-REACHABLE-via-CLI, NOT the
deploy-swap (ship checklist #1 still open).** Also added **`--title` + presentation flags + `--inherit_from SM|auto`**:
inherit a source chart's banner/background/**#BGCHANGES music-video** tags and COPY the media into the new folder so
they resolve (new module `src/generation/sm_headers.py`; `sm_writer` gained a `header=` dict in StepMania tag order,
default output unchanged; `meter_detect.detect_triple_pref_audio` = the b_trip switch from audio+BPM, no `.sm` needed).
Timing tags (`#BPMS`/`#STOPS`) deliberately NOT inherited (the generator owns its grid). **Deliverable:** 34/35 v2 Hard
charts for `~/sm-personal` → `~/sm-generated/v2_personal_hard/` (1 fail = corrupt `.ogg` "Raining Down"; 26 inherited a
source header, 9 with playing videos). No new experiment/finding — engineering deliverable, no lineage arc. Tag order
read from the local SM source `~/stepmania-5.1.0-b2/src/NotesWriterSM.cpp`.

**PRIOR session (2026-07-06c) — LOW-DIFFICULTY verification + the 16th-grid SNAP** (`notes/grid_snap_findings.md`,
[[low-diff-gridsnap]]): verified the v2 deploy candidate at Beginner/Easy/Medium (it had ONLY been by-ear'd on Hard).
**v2 generates them coherently** (no degeneration, density tracks, critic reads low, sparse songs ~100% on-grid ==
originals), EXCEPT busy low-diff songs place **8–23% of notes on pure-48th cells `{1,5,7,11}`** human originals never
use (real 48th-usage ~0% at all low/mid diffs). Hypothesised the 16th-unlock; **A/B REFUTED it** (unlock OFF didn't
drop off-grid; note-count decomp → the unlock moves ON-grid density, the 48th count is independent). **True cause =
the 48th grid's double edge** — the same beat-synced sub-16th capability that gives v2 its triplet win also admits
48th jitter on busy DUPLE songs. **FIX = `grid_snap`** (`decode_defaults.grid_snap_offset`, −30 logit veto ridden
through the exporter `harm_off_t` slot → single-sourced into tau+decode; v1 no-op by construction): off-grid
6.6%→0%, density preserved, INERT on already-clean songs. **WIRED TO THE CANONICAL DEFAULT this session (per user
directive; BY-EAR PENDING):** `--grid_snap auto` (keep-triplets 48th-veto for difficulty ≤ Medium, OFF at Hard) +
flipped **`--auto_b_trip` default False→True**. The prior 06-06b uncommitted code (`--auto_b_trip`, `INFERENCE_GATES`)
is now committed alongside grid-snap. Guard 21→**25 ✓**; validated v1 byte-identical + the v2 auto-gate (snaps @Easy,
not @Hard). **PRIOR session (06-06b)** built the auto-switch + safe-settings sweep + the gate fix — folded in.

## WHERE WE ARE
- **Deployed model STILL v1** = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim, 16th grid). Nothing v2 is
  deployed yet. ⚠️ The canonical defaults NOW include the two v2-only auto knobs (`grid_snap=auto`, `auto_b_trip=True`)
  — but both are **v1 no-ops** (verified byte-identical), so what the user PLAYS on v1 is unchanged; `check_export_
  defaults.py` = **25 ✓**.
- **v2 deploy candidate** = `checkpoints/gen_motif_v2_48th_cont/best_val.pt` (val 0.7435), `--features highres_v2`.
- **Prior commits on `feat/governor-subdiv-recalib`** (verify via `git log`): `33de530` governor subdiv-recalibration
  → `63125eb` footspeed floor + `--style` density fix → `46a25b4` triplet phase band → `ed26aa6` groove-radar subdiv
  chaos (retrain-gated) → docs-refresh commits. All subdiv=4 BYTE-IDENTICAL (v1 untouched).
- **Latest commit `cf0b820`** (2026-07-07): `scripts/generate.py` v2 support + `.sm` header inheritance + new
  `src/generation/sm_headers.py` + `sm_writer` `header=` dict + `meter_detect.detect_triple_pref_audio` (all v1
  byte-identical; `export_typed_samples.py` argparse defaults untouched → `check_export_defaults.py` still 25 ✓).
- **06-06c commit** (16th-grid SNAP): `decode_defaults.grid_snap_offset` + exporter `--grid_snap`/
  `--grid_snap_keep_triplets`, the difficulty auto-gate, `--auto_b_trip` default→True, PLUS the 06-06b work
  (`src/data/meter_detect.py`, the `--auto_b_trip` switch, `INFERENCE_GATES` in `stepmania_parser.py`,
  `analyze_v2_envelope.py`, `notes/v2_safety_envelope_findings.md`).

## v2 DECODE-PLAYABILITY WORK ON THE BRANCH (prior sessions; all in `generate()` / the exporter / groove_radar; v1 byte-identical)
1. **Governor subdiv-recalibration** (`conditioning-mechanics §8`): the §8 governors reasoned in FRAMES assuming a
   frame=16th (`frame_hz=BPM·4/60`); on the 48th grid a frame=1/12 beat. Threaded `subdiv`: `frame_hz=BPM·subdiv/60`,
   `tau_frames=fatigue_tau·subdiv`, `stamina_decay` per-frame, and the integer gap/window thresholds via
   `f16=subdiv//4` (`since_onset≤f16`, the `free_gap` 16th/8th bands, `jack_max_gap·f16`, `hold_stream_win·f16`,
   `stamina_breathe_win·f16`). Exertion accumulators/caps need NO rescale (press-rate = frame_hz/gap is
   grid-invariant). **BY-EAR PASSED** (governor recalib playtest: Equinox "much better," maxJackRun 3→2).
2. **Footspeed floor** (`min_onset_gap`, generate()): a decode-time onset REFRACTORY (NMS on the precomputed onset
   tensor: enforce min pairwise gap = `min_onset_gap` frames, keep the higher-`p_onset` note in each too-close pair).
   Default auto: **2 on the 48th grid** (forbids 1-frame 48th flams, PRESERVES 2-frame triplet-16ths), **1 on the
   16th grid = no-op**. Killed all 31 of Equinox's 34ms flams. Exporter `--min_onset_gap`.
   ⚠️ BY-EAR: floor-ALONE reads "bland, messy between yellows/greens" — it's a blunt safety net, SUPERSEDED by #3.
3. **Triplet phase band** (`onset_phase_calib` optional 3rd element `b_trip`): applied to the triplet-only frames
   (`decode_defaults.triplet_band_positions` = {2,4,8,10}@subdiv=12, empty on the 16th grid). Single-sourced with the
   tau path via `decode_defaults.phase_calib_offset`. Default `b_trip=0` (off; canonical stays `(0.0,1.0)`).
   **BY-EAR WON:** at `b_trip=0.7` triplet-occupancy 0.107→0.390 (human 0.40–0.57), "more committal to greens, very
   even rhythm." Exporter: `--onset_phase_calib "0,1.0,0.7"`.
4. **`--style` manifold density subdiv-fix** (exporter, `footspeed_floor_findings.md §1`): the manifold's
   `E[density|·]` is a 16th-grid frac; on the 48th grid it placed ~3× too many notes. `style_density *= 4/subdiv`.
   **BY-EAR confirmed** (pt_chaos_v2 "conditioning effective," not a wall). Bare export unaffected (source-chart density).
6. **No-fast-jump cap** (`no_fast_jump`, default ON; `conditioning-mechanics §8d`, `footspeed_floor_findings.md §4`):
   in `generate()` after `max_jack_run` — when `since_onset < f16` (strictly sub-16th) forbid `fresh_cnt ≥ 2`
   patterns → the two-foot-jump sibling of `max_jack_run`. Forces a playable single, KEEPS the onset; causal
   (trailing-note-only, leader jump survives). v1 byte-identical (`f16=1`). **BY-EAR PASSED.**
5. **Groove-radar subdiv chaos** (retrain-gated, `manifold_radar_subdiv_findings.md`): `groove_radar._build_color_values`
   is now subdiv-aware (color by quantization denominator → triplets get DDR-green 1.25 not 1.0) + `dataset.py:104`
   threads `parser.timesteps_per_beat` (was hard-coded 4). ⚠️ **RETRAIN-GATED:** the v2 model + v1 manifold BOTH
   trained on tpb=4 radar (verified from the cache) — do NOT rebuild the v2 cache + reuse the current checkpoint, and
   do NOT refit the manifold (it would DE-SYNC from the model). Refit + subdiv-tagging is bundled with the next retrain.

## THE v1.0.0 SHIP CHECKLIST (this is the active work — see [[ship-mode-park-research]])
1. **★ DEPLOY SWAP — make v2 the default.** Coordinated bump in `generation-defaults §0` + `conditioning-mechanics §6`
   + `export_typed_samples.py` argparse: default checkpoint → `gen_motif_v2_48th_cont`, default `--features highres_v2`,
   `--min_onset_gap` auto (already), `no_fast_jump` on (already), and set **`b_trip=0.7` as the v2 default**
   (`--onset_phase_calib "0,1.0,0.7"`; song-dependent, 0.7 is the gentle default — 1.0 adds triplet busyness on duple
   songs). Re-point the CANONICAL EXPORT DEFAULTS block below + re-run `tools/check_export_defaults.py`.
   ⚠️ The groove-radar chaos refit stays RETRAIN-GATED (do NOT refit the manifold; `manifold_radar_subdiv_findings.md`).
   ✅ Song-REACH is already handled: v2 export now DEFAULTS to the widened `INFERENCE_GATES` (bpm[40,320]/len[30,600]/
   simul4/gimmick), +55% val reach (532→822); `--strict_gates` reverts. So the deployed v2 can chart far more songs.
2. **Decide the hold-stream edge.** The `freeze=high` v2 free-foot-stream-under-hold defect is PARTIAL-fixed only
   (`footspeed_floor_findings.md §5/§5b`; real fix DESIGNED + PARKED = position-based `stamina_hold_bump`, ~6 lines).
   SHIP decision: does this freeze=high-only edge block v1.0.0, or ship as a documented known-limitation? (Lean: ship
   it — it only bites the extreme `--style freeze=high` combo; the parked fix is ready if a user hits it.)
3. **Safe-settings envelope — ✅ CHARACTERIZED (2026-07-06b), guide still TO WRITE.** The 5-arm × 12-song v2 sweep
   (`analyze_v2_envelope.py`, `notes/v2_safety_envelope_findings.md`) established: **playability is rock-solid across
   the whole range** (0 fast-jumps/flams, jack ≤2, no smear) → the zone EXISTS. DEFAULT (bare) is clean; `--style
   chaos/freeze` is a use-with-care edge (24–28-beat dead gaps on long/sparse songs). The guide can be written on
   this. **The auto-`b_trip` nicety is BUILT** (`--auto_b_trip`, opt-in) BUT is SAFE-not-clean-win — the ρ+0.47
   detector fires on only 3/6 chart-triplet songs, so **auto-vs-global b_trip is an OPEN BY-EAR call** (pack
   `~/sm-generated/v2byear_*`, the Sway/Parousia A/B). It never harms duple songs, so it's shippable either way.
   NOTE: the deploy-swap (#1) can set `--auto_b_trip` OR a fixed `b_trip=0.7` as the v2 default — the by-ear verdict
   decides which.
4. **Release:** clean up, cut the `v1.0.0` tag, host, announce ([[marketing-track]]; adapt `RELEASE_CRITERIA.md`).
5. **PARKED — do NOT start without "the times have changed":** the position-based hold-stream fix (§5b); the manifold
   refit + groove-radar retrain (bundled); the seq-onset retrain (the note-context placement ceiling); the
   good-settings tolerance formula ([[good-settings-region]]); GDL/meter-equivariance ([[meter-4-4-grid]]).

## AWAITING USER
- **★ BY-EAR: the 16th-grid SNAP (low/mid difficulty)** — installed groups `~/sm-generated/v2_low_beginner`,
  `v2_low_easy`, `v2_low_medium` (canonical) vs `v2_low_easy_snap`, `v2_low_medium_snap` (fix). Play the BUSY songs
  (**See Me Now, SUPER SUMMER DIVE, Gengaozo, Deja loin**) canonical-vs-snap; the sparse songs are identical between
  groups. Question: does snapping the 48th jitter to the 16th grid read MORE coherent/human, or do you miss the
  sub-16th detail? Offline says off-grid 6.6%→0% with density preserved. If it wins → the `--grid_snap auto` default
  is validated; if not → set `--grid_snap off`. The **Hard boundary is UNTESTED** (auto leaves Hard on canonical).
  Log to `notes/playtest_log.md`.
- **BY-EAR: auto-vs-global b_trip** — pack `~/sm-generated/v2byear_01..09` (tagged titles). The key A/B is
  **03 vs 04 (Sway)** and **05 vs 06 (Parousia)**: band-OFF (`auto`, detector said duple) vs band-FORCED-ON
  (`global`). If forcing triplets sounds BETTER → the detector missed, favor `global b_trip=0.7`; if band-off
  sounds right → the audio really is duple, favor `--auto_b_trip`. Also 09 = After The Rain under chaos (does the
  ~28-beat dead gap sound broken?). Log to `notes/playtest_log.md`. This decides the v2 default b_trip in the swap.
- **DEPLOY-SWAP (ship checklist #1) still pending** — grid-snap + auto_b_trip are now defaults, but v2 is not yet
  THE default checkpoint/features. That coordinated bump (+ `b_trip=0.7` or `--auto_b_trip` as the v2 default) is the
  next ship step, gated on the two by-ear verdicts above.

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py` = 25 ✓)
The bare `export_typed_samples.py` run reproduces what the user plays. These MUST equal the script's argparse
defaults. `grid_snap`/`auto_b_trip`/`grid_snap_keep_triplets`/`triple_pref_thresh` are v2-only behaviors that are
**no-ops on the deployed v1 regime** (byte-identical verified). **Permanent section — keep in every rewrite.**
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
NOTE: v2 is a SEPARATE regime — `--features highres_v2` + `gen_motif_v2_48th_cont` + `for_v2()` + the 48th `sm_writer`
+ `V2_MSL=5400`. v2 decode knobs (all v1 no-ops): `--min_onset_gap` (None→auto 2 on v2); `--grid_snap auto`
(keep-triplets 48th-veto for difficulty ≤ Medium, OFF at Hard — the DEFAULT now); `--auto_b_trip` (DEFAULT ON;
per-song triplet band keyed on the audio meter detector, `b_trip=0.7`); the triplet band via `--onset_phase_calib
"0,1.0,b_trip"`. v2 export uses the WIDENED `INFERENCE_GATES` by default (`--strict_gates` reverts). Do NOT mix
`highres_v2` with the v1 checkpoint. `grid_snap` + `auto_b_trip` are OFFLINE-VALIDATED but BY-EAR PENDING as defaults.

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/governor-subdiv-recalib`** (off `feat/data-layer-v2`, off `main`). **Verify PR/branch state via
  `gh pr view` / `git log`.** Latest code commit **`cf0b820`** (generate.py v2 + `.sm` header inheritance) + this
  refresh's docs are on this branch; the grid-snap/auto_b_trip and 06-06b code are folded in. The DEPLOY SWAP (ship
  checklist #1 — make v2 THE default) is a NEW code change not yet made. Whether to open a PR to `main` for the whole
  long-lived v2 feature branch is a user call.
- Gitignored / not committed: `outputs/` (incl. `outputs/v2_sweep/*` the safety sweep), `transcripts/`, scratchpad
  probes. Installed by-ear pack: `~/sm-generated/v2byear_01..09`.

## READ-FIRST (in order)
Memory **[[ship-mode-park-research]]** (the operative directive) → **this checklist above** → `generation-defaults §0`
(the deploy-swap targets: v2 checkpoint/features/knobs) → `conditioning-mechanics §6/§7/§8` (the v2 decode levers +
the hold-stream partial fix + stamina-is-ON correction) → `notes/footspeed_floor_findings.md §4/§5/§5b` (no-fast-jump
= shipped; hold-stream = partial + the PARKED fix design) → **`notes/v2_safety_envelope_findings.md`** (the LATEST
session: the safety sweep, the switch, the gate fix, the auto-vs-global open question) → **`notes/grid_snap_findings.md`**
(the low-diff verification + the 16th-grid SNAP, now a canonical default) → `notes/playtest_log.md`
(the by-ear record) → lineage `experiment_lineage/meter-grid-arc.md` (§Session 5). Load-bearing skills:
**generation-defaults, conditioning-mechanics §6/§8, experiment-design** (the ship-vs-tune call = a Rule-"is this the
right investment" judgment).

## DISCIPLINE
**SHIP MODE: don't wander into parked research** ([[ship-mode-park-research]]; un-park only on "the times have
changed"). **Match the metric to the FELT property** (the hold-stream defect was mis-analyzed TWICE by trusting an
aggregate over the raw grid — dump the grid). **Stamina is ON canonically (ceiling 50), NOT off** (the skill text
that said "off" burned a session — corrected `conditioning-mechanics §8c`). **subdiv=4 must stay byte-identical** on
any v2 decode change. **Don't rebuild the v2 cache + reuse the current checkpoint** (radar de-sync). One change at a
time; match the verb to the evidence ([[claim-precision]]); HARNESS/COUPLING-first when a result looks wrong.
