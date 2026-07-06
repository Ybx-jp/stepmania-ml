# HANDOFF — data-layer-v2 DECODE-PLAYABILITY done: no-fast-jump cap + hold-stream subdiv fix SHIPPED. Next = deploy swap.

**Written 2026-07-05 (session 2) for the next Claude.** The v2 48th-grid model was a deploy candidate but its
decode governors were miscalibrated on the finer grid. This session **recalibrated the governors, added a footspeed
floor + a triplet phase band, fixed the `--style` manifold density, and fixed (retrain-gated) the groove-radar
triplet-chaos measurement** — then playtested. Result: **a big expressive WIN by ear** ("brand new note colors,"
"flowy streams," "conditioning effective"), with ONE diagnosed remaining playability hole (fast jumps) queued as the
next action, and a second (hold-stream gate on freeze=high) freshly reported.

## WHERE WE ARE
- **Deployed model STILL v1** = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim, 16th grid). Canonical v1
  decode defaults UNCHANGED (block below; `tools/check_export_defaults.py` passes). Nothing v2 is deployed yet.
- **v2 deploy candidate** = `checkpoints/gen_motif_v2_48th_cont/best_val.pt` (val 0.7435), `--features highres_v2`.
- **This session's 4 commits on branch `feat/governor-subdiv-recalib`** (verify via `git log`): `33de530` governor
  subdiv-recalibration → `63125eb` footspeed floor + `--style` density fix → `46a25b4` triplet phase band →
  `ed26aa6` groove-radar subdiv chaos (retrain-gated). All subdiv=4 BYTE-IDENTICAL (v1 untouched).

## WHAT SHIPPED THIS SESSION (all in `generate()` / the exporter / groove_radar; v1 byte-identical)
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

## THE OPEN FORK / NEXT ACTIONS (in order)
1. **✅ DONE — the no-fast-jump cap** (`footspeed_floor_findings.md §4`; `no_fast_jump`, default ON). BUILT in
   `generate()` (right after `max_jack_run`): when `since_onset < f16` forbid `fresh_cnt ≥ 2` → forces a playable
   single, KEEPS the onset. Causal/backward-looking (trailing note only; leader jump survives; rolling gap not
   f16-cell binning). v1 byte-identical (`f16=1`). Exporter `--no_fast_jump/--no-no_fast_jump` + `--ab_no_fast_jump`.
   **BY-EAR PASSED** (`nofastjump_ab`, Equinox: capped ≈ uncapped feel; uncapped had a "silly" 3-jump-jack in sub-16th
   space). Committed `df39c3c` (verify via `git log`).
2. **✅ FIXED (offline-confirmed, by-ear PENDING) — hold-stream gate DEAD on the 48th grid** (`footspeed_floor_findings.md
   §5`; `conditioning-mechanics §7`). Root: the gate compares a frame-FRACTION `dens` to `hold_stream_floor=0.45`, but a
   fraction shrinks ~`subdiv/4`× on the finer grid (a 16th stream = 1.0 on v1 but ~0.33 on v2) → the floor never fired →
   holds flooded streams (Watch Out `freeze=high` "gate broken"). The governor pass fixed `win` (a COUNT) but missed
   `dens` (a FRACTION). FIX (one line): `dens=(dens·subdiv/4).clamp(max=1.0)` before the floor — 16th-native, v1
   byte-identical. Confirmed: holds-in-dense-frames 6→3, total 28→19, density held 0.110. **BY-EAR:** `~/sm-generated/
   watchout_holdfix` (fixed) vs `watchout_holdbug` (broken; same seed → paired A/B).
3. **★ NEXT: Deploy swap** (after item 2 by-ear lands): coordinated `conditioning-mechanics §6` + `generation-defaults §0`
   version bump → checkpoint `gen_motif_v2_48th_cont` + default `--features highres_v2`. Consider making `b_trip=0.7`
   a v2 default (it WON by ear). Both v2 playability holes are now closed (no-fast-jump + hold-stream).
4. **Parked:** the manifold refit + groove-radar retrain (bundled, item 5 above); the seq-onset retrain
   ([[good-settings-region]], separate).

## AWAITING USER
- **The hold-stream fix by-ear** (item 2): `~/sm-generated/watchout_holdfix` (fixed) vs `watchout_holdbug` (broken,
  same seed) — does `freeze=high` now keep holds OUT of the streams? The no-fast-jump cap already PASSED by ear
  (`nofastjump_ab`). Once the hold-stream fix lands by ear, the only remaining item is the deploy swap (item 3).

## CANONICAL EXPORT DEFAULTS (the DEPLOYED v1 config — VALIDATED by `tools/check_export_defaults.py`; UNCHANGED by v2)
The bare `export_typed_samples.py` run reproduces what the user plays. These MUST equal the script's argparse
defaults. **Permanent section — keep in every rewrite.**
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
NOTE: v2 is a SEPARATE regime — `--features highres_v2` + `gen_motif_v2_48th_cont` + `for_v2()` + the 48th `sm_writer`
+ `V2_MSL=5400`. New v2 decode knobs (all default to a v1 no-op): `--min_onset_gap` (None→auto 2 on v2), the triplet
band via `--onset_phase_calib "0,1.0,b_trip"` (b_trip default 0). Do NOT mix `highres_v2` with the v1 checkpoint.

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/governor-subdiv-recalib`** (off `feat/data-layer-v2`, off `main`). 4 commits this session (above).
  The `/refresh` docs commit + PR are the last steps. **Verify PR state via `gh`.**
- Gitignored / not committed: `outputs/` (all playtest sets incl. pt_chaos_v2/pt_surprise_v2/footspeed_new/
  triplet_band_new), `transcripts/`, scratchpad probes.

## READ-FIRST (in order)
`notes/footspeed_floor_findings.md` (the floor + triplet band + the no-fast-jump diagnosis = the NEXT action) →
`notes/manifold_radar_subdiv_findings.md` (why the refit is DEFERRED, not done) → `conditioning-mechanics §8` (the
recalibrated governors + the fast-jump hole) → `generation-defaults §0/§1a` (v1 canonical + the v2 knobs) →
`notes/playtest_log.md` (this session's by-ear WIN + the Watch Out hold-stream bug) → lineage
`experiment_lineage/meter-grid-arc.md`. Load-bearing skills: **conditioning-mechanics §6/§8, generation-defaults,
experiment-design** (Rule 7 harness-first + "verify the coupling before asserting" — it caught the manifold-refit
regression this session).

## DISCIPLINE
**The no-fast-jump lever needs a BY-EAR re-A/B** (playability = play-feel). **Keep the note, fix the footing** (user:
don't remove pink notes). **subdiv=4 must stay byte-identical** (test every v2 decode change against v1). **Don't
rebuild the v2 cache + reuse the current checkpoint** (radar de-sync). One change at a time. Match the verb to the
evidence ([[claim-precision]]). HARNESS/COUPLING-first when a result looks wrong.
