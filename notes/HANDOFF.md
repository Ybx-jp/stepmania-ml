# HANDOFF — ★ SHIP MODE: cut v1.0.0 (deploy-swap v2 + guide). Research PARKED. Don't wander.

**Written 2026-07-06 for the next Claude. THE PROJECT IS IN SHIP MODE** (user decision — memory
[[ship-mode-park-research]]): clean up, cut **v1.0.0** (tags `0.1.0`/`0.2.0` already exist), host, announce, move on.
The v2 48th-grid model is "a full step better in expressiveness… already pretty incredible." The remaining gap ("onset
allocation undertuned") is the **note-context PLACEMENT CEILING** — a RETRAIN problem, not a decode-tune — so more
knob-tuning is diminishing returns. **Do NOT initiate the parked research** (good-settings tolerance formula,
GDL/meter-equivariance retrain, seq-onset retrain); redirect tangents to the release checklist. Un-park ONLY on the
user's literal phrase **"the times have changed."**

This session (before the ship decision): shipped the **no-fast-jump cap** (by-ear PASSED), a PARTIAL **hold-stream
subdiv fix** (real fix parked), and found **b_trip 0.7-vs-1.0 is song-dependent/inconclusive** — which is what tipped
the call to ship rather than keep tuning.

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

## THE v1.0.0 SHIP CHECKLIST (this is the active work — see [[ship-mode-park-research]])
1. **★ DEPLOY SWAP — make v2 the default.** Coordinated bump in `generation-defaults §0` + `conditioning-mechanics §6`
   + `export_typed_samples.py` argparse: default checkpoint → `gen_motif_v2_48th_cont`, default `--features highres_v2`,
   `--min_onset_gap` auto (already), `no_fast_jump` on (already), and set **`b_trip=0.7` as the v2 default**
   (`--onset_phase_calib "0,1.0,0.7"`; song-dependent, 0.7 is the gentle default — 1.0 adds triplet busyness on duple
   songs). Re-point the CANONICAL EXPORT DEFAULTS block below + re-run `tools/check_export_defaults.py`.
   ⚠️ The groove-radar chaos refit stays RETRAIN-GATED (do NOT refit the manifold; `manifold_radar_subdiv_findings.md`).
2. **Decide the hold-stream edge.** The `freeze=high` v2 free-foot-stream-under-hold defect is PARTIAL-fixed only
   (`footspeed_floor_findings.md §5/§5b`; real fix DESIGNED + PARKED = position-based `stamina_hold_bump`, ~6 lines).
   SHIP decision: does this freeze=high-only edge block v1.0.0, or ship as a documented known-limitation? (Lean: ship
   it — it only bites the extreme `--style freeze=high` combo; the parked fix is ready if a user hits it.)
3. **Safe-settings envelope + guide.** Characterize "the settings most pad players would ever actually attempt" (a
   conservative region) and write the user guide. Optional nicety (NOT required): auto-`b_trip` per-song from the
   subdivision statistic (SB DFT ρ+0.47) — triplet songs get the band, duple songs don't.
4. **Release:** clean up, cut the `v1.0.0` tag, host, announce ([[marketing-track]]; adapt `RELEASE_CRITERIA.md`).
5. **PARKED — do NOT start without "the times have changed":** the position-based hold-stream fix (§5b); the manifold
   refit + groove-radar retrain (bundled); the seq-onset retrain (the note-context placement ceiling); the
   good-settings tolerance formula ([[good-settings-region]]); GDL/meter-equivariance ([[meter-4-4-grid]]).

## AWAITING USER
- **Nothing blocking.** The b_trip A/B is installed (`~/sm-generated/gc_similar_bpm20_v2_full` = 0.7 vs
  `gc_similar_bpm20_v2_btrip10` = 1.0) but the verdict was "inconclusive/song-dependent" → resolved by shipping 0.7 as
  the default. Next user touchpoints are ship-checklist reviews (the deploy-swap diff, the guide), not a playtest gate.

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
- Branch **`feat/governor-subdiv-recalib`** (off `feat/data-layer-v2`, off `main`). This session added `df39c3c`
  (no-fast-jump cap) + `e964b1f` (hold-stream subdiv fix) + the `/refresh` docs commit. **Verify PR state via `gh`.**
  The DEPLOY SWAP (ship checklist #1) is a NEW code change not yet made.
- Gitignored / not committed: `outputs/` (all playtest sets incl. gc_similar_bpm20_v2_full / _btrip10, nofastjump_ab,
  watchout_holdfix), `transcripts/`, scratchpad probes.

## READ-FIRST (in order)
Memory **[[ship-mode-park-research]]** (the operative directive) → **this checklist above** → `generation-defaults §0`
(the deploy-swap targets: v2 checkpoint/features/knobs) → `conditioning-mechanics §6/§7/§8` (the v2 decode levers +
the hold-stream partial fix + stamina-is-ON correction) → `notes/footspeed_floor_findings.md §4/§5/§5b` (no-fast-jump
= shipped; hold-stream = partial + the PARKED fix design) → `notes/playtest_log.md` (the by-ear record incl. the
b_trip inconclusive verdict) → lineage `experiment_lineage/meter-grid-arc.md`. Load-bearing skills:
**generation-defaults, conditioning-mechanics §6/§8, experiment-design** (the ship-vs-tune call = a Rule-"is this the
right investment" judgment).

## DISCIPLINE
**SHIP MODE: don't wander into parked research** ([[ship-mode-park-research]]; un-park only on "the times have
changed"). **Match the metric to the FELT property** (the hold-stream defect was mis-analyzed TWICE by trusting an
aggregate over the raw grid — dump the grid). **Stamina is ON canonically (ceiling 50), NOT off** (the skill text
that said "off" burned a session — corrected `conditioning-mechanics §8c`). **subdiv=4 must stay byte-identical** on
any v2 decode change. **Don't rebuild the v2 cache + reuse the current checkpoint** (radar de-sync). One change at a
time; match the verb to the evidence ([[claim-precision]]); HARNESS/COUPLING-first when a result looks wrong.
