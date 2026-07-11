# HANDOFF — ★ v1.0.0 SHIP: v2 is the DEFAULT + the BYO offset detector is WIRED. Remaining = regen personal set, cut/host/announce.

**Written 2026-07-09 for the next Claude.** Ship mode is the standing directive ([[ship-mode-park-research]]: cut
v1.0.0, host, announce; don't wander into parked research — un-park ONLY on the literal phrase "the times have
changed"). This session cleared the two biggest ship-checklist blockers: the **v2 deploy-swap** and the **BYO
beat-anchor offset detector**. What's left is mechanical (regenerate the deliverable, document one limitation, cut).

## WHERE WE ARE
- **✅ v2 IS THE DEFAULT (2026-07-09).** Both CLIs (`scripts/generate.py` + `export_typed_samples.py`) default to
  `--features highres_v2` + `checkpoints/gen_motif_v2_48th_cont/best_val.pt` (42-dim, 48th grid). The v2-only auto
  knobs (`grid_snap auto`, `auto_b_trip` with `b_trip=0.7`) ship ON; the auto-vs-global b_trip by-ear A/B was WAIVED
  for the cut (offline-validated + `toulouse_win_anchor` "cut v1" verdict). `--features highres` still reaches the
  legacy v1 model (`gen_motif_full_fixed`). Validator 25 ✓, 44/44 tests, end-to-end 48-row output confirmed.
- **✅ BYO "deaf choreography" FIXED — the offset detector is wired.** `src/data/offset_detect.py` recovers `#OFFSET`
  from audio (full-band onset pulse-train + 31 ms latency cal); `generate.py` uses it as the DEFAULT beat-anchor
  (extraction skips the detected within-beat phase → frame 0 = downbeat; writes `#OFFSET = −phase` so the untrimmed
  audio still plays in time). Re-validated vs the `~/sm-personal` oracle: **median 4.6 ms, 19/23 ≤40 ms, Toulouse
  7.1 ms**. The ~20% half-beat slips: `--offset <reference #OFFSET>` override (the confidence flag is weak, 2/4 — do
  NOT lean on it). `--no_auto_offset` = pre-fix t=0. Also: BPM STILL must be user-supplied (`--bpm`; estimation is
  unreliable, [[byo-audio-bpm-footgun]]). Details `notes/byo_offset_detection_findings.md`, tests
  `tests/test_offset_detect.py`.
- **✅ `--harm_calib` added to `generate.py`** (the arg-parity sweep result — the one validated lever the public CLI
  lacked; sparse-harm-in-quiet onset phrase calibrator, off by default). The full canonical decode stack is otherwise
  shared via `CANONICAL_DECODE` so the two CLIs can't drift.
- **Sliding-window onset** (Ch.1): BUILT + by-ear PASSED (`toulouse_win_anchor`); no-op when a song fits the trained
  context. The harder decoder-side windowing stays parked (not needed).
- **✅ BYO acquisition/assembly TOOLING (2026-07-10, tooling not model; `notes/byo_audio_acquisition_tooling.md`,
  [[audio-acquisition-tooling]]).** `generate.py --audio` now accepts a YouTube/yt-dlp URL (→ Vorbis `.ogg` cached
  by video id; also `scripts/pull_audio.py`); `--trim-audio START[,END]` slices a range BEFORE gen; `--sm_difficulty`
  writes the real `.sm` slot (**default changed**: follows `--difficulty`, was hardcoded `Challenge`); `--append_to
  CHART.sm` splices a difficulty into an existing song (bpm/subdivision grid guards + `.bak`). Deps: yt-dlp+ffmpeg
  (+deno) on PATH. ⚠️ Load-bearing: playback `#OFFSET` ≠ generation beat-anchor — all difficulties of one song MUST
  share one `--offset` ([[byo-audio-bpm-footgun]] 4th mode). On branch `feat/youtube-audio-pull-trim-append`.

## ★ ACTIVE THREAD — the v1.0.0 CUT (nothing research-y is blocking)
Lineage: `meter-grid-arc.md` (v2 default) + `byo-audio-alignment-arc.md` **Ch.2** (offset, now CLOSED for the ship).

## THE v1.0.0 SHIP CHECKLIST
1. **✅ BYO alignment** — offset detector + beat-anchoring wired into `generate.py`, oracle-validated.
2. **✅ DEPLOY SWAP** — v2 default in both CLIs (validator 25 ✓).
3. **Hold-stream `freeze=high` edge** — ships as a documented KNOWN LIMITATION (user call 2026-07-09): a 5–6 beat
   hold with a one-foot stream under it; only bites `freeze=high` conditioning. The real position-based fix is
   DESIGNED + PARKED (`notes/footspeed_floor_findings.md §5b`). ▷ TODO: write it into the release notes / README.
4. **REGEN the personal deliverable** — `~/sm-generated/v2_personal_hard` (34 charts) was built PRE-fix → wrong
   BPM/offset/density + pre-sliding-window. Regenerate with the new default (v2 + auto offset detector); pass `--bpm`
   per song (from the user or their `~/sm-personal` reference charts) and `--offset` where the detector flags low
   confidence. ▷ This is the end-to-end confirmation that the deaf-chore fix landed on real songs (the fix is unit +
   oracle validated but not yet by-ear'd on a freshly-regenerated personal chart).
5. **Release:** cut `v1.0.0`, host, announce ([[marketing-track]]).
6. **PARKED (needs "the times have changed"):** seq-onset retrain (placement ceiling), GDL/meter-equivariance,
   good-settings tolerance formula, manifold refit + groove-radar retrain, decoder-side sliding window.

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py` = 25 ✓)
The bare `export_typed_samples.py` run reproduces what the user plays. These MUST equal the script's argparse
defaults. As of 2026-07-09 the DEFAULT regime is **v2** (`gen_motif_v2_48th_cont` + `highres_v2`); `grid_snap`/
`auto_b_trip`/`grid_snap_keep_triplets`/`triple_pref_thresh` are now ACTIVE (they were v1 no-ops before the swap).
**Permanent section — keep in every rewrite.** NOTE: the BYO density `×4/subdiv` fix + the offset detector +
`--harm_calib`/`--offset` are `scripts/generate.py` behaviors and do NOT touch the exporter argparse defaults below
(generate.py imports `CANONICAL_DECODE` for the shared palette only; the `--harm_calib`/`--harm_quiet_q` lines below
ARE exporter args too, at their off defaults).
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
NOTE: v2 is a SEPARATE regime — `--features highres_v2` + `gen_motif_v2_48th_cont` + `for_v2()` + the 48th `sm_writer`
+ `V2_MSL=5400`. v2 decode knobs (all v1 no-ops): `--min_onset_gap` (None→auto 2 on v2); `--grid_snap auto`; the
triplet band via `--onset_phase_calib "0,1.0,b_trip"`; `--auto_b_trip`. Do NOT mix `highres_v2` with the v1 checkpoint.

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/byo-sliding-window-onset`** (off `main`; pushed) holds the v2 deploy-swap + `--harm_calib` + the
  offset detector + that docs refresh. **Verify its PR state via `gh pr list`; do not trust a number written here.**
- Branch **`feat/youtube-audio-pull-trim-append`** (off `feat/byo-sliding-window-onset`, 2026-07-10) = the BYO
  acquisition/assembly tooling (URL `--audio`, `--trim-audio`, `--sm_difficulty`, `--append_to`) + this docs refresh.
  Its PR is based on `feat/byo-sliding-window-onset` (clean diff, since the generate.py edits depend on that branch);
  **verify number/state/base via `gh pr view` — retarget to `main` if byo lands first.**
- Gitignored / not committed: `outputs/`, `transcripts/`, scratchpad probes (`$CLAUDE_JOB_DIR/tmp/*`). Untracked and
  NOT mine: `.claude/commands/begin.md` (left unstaged). Test charts under `~/sm-generated/` are gitignored.

## READ-FIRST (in order)
Memory [[ship-mode-park-research]] (directive) → the SHIP CHECKLIST above (what's left for v1.0.0) →
`generation-defaults §0` (v2 is now the default) → **`notes/byo_offset_detection_findings.md`** (the wired offset
detector) → lineage `experiment_lineage/{meter-grid-arc.md, byo-audio-alignment-arc.md Ch.2}` → memories
[[meter-4-4-grid]], [[personal-reference-charts]], [[byo-audio-bpm-footgun]]. Load-bearing skills:
**generation-defaults, conditioning-mechanics §6, experiment-design**.

## DISCIPLINE
**Match the fix to the FELT property** (density≠placement — the `×4/subdiv` fix was real but the wrong axis; the ears
caught it). **A playback-sync argument is NOT a choreography argument** (the "offset red herring" overturn). **Use the
oracle** (`~/sm-personal`) to validate an audio estimator BEFORE trusting it — it killed 3 confident-but-wrong
detector variants (DFT/kick/tiebreak) and just RE-validated the productionized detector (median 4.6 ms). **BPM must
be user-supplied.** Ship mode: don't wander into parked research. One change at a time; match the verb to the
evidence ([[claim-precision]]).
