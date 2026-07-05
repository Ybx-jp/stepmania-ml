# HANDOFF — data-layer-v2: PHASE 6 BY-EAR PASSED. v2 is a DEPLOY CANDIDATE. Next = governor subdiv-recalibration.

**Written 2026-07-05 for the next Claude.** The 4/4-grid meter thread (triplet tax) reached its **binding gate and
CLEARED it**: the data-layer-v2 48th grid was exported for the triplet songs, PLAYED, and the tax is GONE with zero
degradation ("resounding 100% success… finally able to REALLY express tasty percussion"). v2 is now a **deploy
candidate**. The deployed model is STILL UNCHANGED (v1) — one thing stands between v2 and deployment: the decode
**governors are miscalibrated on the 48th grid** ("playability constraints kinda fell apart"), a bounded recalibration.

## WHERE WE ARE
- **Deployed model UNCHANGED** = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres, 16th grid). Canonical
  decode defaults UNCHANGED (block below; validator passes). Nothing v2 is deployed yet.
- **v2 VALIDATED BY EAR (2026-07-05).** Best checkpoint `checkpoints/gen_motif_v2_48th_cont/best_val.pt` (val_total
  **0.7435**, epoch 19; below the base run's 0.8098). Phase 6 playtest logged in `notes/playtest_log.md` (top entry).
- **v2 EXPORT TOOLING BUILT + committed `837c1ed`** (the Phase-6 prerequisite that had never been wired):
  - `src/generation/sm_writer.py`: rows-per-measure parameterized by `timesteps_per_beat` (48th grid → 48 rows/measure
    so triplets land at true 1/3-beat positions). **subdiv=4 verified BYTE-IDENTICAL** to before.
  - `experiments/generation_typed/export_typed_samples.py`: `--features highres_v2` (auto-selects the `for_v2()` parser
    + the 48th `sm_writer` + `cache_dir=None`), builds the model at `max_len=5504`, and the **msl-truncation FIX** (the
    v1 config `msl=1440` = only 120 beats on the 3×-finer grid → clipped every song to ⅓; use `V2_MSL=5400`).
- **Cross-arc probe** `notes/seqonset_v2grid_findings.md` (`probe_seqcontext_frozenh_v2.py`): audio is CHANCE at triplet
  placement (0.505 vs note-context ceiling 0.930) → v2 fixed the TARGET (triplets placeable by the trained prior / in
  the decoder `h`), NOT audio-derivability. The seq-onset wall is corroborated on the finer grid.
- **Branch: `feat/data-layer-v2`.** This session's commits: `837c1ed` (phase-6 tooling + probe) → the docs(refresh)
  commit carrying this HANDOFF. Verify via `git log`. A PR to `main` is being opened as the last refresh step.

## THE ACTIVE THREAD — data-layer-v2 (lineage `meter-grid-arc.md`, memory [[meter-4-4-grid]])
The 48th-grid (12/beat) + beat-sync refactor off the hard-4/4 duple-16th grid. **BUILD ARC COMPLETE** (phases 0–6 all
done; per-phase detail `notes/data_layer_v2_scope.md`). Phase 6 by-ear ✅ PASSED. The thread is now at DEPLOYMENT.

## THE OPEN FORK / NEXT ACTION
1. **Governor subdiv-recalibration (the binding blocker before deploy).** The §8 decode governors
   (`max_jack_run`, fatigue/stamina, hold_stream) compute `frame_hz = BPM·4/60` and reason in FRAMES assuming a frame
   = a 16th. On the 48th grid a frame is 1/12 beat (3× finer), so jack-adjacency, fatigue/stamina rates, and the hard
   caps are ~3× miscalibrated — the user's "playability constraints kinda fell apart." **Thread `subdiv` into `frame_hz`
   (`BPM·subdiv/60`) and the constraint spacings** (`conditioning-mechanics §8`; the skill's `frame_hz` "analysis-only"
   claim was CORRECTED this session — it IS decode-critical on the finer grid). One change at a time; re-A/B by ear.
2. **Deploy swap (only after #1).** A coordinated `conditioning-mechanics §6` + `generation-defaults §0` version bump:
   swap the deployed checkpoint → `gen_motif_v2_48th_cont` + default `--features highres_v2`. NOT before the governors
   are playable.
3. **Parked leads:** (a) the retrain-HP/Optuna question (`data_layer_v2_scope.md` PARKED LEADS — Optuna deferred:
   val_total is blind to placement; the descent is pattern-head-only, onset converged at epoch 1). (b) A triplet phase
   band for the songs that still hedge onto duple-16ths (First of the Year gen 0.14 vs human 0.40) — only if the
   governor re-tune doesn't lift it. (c) The seq-onset retrain (musicality cliff, [[good-settings-region]]) — separate.

## AWAITING USER
- **Nothing pending a user verdict.** The Phase-6 playtest is DONE and logged. The next work (governor recalibration)
  is Claude-side; a re-A/B playtest of the recalibrated governors on the 48th grid will be the next user touchpoint.

## CANONICAL EXPORT DEFAULTS (the deployed v1 config — VALIDATED by `/refresh`; UNCHANGED by v2)
The bare `export_typed_samples.py` run reproduces what the user plays. These MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` FAILS the refresh if they drift. **Permanent section — keep in every rewrite.**

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

NOTE: the v2 export is a SEPARATE regime — `--features highres_v2` + a v2 checkpoint (`gen_motif_v2_48th_cont`) + the
`for_v2()` parser (auto-selected by the exporter now). It uses the 48th-grid writer + `V2_MSL=5400`. Do NOT mix
`highres_v2` with the v1 checkpoint, or a v1 checkpoint with the v2 cache (grid mismatch). The canonical block above is
the DEPLOYED v1 config and stays authoritative until the deploy swap (fork item 2).

Also shipped earlier (independent of v2): the CHEAP inference-gate reach win — `StepManiaParser.for_inference()`
(BPM `[40,320]`, length `[30,600]s`, gimmick guard) + `export_typed_samples.py --relax_gates`.

## v2 EXPORT — the exact invocation (reproduce the Phase-6 set)
```
python experiments/generation_typed/export_typed_samples.py --data_dir data --audio_dir data \
  --checkpoint checkpoints/gen_motif_v2_48th_cont/best_val.pt --features highres_v2 \
  --song_filter "first of the year,my christmas list" --hardest --num_songs 2 \
  --out_dir outputs/meter_triplet_test_v2 --install --songs_dir /home/ybx/sm-generated
```
The exporter auto-raises `--max_len`→5400 and `msl`→5400 for `highres_v2`. Installed set: `~/sm-generated/
meter_triplet_test_v2/` (A/B vs the v1 `meter_triplet_test/`). ⚠️ StepMania CACHES the song list — after re-installing a
folder, clear `~/.stepmania-5.1/Cache/Songs/*<setname>*` + `Cache/index.cache` and restart the game (done this session).

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/data-layer-v2`** (off `feat/inference-gate-relaxation`, off `main`; `main` IS an ancestor). This
  session added `837c1ed` (phase-6 tooling + probe) + this docs(refresh) commit. A PR `feat/data-layer-v2 → main` is
  opened as the final refresh step (merging the v2 build; deploy is a SEPARATE later step). **Verify PR state via `gh`.**
- Gitignored / not committed: `train_v2_48th*.log`, `probe_v2grid.log`, `cache/samples_v3_48th/`,
  `cache/seqctx_frozenh_v2_*.npz`, `outputs/`, `transcripts/`.

## READ-FIRST (in order)
`notes/data_layer_v2_scope.md` (per-phase status incl. Phase 6 PASSED + PARKED LEADS) → lineage
`meter-grid-arc.md` (the meter tax → build arc → Phase 6 pass + the two harness-bug catches) → `conditioning-mechanics
§8` (the governors that need the subdiv-recalibration — the `frame_hz` correction) → `generation-defaults §0` (v1
canonical + the `highres_v2` deploy-candidate regime) → `notes/playtest_log.md` (the Phase-6 verdict). Load-bearing
skills: **conditioning-mechanics §6/§8**, **generation-defaults §0**, **experiment-design** (the HARNESS-first Rule 7 —
it's what caught BOTH Phase-6 export bugs: the hard-16th writer AND the msl truncation the user flagged as 150-vs-450).

## DISCIPLINE
**The next lever (governor recalibration) needs a BY-EAR re-A/B** — playability is a play-feel property. **Verify
volatile state at read time** (checkpoint val, PR status, StepMania cache) — never trust a number written here as
current. **DELETE the cache dir on any feature-CONFIG change** ([[dataset-cache-footgun]]). **Don't pair `highres_v2`
with the v1 checkpoint.** One change at a time. Match the verb to the evidence ([[claim-precision]]). HARNESS-first when
a result looks wrong (the 150-vs-450 taps was a truncation bug, not the model — the user's instinct was right).
