# HANDOFF — ACTIVE: data-layer-v2 BUILT THROUGH PHASE 5. Only Phase 6 by-ear (deployment gate) remains.

**Written 2026-07-05 for the next Claude.** The 4/4-grid meter thread (triplet tax BY-EAR CONFIRMED) was GREENLIT
into the **data-layer-v2 refactor**, now BUILT through **Phase 5** (grid design → timing spine → 2a/2b re-grid →
cache → Phase-4 retrain → Phase-5 decode re-index all ✅). This session: (1) built the 48th cache, (2) ran the
Phase-4 warm-started retrain + a continuation, (3) implemented **Phase 5** (the decode-side `t%4→t%12` re-index),
(4) added transcript-export tooling wired into `/refresh`. **The deployed model is UNCHANGED** — v2 is NOT deployed;
**Phase 6 by-ear is the single binding gate before it can be.**

## WHERE WE ARE
- **Deployed model UNCHANGED** = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres, 16th grid).
  Canonical decode defaults UNCHANGED (block below; validator passes). Nothing v2 is deployed.
- **v2 CHECKPOINTS EXIST (NOT deployed):** `checkpoints/gen_motif_v2_48th/best_val.pt` (Phase-4 retrain, val_total
  **0.8098**, epoch 20 — still descending, never plateaued). A CONTINUATION `checkpoints/gen_motif_v2_48th_cont/
  best_val.pt` (warm-started from that ckpt, `--warmup_freeze 0 --epochs 20 --patience 3`) is improving further
  (~0.767 by epoch 11 when this was written). **Verify live state:** `tail train_v2_48th_cont.log`; `kill -0 1877046`.
  Use the `_cont` checkpoint if it finished lower than 0.8098; else the base one.
- **Phase 5 DONE (commit `590daa1`):** the decode phase grid is parameterized by `subdiv` (timesteps/beat). A
  `--features highres_v2` export now drives the phase bands + tau on the 48th grid automatically.
- **Branch: `feat/data-layer-v2`.** Commits this session: `590daa1` (phase 5) → `7482401` (transcript tooling) →
  the docs(refresh) commit that carries this HANDOFF. Verify via `git log`.

## THE ACTIVE THREAD — data-layer-v2 (lineage `experiment_lineage/meter-grid-arc.md`, memory [[meter-4-4-grid]])
Refactor off the hard-4/4 duple-16th grid onto a **48th grid (12/beat)** + **beat-synchronous audio**. Full plan +
per-phase status: `notes/data_layer_v2_scope.md`. State:
- **0 grid design ✅** A1 fixed 48th confirmed (fit + emptiness checks).
- **1 timing spine ✅** `src/data/timing.py TimingMap`.
- **2a finer quantization ✅** `StepManiaParser.for_v2()` — triplet displacement 50.5→0.3 ms.
- **2b beat-sync audio ✅** `audio_features.beat_sync` (gated on tempo variation; constant-BPM keeps EXACT v1).
- **3 feature re-grid ✅** `highres_v2` spec; `metric_phase` re-indexes to `t%12`/`t%48` AUTOMATICALLY via config.
- **4 retrain ✅** cache built (train 4547 / val 951). `train_motif_figure_v2.py` warm-started `gen_motif_full_fixed`
  (only `pe` filtered), bf16, T=3072/B4. **`val_onset` never collapsed** (~0.025 throughout — the sparse-target worry
  didn't materialize; pattern loss was the mover). Still-descending at epoch 20 → `--epochs 30` is the cheap lever if
  by-ear is close. NOTE: training loss can't confirm the win (triplet placement is invisible to `val_total`).
- **5 decode-side phase re-index ✅ (commit `590daa1`):** `decode_defaults.phase_band_positions(subdiv)` is the
  single band-math source (8th=`subdiv//2`, 16th=`{subdiv//4, 3·subdiv//4}`); used by `apply_phase_calib` (tau) +
  `generate()` calib/penalty/alloc + `phase_shares` + the chaos gate; `subdiv` threaded from
  `feat_ext.config.timesteps_per_beat` into BOTH tau and generate. **subdiv=4 BYTE-IDENTICAL to v1** (verified).
  Deliberate deferral: TRIPLETS get no phase band (retrained weights place them; a triplet-unlock is unvalidated) —
  add one only if Phase-6 by-ear shows triplet under-placement. SB/tolerance + governor `frame_hz` still `t%4`
  (analysis-only, not decode-critical). ⚠️ **Two different `t%12` conversions** — metric_phase (INPUT, Phase 3) ≠
  the decode levers (Phase 5); a prior session did the former and thought Phase 5 was done. Verify at the code.
- **6 by-ear validation ⬜ THE BINDING GATE:** export a triplet song (First of the Year / My Christmas list) with the
  v2 checkpoint + `--features highres_v2` + `for_v2()` features, PLAY it — did the limp go away? Set is in
  `~/sm-generated/meter_triplet_test/`. Do NOT mix `highres_v2` with the v1 checkpoint (grid mismatch).

## THE OPEN FORK / NEXT ACTION
1. **Phase 6 by-ear (the gate).** Pick the lower-val v2 checkpoint (verify `_cont` vs base), export the triplet set
   with `highres_v2` + `for_v2` at canonical decode defaults, install to `~/sm-generated/`, and have the user play it.
   That is the single decision that turns v2 from a build into a deploy candidate.
2. **Only after by-ear passes** does v2 become a deploy candidate (a coordinated `conditioning-mechanics §6` +
   `generation-defaults` version bump: swap the deployed checkpoint + default `--features`).
3. Parked elsewhere: the seq-onset anchoring retrain (the MUSICALITY cliff — [[good-settings-region]]); v2 is the
   DATA-LAYER fix, complementary not a substitute.

## AWAITING USER
- **Phase 6 v2 re-export by-ear** — not yet generated (needs the chosen v2 checkpoint). This is the next user
  touchpoint; log the verdict to `notes/playtest_log.md`. The meter triplet-test set already CONFIRMED the tax on v1;
  Phase 6 asks whether v2 REMOVES it.

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

NOTE: v2 export uses `--features highres_v2` + a v2 checkpoint + `for_v2()` — a SEPARATE regime from this deployed
block. The Phase-5 re-index makes the phase levers above (esp. `onset_phase_calib`) grid-correct on v2 automatically
(subdiv derived from the feature spec). Do NOT mix v2 features with the v1 checkpoint.

Also shipped earlier (independent of v2): the CHEAP inference-gate reach win — `StepManiaParser.for_inference()`
(BPM `[40,320]`, length `[30,600]s`, gimmick guard) + `export_typed_samples.py --relax_gates`.

## NEW TOOLING THIS SESSION
- **`tools/export_transcript.py`** — renders a session's on-disk JSONL to markdown for learning-material mining
  (prose + tool calls + truncated results; thinking isn't persisted on disk). **Wired into `/refresh` step 6b** →
  gitignored `transcripts/`. `/export` is interactive (a skill can't call it) → this reads the JSONL directly.
  Memory [[transcript-export-learning]].

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/data-layer-v2`** (off `feat/inference-gate-relaxation`, off `main`). This session's commits:
  `590daa1` (phase 5 decode re-index) → `7482401` (transcript tooling) → this docs(refresh). **Verify pushed state /
  PRs via `gh`.** v2 is NOT ready for a PR to `main` (Phase 6 by-ear is the gate) — no premature merge.
- Gitignored / not committed: `train_v2_48th.log`, `train_v2_48th_cont.log`, `cache/samples_v3_48th/`, `transcripts/`.

## READ-FIRST (in order)
`notes/data_layer_v2_scope.md` (the v2 plan + per-phase status incl. Phase 5 done + the two-`t%12` note) → lineage
`experiment_lineage/meter-grid-arc.md` (meter tax → build arc → Phase 4/5 done) → `conditioning-mechanics §6` (the
phase grid, now `subdiv`-parameterized) → `generation-defaults §0` (v1 canonical + the `highres_v2` regime). Load-
bearing skills: **conditioning-mechanics §6**, **generation-defaults**, **experiment-design** (Rule 7 / HARNESS-first
— it's what disambiguated the "Phase 5 already done?" question this session).

## DISCIPLINE
**Phase 6 by-ear is the binding gate** (training loss can't confirm the triplet fix — it's a placement property).
**Verify volatile state at read time** (training PID, checkpoint val, PR status) — never trust a number written here
as current. **DELETE the cache dir on any feature-CONFIG change** (identity stamp ≠ config; [[dataset-cache-footgun]]).
**Don't pair `highres_v2` with the v1 checkpoint.** One change at a time. Match the verb to the evidence
([[claim-precision]]).
