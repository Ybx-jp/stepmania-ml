# HANDOFF — ACTIVE: data-layer-v2 BUILD IN PROGRESS (48th grid + beat-sync). Cache building → retrain next.

**Written 2026-07-05 for the next Claude.** The 4/4-grid meter thread (triplet tax BY-EAR CONFIRMED) was
GREENLIT into the **data-layer-v2 refactor**, now BUILT through Phase 4 and mid-rebuild. This session: (1) shipped
the cheap inference-gate reach win, (2) confirmed the A1 fixed-48th grid design with fit + emptiness checks, (3)
built v2 phases 1→4 (timing spine → finer-grid quantization → feature re-grid → trainer) + phase 2b (beat-sync
audio), and (4) launched the bundled 2a+2b corpus cache build. **The deployed model is UNCHANGED** — v2 is a
version bump that has NOT replaced anything yet (no retrained checkpoint exists; the cache is still building).

## WHERE WE ARE
- **Deployed model UNCHANGED** = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres, 16th grid).
  Canonical decode defaults UNCHANGED (block below; validator passes). Nothing v2 is deployed.
- **v2 CACHE BUILDING** in the background: `cache/samples_v3_48th` (2a finer grid + 2b beat-sync), ~5–6 h, 4 cores.
  Verify done: `ls cache/samples_v3_48th/{train,val} | wc -l` should approach train ~4547 / val ~951 (v2 admits
  MORE difficulties than v1's 4452 — fewer floor-collision false-hands rejections). Log: `cache_v2_build.log`
  (gitignored). If it died, relaunch the command in `data_layer_v2_scope.md` phase 3 (DELETE the dir first — the
  cache stamp checks song identity but NOT feature config, so a stale partial dir is served as cache hits;
  [[dataset-cache-footgun]]).
- Branch: **`feat/data-layer-v2`**. All v2 commits `90288c2`…`f4321eb` (verify via `git log`).

## THE ACTIVE THREAD — data-layer-v2 (lineage `experiment_lineage/meter-grid-arc.md`, memory [[meter-4-4-grid]])
Refactor the whole pipeline off the hard-4/4 duple-16th grid onto a **48th grid (12 subdivisions/beat)** +
**beat-synchronous audio**. Full plan + phase status: `notes/data_layer_v2_scope.md`. State by phase:
- **0 Grid design — ✅ A1 fixed-48th CONFIRMED** (12/beat = LCM(duple-16th, triplet); NOT meter-adaptive → no
  fallible deploy-time detector). Fit check: 3× context fits (bf16); emptiness: 4.2% of notes are triplet payload,
  49% of songs gain nothing but the 3× is cheap → robustness wins.
- **1 Timing spine — ✅** `src/data/timing.py` `TimingMap` (beat↔time + STOPS + `frame_times`/`resample_frames`),
  9→ tests in `tests/test_timing.py`. The proven `bpm_map` core, ported + hardened.
- **2a Finer-grid quantization — ✅** parser `_beat_to_ts` + `round_quantize` + `StepManiaParser.for_v2(subdiv=12)`.
  Legacy 4-grid floor path BYTE-IDENTICAL. **Success criterion MET** (`probe_v2_displacement.py`): triplet-note
  displacement **50.5 → 0.3 ms** (ρ+0.808 reproduced the meter thread's +0.83). `tests/test_v2_quantize.py`.
- **2b Beat-sync audio — ✅** `audio_features.py` `beat_sync` flag + `_highres_pooled_onset_beatsync`. SIZING
  (`probe_v2_bpm_misalignment.py`): 2b is BIGGER than first assumed — ~20% of songs, **14.6% ≥23 ms drift** (double
  the triplet tax, second-scale on half-tempo sections). GATED on ACTUAL tempo variation → constant-BPM keeps the
  EXACT v1 features (verified 0.00000 diff); variable-BPM re-timed (0.755 diff). Rule-7: two probe bugs caught.
- **3 Feature re-grid — ✅ plumbed + DE-RISKED** `highres_v2` spec (`decode_harness`, tpb=12 + beat_sync, 42-dim,
  `cache/samples_v3_48th`) + `warm_cache_v2 --v2`. `metric_phase` → `t%12`/`t%48` AUTOMATICALLY (phase-5 free).
  Alignment de-risk (`probe_v2_alignment.py`): v2 audio==chart frames, exactly 3.00× v1. **Cache building now.**
- **4 Retrain — ✅ TRAINER READY + DE-RISKED, ⬜ NOT LAUNCHED** `train_motif_figure_v2.py`: warm-starts from
  `gen_motif_full_fixed` (verified clean load, only `pos_encoding.pe` filtered), bf16, FITTED config **T=3072 / batch
  4** (the training-shaped memory sweep — the earlier no-mask fit probe was optimistic; B8 OOMs). ~1.5 h. Onset loss
  unchanged focal_bce; the 48th target is ~3× SPARSER → WATCH val_onset epoch 1, retune gamma/pos_weight only if
  recall collapses.
- **5 Decode-side phase-vocab re-index — ⬜ PENDING (for export/gen only):** `onset_phase_calib` (16th-unlock),
  `phase_shares`, SB are still `t%4` — must become `t%12` bands BEFORE a v2 chart can be EXPORTED/played. Not needed
  for training (metric_phase is index-driven). Do this at Phase 6.
- **6 By-ear validation — ⬜ THE BINDING GATE:** export a triplet song (First of the Year / My Christmas list) with
  the v2 model + `for_v2` features, PLAY it — did the limp go away? Set songs are in `~/sm-generated/meter_triplet_test/`.

## THE OPEN FORK / NEXT ACTION
1. **When the cache finishes → launch Phase 4 retrain** (pre-configured, de-risked): `python experiments/
   generation_typed/train_motif_figure_v2.py --data_dir data --audio_dir data`. WATCH val_onset (sparser target).
2. **Then Phase 5** (decode-side `t%12` re-index) → **Phase 6 by-ear** (the gate). Only after by-ear passes does v2
   become a deploy candidate (a coordinated `conditioning-mechanics §6` + `generation-defaults` version bump).
3. Parked elsewhere: the seq-onset anchoring retrain (the MUSICALITY cliff — [[good-settings-region]]); v2 is the
   DATA-LAYER fix, complementary not a substitute.

## AWAITING USER
- No pending playtest verdict (the meter triplet-test set already CONFIRMED the tax; the v2 re-export is the NEXT
  by-ear, not yet generated — needs the retrained model). Next user touchpoint = Phase 6 v2 re-export by-ear.

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`; UNCHANGED by v2)
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

NOTE: v2 export will use `--features highres_v2` + a v2 checkpoint (once retrained) + the Phase-5 re-indexed
phase levers — a SEPARATE regime from this deployed block. Do NOT mix v2 features with the v1 checkpoint.

Also shipped this session (independent of v2): the CHEAP inference-gate reach win — `StepManiaParser.for_inference()`
(BPM `[40,320]`, length `[30,600]s`, gimmick guard on raw `#BPMS` > 400) + `export_typed_samples.py --relax_gates`
(forces `cache_dir=None`). Training path byte-identical.

## BRANCH / PR STATE  (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/data-layer-v2`** (off `feat/inference-gate-relaxation`, off `main`). Commits `1cb5e3d` (cheap win)
  → `f726063` (A1 checks) → `90288c2`…`f4321eb` (v2 phases) + this refresh. **Verify pushed state / PRs via `gh`.**
- New tooling this session: `probe_v2_{context_fit,grid_emptiness,displacement,alignment,bpm_misalignment}.py`,
  `train_motif_figure_v2.py`, `src/data/timing.py`, `tests/test_{timing,v2_quantize}.py`,
  `notes/data_layer_v2_scope.md`, memory `autotune-skill-stale.md`.
- Gitignored / not committed: `cache_v2_build.log`, `cache/samples_v3_48th/` (reproducible).

## READ-FIRST (in order)
`notes/data_layer_v2_scope.md` (the whole v2 plan + phase status + the CHECKS + the memory corrections) → lineage
`experiment_lineage/meter-grid-arc.md` (meter tax → v2 build arc) → `notes/meter_4_4_assumption_scope.md` (the
diagnosis that justified v2) → (for export later) `conditioning-mechanics §6` (the `t%4` grid + its 4/4 flag, which
Phase 5 re-indexes). Load-bearing skills: **conditioning-mechanics §6**, **generation-defaults** (v1 canonical +
the new `highres_v2`/`for_v2`), **experiment-design** (Rule 7 caught 2 probe bugs + the no-mask fit-probe optimism
this session), **autotune** ([[autotune-skill-stale]] — benchmark `LayeredTypedChartGenerator` DIRECTLY, not the
skill's `train_factorized`).

## DISCIPLINE
Rule 0 (the scope already tracks every phase). **By-ear (Phase 6) is the binding gate.** **Measure training-shaped
memory, not a bare `model()` call** (the no-mask fit probe was 2× optimistic → B8 OOM). **DELETE the cache dir on
any feature-CONFIG change** (identity stamp ≠ config; [[dataset-cache-footgun]]). Rule 7: a surprising number gets
the fair test AND an eyeball of the artifacts (2b sizing = a probe bug hiding a real effect). One change at a time.
Match the verb to the evidence ([[claim-precision]]).
