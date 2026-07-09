# HANDOFF — ★ ACTIVE THREAD: BYO "deaf choreography" = audio↔beat-grid MISALIGNMENT (offset/BPM). Ship mode still on.

**Written 2026-07-08 for the next Claude.** Ship mode is still the standing directive ([[ship-mode-park-research]]:
cut v1.0.0, don't wander into parked research), BUT the live work is a **shippable deliverable-quality bug**: the
BYO personal charts (`~/sm-generated/v2_personal_hard`) play "randomly hard / no choreography / like the model is
deaf" (user by-ear). This is a HARNESS bug in `scripts/generate.py`, not the parked placement ceiling — so fixing it
is ship work, not research. Un-park research ONLY on the literal phrase "the times have changed."

## WHERE WE ARE
- **Deployed model STILL v1** (`checkpoints/gen_motif_full_fixed/best_val.pt`, 42-dim 16th grid). No model change.
  The v2 deploy-swap (ship checklist below) is still pending and UNSTARTED.
- **This session's thread = BYO CHOREOGRAPHY/OFFSET** (`notes/byo_offset_detection_findings.md`, lineage
  `byo-audio-alignment-arc.md` **Ch.2**, memory [[personal-reference-charts]] + [[byo-audio-bpm-footgun]]). The
  `~/sm-generated/v2_personal_hard` deliverable was generated BEFORE this thread → it is DEAF (wrong BPM/offset +
  ~2× density) and should be regenerated once the fix lands.
- **Root cause (found + partially fixed):** `generate.py` never beat-aligns the audio. The model chores on
  `metric_phase` (beat-phase), so a shifted grid → phantom-grid placement = "deaf." **Two parts:**
  1. **BPM** — Toulouse was charted at the librosa estimate **129.199** vs true **128** (from the user's reference
     chart); +0.9% drifts a full beat every ~51s. (Corroboration: `auto_b_trip` false-fired TRIPLET at 129.199,
     correctly read duple at 128.) BPM must be user-supplied (estimation separately unreliable, [[byo-audio-bpm-footgun]]).
  2. **OFFSET** — `build_stub_chart` hardcodes `offset=0.0` (`generate.py:79`); frame 0 must be the first beat. **This
     OVERTURNS the prior "offset is a red herring" note** (that was playback-only reasoning; see the arc).
- **APPLIED (uncommitted→committed this refresh):** the density `×4/subdiv` fix (`generate.py:272`) — v2 BYO charts
  ran ~2× real-Hard density because generate.py never got the exporter's manifold-density grid correction. REAL bug,
  but the WRONG axis for the deafness (count≠placement; the real-chart oracle showed placement RATIO already matched
  real). It's a correctness fix, kept.
- **Auto-offset detector (chosen, NOT yet wired):** full-band onset pulse-train + a 31ms latency calibration recovers
  the offset to ~7ms on ~80% of songs (validated vs `~/sm-personal` + `data/external` packs as a regression oracle);
  ~20% half-beat-slip → needs a confidence flag. DFT-phase / kick-band / kick-tiebreak were all oracle-REFUTED
  (worse). `librosa.beat_track` segfaults in this env. **User decision: ship the detector as the UNIVERSAL offset
  source** (not reference-chart inheritance) + a confidence flag for the ambiguous ~20%.

## ★ AWAITING USER — the binding BY-EAR GATE (blocks the generate.py wiring)
- **Anchoring A/B on Toulouse** (both at true 128 BPM + density-fixed, playback-synced; judge CHOREOGRAPHY):
  - **Arm A** = `~/sm-generated/toulouse_bpm128/` "Toulouse BPM128" — frame 0 = audio start (training convention).
  - **Arm B** = `~/sm-generated/toulouse_anchor_beat/` "Toulouse ANCHORbeat" — frame 0 = true downbeat (audio trimmed
    0.281s = the reference `#OFFSET`).
  - **Question:** does Arm B choreograph noticeably better than Arm A (→ wire the detector's offset to the EXTRACTION
    anchor, skip-to-first-beat), or a wash (→ keep t=0, only write `#OFFSET`)? Also: is EITHER finally not "deaf"?
  - ⚠️ **Anchor + written `#OFFSET` must move together** or playback desyncs. Wrinkle: the dataset only skips POSITIVE
    offsets (`audio_features.py:203`), so negative-offset songs (Toulouse −0.281) trained anchored at t=0. Log to
    `notes/playtest_log.md`.
- **After the verdict:** wire the detector + chosen anchoring into `generate.py`, then REGEN the whole
  `v2_personal_hard` set (all 34 were built pre-fix → wrong BPM/offset/density). BPM per song from the user or their
  `~/sm-personal` reference charts.
- (Superseded/of lower priority now: the earlier grid-snap low-diff A/B and auto-vs-global b_trip A/B — still valid
  but the deploy-swap is gated behind getting BYO alignment right first.)

## THE v1.0.0 SHIP CHECKLIST (still standing; BYO alignment is the current blocker on the personal deliverable)
1. **BYO alignment fix** (THIS thread): land the offset detector + anchoring in `generate.py`; regen the personal set.
2. **★ DEPLOY SWAP — make v2 the default** (`generation-defaults §0` + `conditioning-mechanics §6` + exporter argparse:
   checkpoint→`gen_motif_v2_48th_cont`, `--features highres_v2`, `b_trip=0.7` default). Groove-radar chaos refit stays
   RETRAIN-GATED (don't refit the manifold). Re-run `tools/check_export_defaults.py`.
3. **Hold-stream edge** (`freeze=high` v2): ship as documented known-limitation or apply the PARKED position-based fix.
4. **Release:** cut `v1.0.0`, host, announce ([[marketing-track]]).
5. **PARKED (needs "the times have changed"):** seq-onset retrain (placement ceiling), GDL/meter-equivariance,
   good-settings tolerance formula, manifold refit + groove-radar retrain, the sliding-window PE refinement.

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py` = 25 ✓)
The bare `export_typed_samples.py` run reproduces what the user plays. These MUST equal the script's argparse
defaults. `grid_snap`/`auto_b_trip`/`grid_snap_keep_triplets`/`triple_pref_thresh` are v2-only behaviors that are
**no-ops on the deployed v1 regime** (byte-identical verified). **Permanent section — keep in every rewrite.**
NOTE: the BYO density `×4/subdiv` fix + the offset detector are `scripts/generate.py` behaviors and do NOT touch the
exporter argparse defaults below (generate.py imports `CANONICAL_DECODE` for the shared palette only).
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
+ `V2_MSL=5400`. v2 decode knobs (all v1 no-ops): `--min_onset_gap` (None→auto 2 on v2); `--grid_snap auto`; the
triplet band via `--onset_phase_calib "0,1.0,b_trip"`; `--auto_b_trip`. Do NOT mix `highres_v2` with the v1 checkpoint.

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- Branch **`feat/governor-subdiv-recalib`** (off `feat/data-layer-v2`, off `main`). This session's commits (density
  fix + this docs refresh) land here. **Verify PR state via `gh pr view 70`** (do not trust this line) — PR #70
  (`feat/governor-subdiv-recalib` → `main`) was the open PR for this branch's BYO work. The deploy swap is a NEW
  change not yet made.
- Gitignored / not committed: `outputs/`, `transcripts/`, scratchpad probes (`$CLAUDE_JOB_DIR/tmp/offset_v*.py`,
  `probe_density_quiet.py`, `real_reference.py`). Test charts: `~/sm-generated/toulouse_bpm128`, `toulouse_anchor_beat`.

## READ-FIRST (in order)
Memory [[ship-mode-park-research]] (directive) → **`notes/byo_offset_detection_findings.md`** (THIS thread: the deaf-
choreography root cause + the offset detector) → [[personal-reference-charts]] + [[byo-audio-bpm-footgun]] (the offset
overturn) → lineage `experiment_lineage/byo-audio-alignment-arc.md` **Ch.2** → `generation-defaults §0` (the BYO
CHOREOGRAPHY/OFFSET + density-fix blurbs) → the ship checklist above. Load-bearing skills: **generation-defaults,
conditioning-mechanics §6, experiment-design** (the density→placement mis-attribution was a live Rule-1/Rule-8 catch).

## DISCIPLINE
**Match the fix to the FELT property** (density≠placement — the `×4/subdiv` fix was real but the wrong axis; the ears
caught it). **A playback-sync argument is NOT a choreography argument** (the "offset red herring" overturn). **Use the
oracle** (`~/sm-personal` + training-pack `#OFFSET`s) to validate an audio estimator BEFORE wiring — it killed 3
confident-but-wrong detector variants (DFT/kick/tiebreak) cheaply. **BPM must be user-supplied.** Ship mode: don't
wander into parked research. One change at a time; match the verb to the evidence ([[claim-precision]]).
