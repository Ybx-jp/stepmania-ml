---
name: generation-defaults
description: >
  The ONE canonical verified decode/export config for this project's chart generator — the exact model,
  features, and full decode-knob palette to pass so a generation, probe, eval, or playtest REPLICATES what the
  user actually plays. Use BEFORE writing or running ANY code that calls `LayeredTypedChartGenerator.generate()`
  or `export_typed_samples.py` (a probe, A/B, re-run of an old experiment, new playtest set), and whenever you
  are about to compare a generated chart to anything. Distilled from the recurring, expensive failure where a
  probe ran with a STALE SUBSET of settings (wrong checkpoint, 41-dim features, pattern_temp 0.7, governor off,
  bpm unset → governor silent, the 16th-unlock left off) and so measured a different model than the deployed one.
  Pairs with `conditioning-mechanics` (the math of each knob) and `experiment-design` (attribution discipline);
  this skill is the ground-truth VALUES.
---

# Canonical generation defaults

**The one job:** when you generate charts in a probe/eval/export, use the SAME palette of verified settings the
user actually plays. Re-running an experiment on a stale subset measures a *different, unplayable, or governor-off*
model and the result is worthless. This has happened repeatedly. `conditioning-mechanics` is the per-knob
mechanism; **this skill is the config VALUES.**

> **There is ONE canonical config: the `export_typed_samples.py` defaults** (the playtest-validated full stack).
> Everything below is that config. The thing that is NOT it: `generate()`'s bare function defaults — raw
> low-level values that are unplayable + governor-off (`hold_aware=False`, `max_jack_run=None`,
> `fatigue_penalty=None`, `bpm=None` → governor silent, `type_temperature=1.0`); never call `generate()` bare.
> **SINGLE SOURCE OF TRUTH (since 2026-06-30): `src/generation/decode_defaults.py`.** Both the public CLI
> (`scripts/generate.py`) and the exporter (`export_typed_samples.py`) import `CANONICAL_DECODE` for their argparse
> defaults, so they CANNOT drift apart — the palette values live in exactly one place (verified: both CLIs resolve
> byte-identical defaults). The module also exports `apply_phase_calib()` (the `tau`↔16th-unlock coupling, shared
> so it can't diverge either) and `calib_arg_default()`/`parse_phase_calib()`. **To change the canonical palette,
> edit the dict in that module — never a script's literal.** This replaced the old failure mode where
> `scripts/generate.py` drifted stale (`pattern_temperature=0.7`, stamina/breathe off, no 16th-unlock) while the
> exporter moved on. Current canonical values: `pattern_temperature=1.0`, `type_temperature=0.4`,
> `fatigue_penalty=2`/`fatigue_free=6`, `stamina_ceiling=50`/`breathe=1.2`, **`hold_stream_penalty=8`/`floor=0.45`
> (holds-in-streams fix) + `footswitch=False` (2026-07-02, playtest-validated)**, **`onset_phase_calib=(0.0,1.0)` the
> 16th-unlock** (offset also baked into `tau`).
>
> **PROBES: don't hand-roll the tau pipeline — import `src/generation/decode_harness.py`.** It exports
> `conditioned_p_onset(model, memory, diff, radar=, style=, guidance=, phase_calib=, extra_offset=)` (the deployed
> onset→p_onset path: conditioning + CFG blend + phase-calib, built exactly as the decode does), `compute_tau(p,
> density)`, and `phase_shares(onset_frames)` (the canonical quarter/8th/16th metric). generate.py + the exporter
> are DOGFOODED through these, so a probe that uses them matches deployment by construction (this is the §3/§6
> footgun — 30+ probes re-derived it, some with a buggy `calibrated_p_onset`). `DEPLOYED_CHECKPOINT` is here too.

## 0. THE MODEL (get this right first — it's the most-missed)
> **★ DEPLOY SWAP DONE (2026-07-09): v2 IS NOW THE DEFAULT in both CLIs.** `scripts/generate.py` and
> `export_typed_samples.py` DEFAULT to `--features highres_v2` + `checkpoints/gen_motif_v2_48th_cont/best_val.pt`
> (the 48th-grid model). The v2-only auto knobs (`grid_snap auto`, `auto_b_trip`) ship ON. The legacy v1 model
> (`gen_motif_full_fixed`, 16th grid) is still REACHABLE via `--features highres` — read the v1 paragraph below as
> "the legacy path," not "the default." Everything else in this skill (the decode palette §1, tau §2/§3, governors
> §8) is grid-agnostic and unchanged.
- **v1 / LEGACY model = `checkpoints/gen_motif_full_fixed/best_val.pt`** — the 42-dim H19 highres retrain (radar +
  continuous motif + discrete figure). Build it `LayeredTypedChartGenerator(audio_dim=42, d_model=128,
  num_layers=4, onset_layers=2)`. Reach it with `--features highres`.
- **Features `highres` (v1/legacy):** `AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True,
  use_highres_onset=True)` → `audio_dim=42`, `cache_dir='cache/samples_v3'`. (41-dim `samples_v2` = the OLD
  stage1 space; 23-dim `samples` = the base/critic space — neither matches the generator.)
- ⏩ **`highres_v2` — ✅ THE DEFAULT since 2026-07-09 (Phase 6 by-ear passed 2026-07-05):**
  same 42-dim channels on the 48th grid (`timesteps_per_beat=12`, `beat_sync=True`, `cache/samples_v3_48th`), paired
  with `StepManiaParser.for_v2()`. Best checkpoint `checkpoints/gen_motif_v2_48th_cont/best_val.pt` (val 0.7435). The
  by-ear gate on the triplet set CLEARED — the 48th grid REMOVES the triplet tax with zero degradation
  (`notes/playtest_log.md` 2026-07-05). **The exporter NOW SUPPORTS it (commit `837c1ed`):** `--features highres_v2`
  auto-selects the `for_v2()` parser + the 48th-grid `sm_writer` (48 rows/measure) + `cache_dir=None` + builds the
  model at `max_len=5504` + uses `V2_MSL=5400` (the v1 config msl=1440 clips a v2 song to ⅓ — a fixed truncation bug).
  **AND `scripts/generate.py` (the bring-your-own-SONG single-file CLI) also speaks v2 now (commit `cf0b820`):**
  `--features highres_v2` there auto-selects the v2 checkpoint (`gen_motif_v2_48th_cont`), the 48th feature extractor,
  the 5504-frame context, `V2_MSL`, the 48-row writer, and the v2 decode flags (`--no_fast_jump`/`--min_onset_gap`/
  `--grid_snap`/`--auto_b_trip`) — all derived from one `subdiv` (`fspec.extractor.config.timesteps_per_beat`); its
  DEFAULT stays highres/v1 (v1 byte-identical), so this is v2-reachable-via-CLI, not the default-swap. It also gained
  `--title` + `.sm` presentation flags + `--inherit_from SM|auto` (carry a source chart's banner/background/#BGCHANGES
  video, copying the media in) via the new `src/generation/sm_headers.py` + `sm_writer`'s `header=` dict.
  ⚠️ **BYO-audio alignment (2026-07-07/08, `notes/byo_audio_alignment_findings.md` / [[byo-audio-bpm-footgun]]):**
  `generate.py` grids the whole chart from BPM, and its auto-estimate (`librosa.beat.tempo`) is UNRELIABLE
  (octave/2:3 errors) → **always pass `--bpm`** for a BYO-song; it warns loudly when omitted (`3d5639d`). Variable-BPM
  / `#STOPS` songs are unsupported (single-BPM tool). Long songs are NO LONGER truncated: `generate.py` EXTENDS the
  sinusoidal positional encoding to cover the whole song (`75cffaf`; `--max_len` default None = whole song, `SAFETY_CAP`
  24000) — the model extrapolates past its trained context gracefully. (These are generate.py behaviors; the exporter,
  which parses a real chart for BPM/length, is unaffected.)
  ⚠️ **CHOREOGRAPHY / OFFSET (2026-07-08, `notes/byo_offset_detection_findings.md` / [[personal-reference-charts]]):**
  the by-ear "no choreography, like the model is deaf" on the personal set traced to **audio↔beat-grid misalignment**
  (the model chores on `metric_phase`, so a shifted grid → phantom-grid placement). **Two independent parts, and the
  old "offset is a red herring" note is OVERTURNED** (that was playback-only): (a) a wrong BPM even ~1% DRIFTS a beat
  every ~51s (Toulouse charted at estimate 129.199 vs true 128); (b) `build_stub_chart` hardcodes `offset=0.0` so
  frame 0 ≠ the first beat. Auto-offset detector (full-band onset pulse-train + 31ms latency cal) recovers the offset
  to ~7ms on ~80% of songs (validated vs `~/sm-personal` + training packs as an oracle); ~20% half-beat-slip → needs a
  confidence flag. NOT yet wired into generate.py; anchoring A/B (extract@t=0 vs @first-beat) is a pending BY-EAR gate.
  ⚠️ **generate.py DENSITY subdiv-fix (2026-07-08, `generate.py:272`):** the manifold `E[density|diff]` is a 16th-grid
  fraction; generate.py never applied the exporter's `×4/subdiv` correction (`export_typed_samples.py:631`), so every
  v2 BYO chart ran ~subdiv/4× (≈3× at the target, ~2× realized) too dense — now scaled, mirroring the exporter.
  Phase-5 decode re-index (`590daa1`): the phase grid is `subdiv`-parameterized (v1 `highres`=subdiv 4, byte-identical).
  Do NOT pair `highres_v2` with `gen_motif_full_fixed`, or a v1 checkpoint with the v2 cache (grid mismatch).
  **DECODE-PLAYABILITY pass DONE (2026-07-05 session 2)** — all v1-no-op / subdiv=4 byte-identical: the governor
  **subdiv-recalibration** (`frame_hz=BPM·subdiv/60` + `f16` gap/window scaling, `33de530`); the **footspeed floor**
  `--min_onset_gap` (None→auto 2 on v2: forbids 1-frame 48th flams, keeps 2-frame triplet-16ths; `63125eb`); the
  **triplet band** via `--onset_phase_calib "0,1.0,b_trip"` (3rd element, default 0; `b_trip=0.7` BY-EAR WON, `46a25b4`);
  the **`--style` density fix** (`style_density*=4/subdiv`, `63125eb`). BY-EAR WIN on pt_chaos_v2 ("brand new note
  colors, flowy streams, conditioning effective"). The **no-fast-jump cap** (`--no_fast_jump`, default ON; sub-16th
  two-foot-jump veto — `conditioning-mechanics §8d`) is now BUILT + BY-EAR PASSED. The `freeze=high` v2 hold-stream
  gate bug is only PARTIALLY fixed (the `dens` subdiv fix `conditioning-mechanics §7` HALVED it; the real defect —
  5–6 beat holds with a one-foot stream under them — PERSISTS; real fix DESIGNED + PARKED, `footspeed_floor_findings.md
  §5b`). ⚠️ **PROJECT IS IN SHIP MODE (2026-07-06, [[ship-mode-park-research]] memory): cut v1.0.0.** The swap =
  make `gen_motif_v2_48th_cont` + `--features highres_v2` the default, `b_trip=0.7` (song-dependent; 0.7 is the gentler
  default, 1.0 adds busyness on duple songs). **DONE 2026-07-09** — the swap landed (b_trip=0.7 auto default); the
  hold-stream `freeze=high` edge ships as a documented KNOWN LIMITATION (user call). Groove-radar chaos fix is
  RETRAIN-GATED (`notes/manifold_radar_subdiv_findings.md` — do NOT refit the manifold for the current model). See
  `notes/footspeed_floor_findings.md`, `notes/data_layer_v2_scope.md`, `conditioning-mechanics §6/§8`.
- ⚠️ **TRAP:** `export_typed_samples.py`'s argparse `--checkpoint` DEFAULT is the legacy `gen_style` (23-dim) and
  `--features` DEFAULTS now correctly resolve to the v2 canonical model (`gen_motif_v2_48th_cont` + `highres_v2`) as
  of 2026-07-09 — the *bare* run loads the RIGHT model (the old "bare default = wrong model" trap is FIXED). The
  remaining footgun is MIXING: never pair `--features highres_v2` with a v1 checkpoint (or vice-versa) — the grid
  mismatches. To run the legacy v1 model, pass BOTH `--checkpoint checkpoints/gen_motif_full_fixed/best_val.pt
  --features highres`.

## 1. THE canonical decode palette (copy-paste)
Values = `export_typed_samples.py` argparse defaults (06-28). `bpm` is **per-song** and MANDATORY (no bpm →
governor silent). Build `tau` the conditioned way (§3). `enforce_playability` FORCES `hold_aware /
no_jump_during_hold / no_cross_during_hold` on regardless.
```python
DECODE = dict(
    type_sample=True,    type_temperature=0.4,
    pattern_sample=True, pattern_temperature=1.0,     # real panel balance & jack rate (NOT 0.7 — that's stale)
    repetition_penalty=1.0,
    hold_aware=True, no_jump_during_hold=True, no_cross_during_hold=True, max_jack_run=2,  # MANDATORY playability
    jack_penalty=None,                                 # DEPRECATED (superseded by fatigue); off
    fatigue_penalty=2.0, fatigue_free=6.0,             # per-NOTE foot governor (§8b). exporter free=6; generate() default 12 — both in vouched 6–12
    stamina_ceiling=50.0, stamina_tau=8.0, stamina_scale=15.0, stamina_breathe=1.2,  # per-REGION stamina+arc (§8c)
    onset_phase_calib=(0.0, 1.0),                      # ★ THE 16th-UNLOCK — un-buries 16th-offbeats so they fire where audio affords; b8=0, b16≈1.0
    hold_stream_penalty=8.0, hold_stream_floor=0.45, hold_stream_win=16,  # HOLD-IN-STREAM fix (2026-07-02): suppress holds in dense streams; floor protects sparse holds
    footswitch=False,                                  # (2026-07-02) forbid footswitch footing -> model ALTERNATES same-panel runs (playtest "sooooo much better")
    hold_release_run=4, hold_release_gap=None, hold_max_beats=6.0,  # ★ FREE-FOOT-UNDER-HOLD force-close (defect #3, 2026-07-12, by-ear PASSED): 8th = fastest allowable note under a hold; a faster note concludes the hold ON that note (onset LOOKAHEAD -> freed foot travels); 6-beat duration cap. Byte-identical where no defect. §5c
)
# generate() supplies the rest of the governor internals — DO NOT hand-set unless retuning:
#   stamina_breathe_floor=0.4 (outro-collapse fix), stamina_max_bump=0.45, stamina_breathe_win=96,
#   fatigue_tau=2.0, fatigue_cap=30, jack_weight=1.0, travel_weight=0.6, footswitch_pen=4.0
# per-song:
g = model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                   onset_threshold=tau, bpm=float(meta['chart'].bpm), **DECODE)[0]
```
- **No groove conditioning by default:** `radar=None, style=None, motif=None, figure=None, guidance_scale=1.0`.
  That's clean audio+difficulty — the baseline most playtests use. Add a groove knob ONLY deliberately, via the
  manifold (`--style`, never `--radar` mean-pin), per `conditioning-mechanics` §2–§6.
- **Three v2-only defaults (on by default, v1 no-ops):** `--grid_snap auto` (16th-grid snap for difficulty ≤ Medium —
  vetoes pure-48th jitter) + `--auto_b_trip` (triplet band on audio-detected triplet songs; both resolve into the
  `onset_logit_offset` / `onset_phase_calib` slots) + **`--onset_window 3600` (2026-07-12, by-ear PASSED, `notes/
  universal_window_findings.md`)** = the UNIVERSAL sub-train-length window (`decode_defaults.UNIVERSAL_ONSET_WINDOW`):
  tile the onset head so a song's END leaves the under-trained abs-PE tail (v2 train len median 3120/max 5128 →
  single-pass fires only ~30% of real tail notes past ~3800). Single-sourced into tau (`conditioned_p_onset window=`)
  + decode (`generate onset_window=`) with `--onset_tail_hangover auto` (=W//2, end-centering). A song < 3600 frames
  or v1 = byte-identical no-op; disable via `--onset_window 0`. generate.py: `--onset_window auto` (v2→3600).

## 1a. ★ `onset_phase_calib` — the 16th-unlock that ties the rhythm together (don't omit it)
The governor/breathe shape DENSITY; `onset_phase_calib` fixes WHERE the notes land on the 16th grid. Without it the
rhythm reads blocky and the head places ~zero 16ths — a READOUT problem, not representation: the head's `p_onset`
ranks real 16th slots at AUC 0.73, but a single global `tau` (16th base-rate ~4%) sits above almost every 16th
frame and buries them (`notes/onset_alloc_findings.md`). The calib adds a per-phase logit offset that un-buries
them so the **16th count floats with the audio per song** (chaotic songs get many, calm ~none — not a flat quota).
- **Value:** `onset_phase_calib=(b8, b16)` in logit space, applied to `t%4==2` (8th) and `t%4∈{1,3}` (16th-offbeat).
  Canonical `b8=0.0, b16≈1.0` (the `unlock16_b10` set). It's **"a KNEE, not a node"** — the right b16 is
  song-dependent (~0.5 calm → 1.0 sweet spot → 2.0 = near-all-16ths "fun & quirky" on dense songs like japa1).
  Exposed as the "16th unlock" slider in `tools/chart_ui.py`. (The exporter help's `"0,0.19"` example is STALE.)
- **Playtest 2026-06-28:** at the unlock HSL read "soo chaotic and expressive"; the user's read is that the
  16th-unlock (a pure decode lever, no retrain) **substantially addresses the melodic-solo "no 16ths" complaint.**
- ⚠️ **`tau` MUST use the SAME offset** (apply the b8/b16 offset to the onset logits BEFORE the density quantile),
  exactly as the exporter does — else the calib floods past a tau computed without it. (§3; `conditioning-mechanics`
  §6.) NOTE: this is the LIVE 16th lever; the flat `onset_phase_alloc` quota is DEPRECATED (it SMEARS), and
  `onset_logit_scale` is a no-op under quantile thresholding.

## 2. tau (onset threshold / density) — never skip the conditioned recompute
```
tau = quantile( sigmoid( GUIDED onset logits + the onset_phase_calib offset ), 1 − gen_density )
gen_density priority:  --target_density  >  manifold E[density|style,diff]  >  real source-chart density
```
- `tau` MUST come from the **same** conditioned + guided + phase-calib-offset onset logits `generate()` decodes
  from. A `tau` calibrated on plain/unconditioned `p` lets conditioning or the calib flood past it → wrong
  density (cond-mech §3, §6).
- With the governor on, the **per-frame** onset decision is made IN the AR loop (stamina gate raises the effective
  threshold), so realized density ≤ the `tau` target where sustained workload is high. A probe that rebuilds onset
  as `p>tau` and skips the per-frame gate silently diverges whenever stamina is on (cond-mech §8c) — pass the full
  governor + bpm to `generate()` and let it handle the gate; don't reimplement it.

## 3. The recurring failures this skill exists to stop (each invalidated a result)
- **Wrong checkpoint** — ran on `gen_stage1`/`gen_style` instead of `gen_motif_full_fixed` (the exporter's legacy
  default checkpoint actively misleads here).
- **Wrong feature space** — built 41-dim (`samples_v2`, no `use_highres_onset`) or 23-dim against the 42-dim model.
- **Stale `pattern_temperature` 0.7** copied from old scripts instead of the validated 1.0. (`scripts/generate.py`
  was the historical source of this stale value; realigned to 1.0 on 2026-06-30 — but check any OTHER script you copy from.)
- **16th-unlock left off** — omitted `onset_phase_calib` (or set it but forgot to apply the same offset to `tau`),
  so the chart reads blocky and you wrongly conclude "the model can't do 16ths."
- **Governor off / silent** — omitted `fatigue_penalty/stamina_*`, OR set them but forgot `bpm=` (governor is
  silent without BPM), OR used `generate()` bare defaults (everything off + unplayable).
- **Playability off** — forgot `hold_aware/no_jump/no_cross_during_hold/max_jack_run`, producing a chart the user
  could never play, then "comparing" it to a played one.
- **`tau` from unconditioned logits**, or rebuilding onset outside the AR loop while stamina is on.

## 4. Deviating from canonical
Fine to deviate — but make it ONE labeled change from this baseline (experiment-design: one variable), say so in
the output/findings, and keep everything else canonical. Examples: a governor-OFF ablation to isolate the
governor's contribution to a metric; a `pattern_temperature` sweep; an `onset_phase_calib` b16 sweep for a 16th
probe. Never deviate by accident (a stale default is not a deviation, it's a bug).

**Perf-only, decode-neutral:** `export_typed_samples.py --prefetch_workers N` (default `min(4, cpu_count)`; `0` =
old synchronous path) overlaps CPU audio-feature extraction with GPU decode via worker processes — a big win on a
COLD feature cache, ~neutral warm. It touches NOTHING in the decode palette: output is byte-identical to the
synchronous path (verified warm + cold, `w0`≡`w4`), because extraction is deterministic and consumes no global RNG.
So it's NOT a "deviation" in the experiment-design sense — never treat a `--prefetch_workers` change as a variable.

**Selection / experimental flags (2026-07-04):** `--hardest` (per-song HARDEST available difficulty — the non-groove
`--song_filter` path otherwise silently picks the FIRST/Beginner chart, too sparse to reveal subtleties; use it for
any playtest set drawn by filter; PR #61). `--chaos_onset_gate GAIN` is **EXPERIMENTAL and known-DEAD** — an
audio-keyed off-beat gate (`decode_harness.chaos_onset_gate_offset`) that Phase-0 showed can't separate expressive
16ths from smear ones (`notes/chaos_onset_gate_scope.md`); default 0 = off; do NOT use it in a canonical run.

**v2 gate reach (2026-07-06):** the v2 export path (`--features highres_v2`) now uses the **widened INFERENCE_GATES
by DEFAULT** (`src/data/stepmania_parser.py`: bpm [40,320], length [30,600]s, `max_simultaneous=4` = admits
hand/quad charts, gimmick guard on) — vs the narrow TRAINING gates [60,200]/[75,130]/simul2. Fixes a bug where
`--relax_gates` was a silent no-op on v2 (the `if subdiv==12` branch was unconditional); +55% val reach (532→822).
`--strict_gates` reverts to the narrow gates (to reproduce a pre-fix v2 run). Training's `for_v2()` default stays
narrow (the v2 training cache is untouched). v1 export is unchanged (still `--relax_gates`-gated → `for_inference()`).

**★ TWO v2-only AUTO DEFAULTS (2026-07-06, now ON in the canonical config; both v1 no-ops; `check_export_defaults.py`
now 25 ✓). BY-EAR-PENDING — offline-validated, wired to default per user ship-mode directive; a later session should
gate them by ear before final cut.**

**`--grid_snap auto` — the 16th-GRID SNAP (v2/48th-grid only; `notes/grid_snap_findings.md`):** vetoes onsets on the
pure-48th cells `{1,5,7,11}@subdiv=12` via a −30 logit offset single-sourced (`decode_defaults.grid_snap_offset`)
into BOTH tau (`conditioned_p_onset extra_offset=`) and decode (`generate onset_logit_offset=`), stacked into the
exporter's `harm_off_t` slot. Fixes the verified v2 defect where BUSY low/mid-difficulty songs place 8–23% of notes
on 24th/48th cells the human originals NEVER use (real 48th-usage ~0% at Beginner/Easy/Medium). Preserves the density
budget (same note count, snapped on-grid → note counts move TOWARD the original). **`auto` = ON for difficulty ≤
Medium (diff_idx≤2), OFF at Hard** (fast 48th runs are legit there + the v2 sub-16th/triplet win lives at Hard);
`all` = every difficulty; `off` = never. **`--grid_snap_keep_triplets` (DEFAULT ON)** keeps the triplet family
`{2,4,8,10}` and vetoes ONLY `{1,5,7,11}`, so the snap kills 48th jitter but PRESERVES triplets (real even at Medium)
and COMPOSES with `--auto_b_trip` (b_trip boosts triplet frames; the snap leaves them open). `--no-grid_snap_keep_
triplets` = full snap (16th grid only). v1 (subdiv=4) = hard no-op (`grid_snap_offset` all-zeros; kept phases = all 4).
Provably inert on already-on-grid songs (byte-identical). **BY-EAR PENDING** — install-set A/B `v2_low_*` vs `v2_low_*_snap`.

**`--auto_b_trip` (now DEFAULT ON) — the duple/triplet SWITCH (v2/48th-grid only; `notes/v2_safety_envelope_findings.md`):**
applies the triplet phase band (`b_trip` = the 3rd `--onset_phase_calib` element, default 0.7) ONLY to triplet-feel
songs, per an AUDIO meter detector (`src/data/meter_detect.py` `detect_triple_pref`; `triple_pref > --triple_pref_
thresh`, default 0). Duple songs → `b_trip=0` (no band, no busyness). Feeds the SAME per-song calib to BOTH the tau
path and `generate()`. On v1 (subdiv=4) the detector is SKIPPED (empty band → no-op, no per-song audio-load cost).
**SAFE but NOT a clean win:** the ρ+0.47 detector fired on only 3/6 chart-triplet songs in the sweep (missed Sway
tf0.61) → `auto` under-helps missed triplets while keeping duple perfectly clean; `global b_trip=0.7` helps all
triplets but adds slight duple busyness. auto-vs-global is a BY-EAR call, not offline-decidable. `--no-auto_b_trip`
reverts to a fixed global b_trip (the 3rd `--onset_phase_calib` element).

## PRE-FLIGHT CHECKLIST (run before any generate/probe/export)
1. **Model:** `gen_motif_full_fixed`, `audio_dim=42`? (NOT gen_style/gen_stage1.)
2. **Features:** `highres` config (4 flags on) + `cache/samples_v3`? (NOT samples_v2/samples.)
3. **Decode palette = the exporter §1 values** (type 0.4 / pattern 1.0 / fatigue 2 free 6 / stamina 50 tau 8
   breathe 1.2 / **onset_phase_calib (0,~1.0)** / max_jack_run 2), playability on?
4. **Governor live:** `fatigue_penalty=2` + stamina/breathe set AND `bpm=` per song? (No bpm = silent.)
5. **tau:** conditioned + guided + the SAME onset_phase_calib offset; density from the right source?
6. **Conditioning:** none by default; any groove knob added deliberately via the manifold (`--style`, not `--radar`)?
7. **Deviations:** exactly one, intentional, and labeled?

Cross-refs: `conditioning-mechanics` (the per-knob math + the alignment checklist), `experiment-design` (attribution
order, don't blame the model for a harness/config bug), `notes/governor_release_region.md` (the vouched governor
ranges), `notes/onset_alloc_findings.md` (why the 16th-unlock is a readout fix), `playtest` skill (set generation
+ logging).
