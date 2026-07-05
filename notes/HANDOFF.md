# HANDOFF — ACTIVE: the 4/4-GRID METER thread. Triplet tax BY-EAR CONFIRMED → data-layer-v2 refactor JUSTIFIED, greenlight pending.

**Written 2026-07-04 for the next Claude.** This session (1) DOWNGRADED the tolerance formula (the expanded k4 run +
a permutation-null 2nd-factor hunt both cut against the R²0.44 headline), then (2) spun off + ran the **4/4-grid
meter thread** end-to-end: the whole pipeline is hard-4/4 duple-16th, triplet songs are mis-gridded (~33 ms
displacement), and this is now **BY-EAR CONFIRMED felt + severe**. All diagnostic — **NO model or decode-default
change.** The pending decision is whether to greenlight the data-layer-v2 refactor.

## WHERE WE ARE
Deployed model UNCHANGED = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres). Canonical decode defaults
UNCHANGED (block below; validator passes). This session added probes + notes + a new memory/lineage; it did NOT
touch the generator, the decode palette, or the parser.

## THE ACTIVE THREAD — the 4/4-grid meter tax (lineage `experiment_lineage/meter-grid-arc.md`, memory [[meter-4-4-grid]])
The whole pipeline is hard-wired to 4/4 **duple-16th subdivision** (parser `timesteps_per_beat=4`, no
`#TIMESIGNATURES`, `ts=floor(beat·4)` floors triplets at parse time; features/model `t%4`/`t%16`; the `t%4` grid is
baked into metric_phase, the 16th-unlock, `phase_shares`, SB, and the tolerance formula). Full scope +
all findings: `notes/meter_4_4_assumption_scope.md`.
- **Census (5345 songs):** ~70% pure-4/4; **7.0% structural triplet tax (≥0.15), 3.3% dominant (≥0.30); explicit
  non-4/4 `#TIMESIGNATURES` = 0.1%** ⇒ a **SUBDIVISION tax, NOT a time-signature tax.**
- **Damage (n=597):** triplet content vs floor-to-16th timing DISPLACEMENT Spearman **+0.83**, up to ~0.083 beats
  ≈ **33 ms @150BPM = 2-3 judgment windows**. Timing DISTORTION (limping), not note loss (collision ≤2%).
- **The critic is STRUCTURALLY BLIND** (triplets floored at parse time → training/critic/generated all de-tripleted →
  the tax is SUB-GRID). Method keeper: **measure a sub-grid defect at the QUANTIZER (displacement), not the SCORER.**
- **✅ BY-EAR CONFIRMED (the binding gate, `playtest_log.md`):** plain-canonical First of the Year (94% triplet
  measures) + My Christmas list (80%, 39% near-pure). "A little off" vs "**badly timing everything**." Severity tracks
  triplet CONCENTRATION (a global sync bug would NOT) ⇒ IS the meter tax. **The 33 ms measurement PREDICTED the ear**
  (a hard representation fact, not a taste proxy — contrast the tolerance thread's ear-overturned metrics).
- **Meter-equivariant SB prototype (`probe_meter_equivariant_sb.py`):** a rotation-invariant DFT of a 12-slot
  beat-phase histogram (triplet energy at 3&6 cycles/beat) recovers subdivision vs chart triplet_frac ρ+0.47. Two
  harness bugs + one units bug caught first (Rule 7/11). NOT yet shown to beat SB on tolerance (needs a triplet-rich set).
- **16th-ceiling → backbone cliff (user hypothesis): partial, NOT established.** Saturation REFUTED (flip at held
  density ~0.40). Missing sub-16th intensity vocabulary REAL but MODEST (real charts: density vs sub-16th +0.65,
  densest decile ~10%, mostly 32nd bursts; causal test underpowered null). ⇒ finer grid COMPLEMENTS the chaos×onset
  gate; it is NOT the cliff fix (the cliff is the anchoring/H4 problem).

## THE OPEN FORK / BINDING DECISION (user's call)
1. **GREENLIGHT the data-layer-v2 refactor** (the by-ear tax justifies it). `constraint_relaxation_roadmap.md` already
   bundles **fixed-BPM + triplets** as ONE beat-synchronous re-grid. Nothing crashes, but it RE-INDEXES the whole
   `t%4` phase vocabulary (metric_phase / 16th-unlock / SB / tolerance) + ~3× sequence length retrain + per-frame
   `frame_hz` governors = a coordinated `conditioning-mechanics §6` + `generation-defaults` version bump. Scope it as
   its own arc. The `bpm_map` in `probe_meter_equivariant_sb.py` is the proven time↔beat core.
2. **CHEAP DECOUPLED WIN (available now, zero grid risk):** the BPM/length gates are DATASET-only —
   `StepManiaParser._validate_phase1_requirements` (BPM avg `[60,200]`, length `[75s,130s]`); **`generate()` is
   filter-free** (consumes precomputed audio + scalar bpm, validates neither). So relax them on the INFERENCE/export
   path: (a) length filter = trivially relaxable (max_len truncation guards it); (b) BPM range = widen ~`[40,320]` +
   a GIMMICK guard (2467/1431/441 = notation tricks that would feed the hop garbage — don't just delete); this is
   PURE reach to songs `generate()` can already chart. Independent of the grid refactor.

## AWAITING USER
- **The by-ear meter gate is DONE + CONFIRMED** (logged in `playtest_log.md`, entry "✅ CONFIRMED: the 4/4-GRID
  TRIPLET TAX"). Set still installed at `~/sm-generated/meter_triplet_test/` (First of the Year + My Christmas list).
- No new set pending. Next action is the user's DECISION on the fork above.

## PARKED / CLOSED (don't re-derive)
- **Tolerance formula — DOWNGRADED + effectively closed** (lineage `good-settings-region-arc.md`, memory
  [[good-settings-region]]). The expanded k4 flip run (`flip_point_v2.csv`, n=32) cut `SB→g₀` from R²0.44 (n=14 =
  small-n optimism) to **ρ+0.39 censored / R²~0.09 / LOO-CV≈0**; the permutation-null 2nd-factor hunt
  (`probe_flip_secondfactor.py`, 84-dim fingerprint) = **CLEAN NEGATIVE** (best ΔCV+0.267 < null 95th +0.387, p0.23;
  neg-control density fired). SB is a real-but-WEAK RANK predictor; the high-SB fork is NOT audio-poolable →
  note-context. **The 3/3 ear result SURVIVES** (tested songs on the clean spine). Open if revisited: operationalize
  SB as an honest rank heuristic, or chase the fork via note-context. Details `tolerance_formula_findings.md`.
- **chaos×onset GATE (retrain ceiling-raiser)** — SCOPED + PARKED (`chaos_onset_gate_scope.md`). Phase-0 decode
  EXHAUSTED (off-beat placement not audio-reachable); reshaped to the seq-onset head + chaos, Stage-1 de-risk GREEN
  (frozen-`h` 0.862 ≫ audio 0.618). User parked the train. It (not the meter grid) is the cliff fix.

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift. Durable mirror
of `generation-defaults` §1; update both (and re-run the validator) on any deliberate change. **This section is
permanent — keep it in every HANDOFF rewrite.**

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

For a triplet by-ear set: `--song_filter` triplet titles + `--hardest`, otherwise the canonical block (plain
generation — no radar/style — isolates the GRID, not conditioning). For the tolerance/flip probes the milestone
HIGH-chaos spec is `--style "chaos=0.9,voltage=0.7,air=0.5,freeze=0.5"` swept over `--guidance`.

## BRANCH / PR STATE  (verify ALL live state via `gh pr view <n>` / `gh pr list` / `git log origin/main`)
- **This session's work is on branch `feat/tolerance-formula`.** Commit `20dabea` (the original formula) + `32b1065`
  (prior refresh) + `20dabea`-successor commits (the downgrade + meter thread + this refresh). **Verify pushed
  state / any PR via `gh pr list` / `git log origin/main..HEAD`.** The refresh opens/updates a PR to `main`.
- New this session: `probe_flip_secondfactor.py`, `probe_meter_equivariant_sb.py`,
  `notes/meter_4_4_assumption_scope.md`, lineage `meter-grid-arc.md`, memory `meter-4-4-grid.md`.
- Gitignored (reproducible): `cache/flip_point_v2.csv`; playtest set `~/sm-generated/meter_triplet_test/`.
- **Prior:** v0.2.0 (PR #58) + `--hardest` (PR #61) MERGED. `main` protected by `protect-main`.

## READ-FIRST (in order)
`notes/meter_4_4_assumption_scope.md` (the whole meter thread: code verification → census → damage → critic-blind →
by-ear → prototype → 16th-cliff → refactor scope) → lineage `experiment_lineage/meter-grid-arc.md` →
`notes/playtest_log.md` (the ✅ by-ear verdict) → `notes/constraint_relaxation_roadmap.md` (the data-layer-v2
bundling + decision rule) → (if revisiting tolerance) `notes/tolerance_formula_findings.md` (the downgrade section).
Load-bearing skills: **conditioning-mechanics §6** (the `t%4` phase grid + its NEW 4/4-limitation flag),
**generation-defaults §1** (canonical config), **experiment-design** (this meter thread is a WIN case: a
mechanism-grounded metric predicted the ear; harness/units bugs caught first), **playtest** (the by-ear gate).

## DISCIPLINE
Rule 0 (check notes first — the roadmap already scoped this refactor). **By-ear is the binding gate** (it confirmed
the meter tax AND overturned my "bursty vs pervasive" split). **Measure a sub-grid defect at the QUANTIZER not the
SCORER** (the critic is structurally blind to triplets). **Iterate on the HARNESS, not the concept** (3 harness/units
bugs caught before any wrong conclusion). Judge added features by LOO-CV + a permutation null (the tolerance R²0.44
was small-n optimism). Match the verb to the evidence ([[claim-precision]]). `playtest_log.md` = subjective only.
