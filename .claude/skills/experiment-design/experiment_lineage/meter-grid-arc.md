# Lineage — The 4/4 grid assumption: meter tax + data-layer-v2 refactor (2026-07-04 →)

**One line:** the whole pipeline is hard-wired to 4/4 duple-16th subdivision; triplet/compound songs are
mis-gridded (notes floored onto 16ths, ~33 ms off), CONFIRMED felt + severe by ear → the finer-grid + variable-BPM
"data-layer-v2" refactor is justified (greenlight pending). A model METHODOLOGY WIN thread: a mechanism-grounded
offline measurement (not a taste proxy) PREDICTED the ear, and two of my own harness bugs + a units bug were caught
before any wrong conclusion committed.

**Status:** ✅ BUILD ARC COMPLETE — **Phase 6 by-ear PASSED (2026-07-05, `_cont` val 0.7435):** the 48th grid REMOVES
the triplet tax, ZERO degradation, user "resounding 100% success… finally able to REALLY express tasty percussion".
v2 = DEPLOY CANDIDATE. Export tooling built (`837c1ed`: `--features highres_v2` + 48th `sm_writer` + the msl-truncation
fix). **NEXT (open): governor subdiv-recalibration** (frame_hz is BPM·4/t%4-coupled → playability ~3× off on the 48th
grid) THEN the deploy swap. NOT yet deployed. Spun off [[good-settings-region]] by verifying SB's 4/4 frame. Primary note
`notes/meter_4_4_assumption_scope.md`; build status `notes/data_layer_v2_scope.md`; memory [[meter-4-4-grid]].

## The hypothesis chain (believed → learned)
1. **Opened from the tolerance downgrade.** User: SB (the tolerance predictor) may be a valid frame in 4/4 but
   MISAPPLIED in other meters — some songs want naked 16ths / uniform triplets. "Verify strong-beat:4/4 first."
2. **`strong-beat : 4/4` VERIFIED (code):** SB's `strong=t%4∈{0,2}` is the simple-meter mask; the grid `t` IS
   metrical (hop `sr·60/(BPM·4)`); the assumption runs through the WHOLE stack — parser `timesteps_per_beat=4` +
   `beats_per_line=4.0/len` + NO `#TIMESIGNATURES` handling; `_metric_phase` = `t%4`/`t%16`; parse quantization
   `ts=floor(beat·4)` floors triplets. Triplets DEFLATE SB (chart-triplet vs SB ρ−0.36, mechanism-correct).
3. **BUT it does NOT explain the tolerance fork** (the reason the formula downgraded): only 1/32 flip songs is
   triplet-heavy (flip set is triplet-poor), and triplet-frac vs g₀-residual = −0.02 (zero). So the meter fix is a
   CORRECTNESS issue for a minority, NOT the fork's cause. *(Rule 9: bounded the claim to what the evidence supports;
   didn't oversell the user's good instinct into "this is the fork.")*
4. **CENSUS (5345 songs, chart-derived):** ~70% pure-4/4; **7.0% structural triplet (≥0.15), 3.3% dominant (≥0.30);
   explicit non-4/4 `#TIMESIGNATURES` = 0.1%.** ⇒ THE REFRAME: it's a **duple-16th SUBDIVISION tax, not a
   time-signature tax** (the `beats_per_measure=4` half is ~vacuous here; the `timesteps_per_beat=4` half bites).
5. **Is meter a QUALITY axis? (B-step 2) — YES at representation, INVISIBLE to the critic.** Structural finding:
   `ts=floor(beat·4)` de-triplets at PARSE time → training targets + critic inputs + generated output ALL on 16ths
   → the tax is SUB-GRID. The cheap graded-critic cut found nothing (deficit vs triplet ρ+0.13 p0.38) but was
   ALSO underpowered (1/48 triplet songs, `p_gen_sd`=0, saturated critic) — do NOT read the null as "no tax."
   **Method keeper: measure a sub-grid defect at the QUANTIZER, not the SCORER.** Damage measured directly (n=597):
   triplet vs floor-to-16th DISPLACEMENT ρ**+0.83**, up to ~33 ms; timing DISTORTION not note loss (collision ≤2%).
6. **✅ BY-EAR CONFIRMED (B-step 3, the binding gate).** Plain-canonical First of the Year (94% triplet-measures) +
   My Christmas list (80%, 39% near-pure). "A little off" vs "**badly timing everything**." Severity tracks triplet
   CONCENTRATION (a global sync bug would NOT) ⇒ IS the meter tax. Both notated 4/4 = pure subdivision tax. **The
   33 ms measurement PREDICTED the ear** — because it's a hard representation fact, not a taste proxy. Self-overturned
   my "bursty vs pervasive" split (BOTH pervasive; `measure_triplet_profile` Rule-8 check).
7. **Meter-equivariant SB prototype (parallel build) — geometry RESOLVES.** Rotation-invariant DFT of a 12-slot
   beat-phase histogram of the FINE onset envelope; triplet energy at 3&6 cycles/beat. vs chart triplet_frac ρ+0.47.
   NOT yet shown to predict tolerance better (needs a triplet-rich set). The detector doubles as the census's meter
   classifier.
8. **User Q: does the 16th ceiling CAUSE the backbone cliff? — partial, NOT established.** Saturation version
   REFUTED (flip at HELD density ~0.40, grid not full). Missing-sub-16th-vocabulary version REAL but MODEST (real
   charts: density vs sub-16th +0.65, densest decile ~10%, mostly 32nd bursts). Causal test underpowered null (1/28
   burst-heavy flip songs). Dominant real intensity = anchored 16ths (on-grid, model smears instead = H4/the gate).
   ⇒ finer grid COMPLEMENTS the chaos×onset gate, is NOT the cliff fix.
9. **Refactor scoped (data-layer-v2).** Rule-0 hit: `constraint_relaxation_roadmap.md` ALREADY bundles fixed-BPM +
   triplets. Gates are DATASET-only (`generate()` filter-free). Three animals: length filter (trivial), BPM range
   (widen + gimmick guard), fixed-BPM-tempo = SAME beat-sync re-grid as the finer grid. Nothing crashes; re-indexes
   the whole `t%4` phase vocabulary (metric_phase / 16th-unlock / SB / tolerance) + ~3× seq retrain. Roadmap gate
   ("defer until musicality plateaus") plausibly met. **Open fork: greenlight the refactor (deliberate arc) + the
   cheap decoupled filter-relaxation now.**

## Methodology wins/losses to learn from
- **WIN — a mechanism-grounded metric that PREDICTED the ear** (contrast the good-settings arc's 5 ear-overturned
  metrics). The difference: this defect is a hard representation fact (notes on the wrong grid); the tolerance metric
  was a taste proxy. When the metric measures the same physical thing the ear does, they agree.
- **WIN — 3 harness/units bugs caught BEFORE a wrong conclusion** (Rule 7/11): meter-detector phase-DRIFT (first-BPM
  only) → +0.01; rotation-SENSITIVE fixed cells → inverted −0.58; and the `d∉{1,2,4}` sub-16th UNITS bug that would
  have reported a spurious +0.96. Each fixed, then the real result. Iterate on the HARNESS, don't blame the concept.
- **WIN — measure a sub-grid defect at the QUANTIZER not the SCORER** (the critic is structurally blind; the null was
  a blind instrument + underpower, NOT evidence of no tax).
- **DISCIPLINE — bounded the user's good instincts to the evidence** (Rule 9): the meter fix is NOT the tolerance
  fork's cause (n=1 triplet in the flip set); the 16th ceiling is NOT established as the cliff cause (underpowered).
  Affirmed the ideas' merit without overselling.

## Cross-arc corroboration
- **SPUN OFF [[good-settings-region]]** (`good-settings-region-arc.md`): verifying SB's 4/4 frame opened this thread.
  That arc's tolerance formula DOWNGRADED the same session (expanded k4: R²0.44→0.09, 2nd-factor null) — the meter
  question was one candidate for the unexplained high-SB fork (ruled out as the CAUSE, but a real correctness issue).
- **CONNECTS-TO the chaos×onset gate / seq-onset arc:** the 16th-ceiling→cliff hypothesis lands on the same
  anchoring failure the gate targets; the finer grid is complementary (new vocabulary) not a substitute (anchored
  placement). The cliff stays the gate's problem.
- **CROSS-TESTS seq-onset-arc.md (48th-grid audio-placement lift, 2026-07-05, `notes/seqonset_v2grid_findings.md`,
  `probe_seqcontext_frozenh_v2.py`):** re-ran that arc's M1a bracket on THIS build's 48th grid to ask if the ~0.65
  audio-placement cap was partly a GRID artifact. Answer: a MODEST duple-16th lift (audio reach 32%→41% of gap;
  suggestive, needs the constant-BPM control) BUT the wall EXTENDS to triplets HARD — audio ≈ CHANCE (0.505) for
  triplet placement vs a note-context ceiling 0.930; `frozen_h_conv` ≡ ceiling (0.939) → the v2 decoder ENCODES
  triplet placement. So this build fixes the TARGET (triplets representable + placeable by the trained prior)
  without making them AUDIO-derivable — the seq-onset wall is corroborated, not overturned, on the finer grid.
  Reciprocal link in `seq-onset-arc.md`.
- **DEPENDS-ON [[quality-feature-attribution]]:** B-step 2 (meter-as-quality-axis) reuses its graded critic + ICC
  discipline; BPM is the precedent for a hidden per-song quality driver (and the BPM RANGE filter is part of this
  refactor's scope).

## The BUILD arc (data-layer-v2) — 2026-07-05, branch `feat/data-layer-v2`
The diagnosis GREENLIT the refactor; this session BUILT it through Phase 4 (deployed model UNCHANGED — no v2
checkpoint exists yet). Chain believed → learned:
- **Grid design (A1 fixed 48th vs meter-adaptive):** believed a fixed 48th grid wastes context on the 70% pure-duple
  songs → BUILT two hardening checks. **Emptiness** (`probe_v2_grid_emptiness.py`): only 4.2% of notes are triplet
  payload, 49% of songs gain nothing. **Fit** (`probe_v2_context_fit.py`): the 3× context is CHEAP (bf16). ⇒
  affordable waste beats a fallible deploy-time meter detector → **A1 fixed 48th CONFIRMED.**
- **2a finer quantization** (`for_v2()`, `round_quantize`): success criterion `probe_v2_displacement.py` = triplet
  displacement **50.5 → 0.3 ms** (ρ+0.808 reproduced the diagnosis's +0.83 → strong cross-val). Legacy floor path
  byte-identical.
- **2b beat-sync audio** (`beat_sync`): SIZING `probe_v2_bpm_misalignment.py` **overturned the scope's "2b = smaller
  separable population"** — ~20% of songs, 14.6% ≥23 ms drift (double the triplet tax), second-scale on half-tempo
  sections. REFINED: gate on ACTUAL tempo variation so constant-BPM keeps EXACT v1 features (0.00000 diff).
- **Attribution corrections (method wins this build):**
  - **Measure TRAINING-shaped memory, not a bare `model()`:** the first fit probe (no causal mask) said bf16 B16
    fits at 3.4 GB; the masked training forward OOMs at B4 (T=4608). Real fitted config T=3072/B4 — a launch at the
    default B8 would OOM instantly (Rule 2, memory edition).
  - **Rule 7 twice on 2b sizing:** a hand-rolled avg-BPM slope bug (16× inflation); the corrected number still
    looked alarming until eyeballing the actual songs confirmed the half-tempo mechanism is real.
  - **Cache identity ≠ feature config** ([[dataset-cache-footgun]]): a v2 rebuild silently read stale 2a-only
    features (instant cache hits) — the stamp checks song identity, not extractor config. DELETE the cache on config change.
  - **The autotune skill is stale** ([[autotune-skill-stale]]): benchmarks `train_factorized.py`, not the deployed
    `LayeredTypedChartGenerator` (the `train_motif_figure` lineage) — benchmark the real class directly.
- **Phase 4 retrain DONE (2026-07-05):** cache built (train 4547/val 951 — MORE than v1's 4452, fewer floor-collision
  false-hands rejections). Full 20 epochs, warm-start clean (only `pos_encoding.pe` filtered). **The sparse-target
  worry did NOT materialize** — `val_onset` locked ~0.025 throughout (the ~3× sparser 48th onset target did not
  collapse recall; the pattern head was the loss mover). Best `gen_motif_v2_48th/best_val.pt` val 0.8098, still
  descending at epoch 20 → a continuation (`_cont`, warmup_freeze 0) improves further (~0.772). "Train more" is a
  real (small) gain; `--epochs 30` is the cheap lever if by-ear is close. NOTE: training loss can't confirm the win —
  the triplet-placement fix is invisible to `val_total` (a placement property); Phase 6 by-ear is the binding evidence.
- **Phase 5 decode re-index DONE (2026-07-05, commit `590daa1`):** parameterized the decode phase grid by `subdiv`
  across all `%4` sites (`decode_defaults.phase_band_positions` = single band-math source; apply_phase_calib + generate
  calib/penalty/alloc + phase_shares + chaos gate), threaded from `feat_ext.config.timesteps_per_beat` into BOTH tau
  and generate. Verified subdiv=4 BYTE-IDENTICAL to v1 (bit-equal calib; CLI tests pass; defaults validator ALIGNED).
  - **Method win — the TWO-`t%12` disambiguation:** the user believed Phase 5 was already done; investigation (git log
    + code grep, HARNESS-first) showed the DONE work was `metric_phase` (the INPUT feature, auto-re-indexed in Phase 3
    via `timesteps_per_beat`), NOT the DECODE levers (Phase 5). Two different `t%12` sites (model input vs model output)
    → don't trust a recollection of "did the t%12 stuff" as "Phase 5 done"; verify at the code. Durable docs were right;
    the recollection drifted.
  - **Deliberate deferral (Rule 16):** triplet frames get NO phase band — a triplet-unlock would be a NEW, unvalidated
    lever the hypothesis didn't ask for; the retrained weights place triplets. Add a band only if by-ear shows triplet
    under-placement. SB/tolerance (analysis-only) + governor `frame_hz` left on `t%4` (not decode-critical).
- **Phase 6 by-ear ✅ PASSED (2026-07-05, `notes/playtest_log.md`):** exported the two near-pure-triplet songs with
  `_cont` + `--features highres_v2` + `for_v2()` (A/B vs the v1 set). User: "it totally worked!!!! none of the new note
  frames felt random… resounding 100% success! no degradation… finally able to REALLY express tasty percussion." The
  triplet tax is GONE; the new positions read musical (the no-triplet-band deferral VALIDATED). **Two harness bugs
  caught + fixed IN the export path (HARNESS-first, both user-flagged):** (1) the Phase-6 export TOOLING was assumed
  ready but never wired — `sm_writer` was hard-16th (`ROWS_PER_MEASURE=16`) → parameterized by `timesteps_per_beat`;
  (2) the exporter read the v1 config `msl=1440` → clipped every v2 song to 120 beats/⅓ (the USER caught it by tap
  count 150 vs 450) → use `V2_MSL=5400`. Method win: the user's "150 vs 450 taps" was a HARNESS smell, not a model
  defect — isolated parser (fine, 443 notes on `for_v2`) from export (the truncator) before fixing.
- **v2 DECODE-PLAYABILITY PASS ✅ DONE (2026-07-05 session 2, branch `feat/governor-subdiv-recalib`, `notes/footspeed_floor_findings.md` + `manifold_radar_subdiv_findings.md`).** 4 commits, all subdiv=4 BYTE-IDENTICAL (each proven against v1 before committing — the discipline that de-risked every change):
  - `33de530` **governor subdiv-recalib:** threaded `subdiv` (`frame_hz=BPM·subdiv/60`, `tau_frames/stamina_decay·subdiv`, integer gap/window thresholds ×`f16=subdiv//4`). KEY insight: exertion accumulators/caps need NO rescale (press-rate=frame_hz/gap is grid-invariant). BY-EAR: maxJackRun 3→2.
  - `63125eb` **footspeed floor** (`min_onset_gap` NMS refractory on the precomputed onset tensor; auto=2 on 48th kills 1-frame 48th flams, keeps 2-frame triplet-16ths) + **`--style` density fix** (`style_density*=4/subdiv` — the 16th-grid manifold frac placed 3× too many notes). Floor-ALONE by-ear = "bland/messy" (a blunt crutch, superseded by ↓).
  - `46a25b4` **triplet band** (`onset_phase_calib` 3rd elem `b_trip` on `triplet_band_positions`={2,4,8,10}@12; single-sourced with tau via `phase_calib_offset`). **`b_trip=0.7` BY-EAR WON:** triplet-occ 0.107→0.390 ≈ human, "committal to greens, even rhythm."
  - `ed26aa6` **groove-radar chaos** subdiv-aware (color by quantization denominator → triplet-green 1.25; `dataset.py:104` threads parser tpb). **⚠️ RETRAIN-GATED.**
- **Methodology wins this session:** (1) **verified the coupling before acting** — I nearly refit the manifold, but traced that the v2 model + v1 manifold BOTH trained on tpb=4 radar (read from the cache) → a refit would DE-SYNC = a regression, not a fix. Refit DEFERRED to the retrain. (2) **Walked back my own "refit fixes chaos"** — tracing showed the hard-coded color-map (triplets → 1.0 on BOTH grids), not the parse grid, was the bottleneck (a refit alone = a no-op). (3) **ascii-dump-first for every by-ear complaint** (footspeed flams; the fast-jump hole) — located the exact spots + mechanism before proposing a fix.
- **BY-EAR WIN (`playtest_log.md`):** pt_chaos_v2 (Grand Chariot "brand new note colors… conditioning effective", Take It "flowy streams") validates the grid+band+density-fix together.
- **Current state / OPEN FORK:** **NEXT = the no-fast-jump cap** — fast sub-16th JUMPS evade playability (the fatigue governor governs WHICH-panels not WHETHER; `max_jack_run` is same-panel-only → no two-foot hard cap). FIX = forbid ≥2-fresh-press patterns at `since_onset<f16` → force a playable single, KEEP the onset (user: "don't remove pink notes [=48th]"). **2nd open:** a hold-stream-gate bug on `freeze=high` v2 (Watch Out). THEN the deploy swap (consider `b_trip=0.7` a v2 default). Full status: `notes/data_layer_v2_scope.md`.

## Session 3 (2026-07-06) — v2 decode-playability finish + the SHIP pivot
Believed → learned, three sub-threads then a strategic call:
- **no-fast-jump cap** (sub-16th two-foot JUMPs uncapped): BUILT (`df39c3c`), by-ear PASSED (capped≈uncapped, the
  uncapped arm exposed a "silly" 3-jump-jack). Method keeper: a hard `masked_fill(-inf)` was needed because the
  fatigue governor is a soft re-router that a 2-note jump splits across both feet → never trips.
- **hold-stream "broken" on freeze=high** — the arc's cautionary tale, **mis-analyzed TWICE**: (1) blamed a `dens`
  frame-fraction subdiv bug (real, `e964b1f`, but only HALVED it); (2) declared it fixed on a PROXY metric
  (holds-in-dense-frames), then on a too-narrow metric (pure-16th runs) — both said "clean" while the user's ear said
  "stream in a hold." **Root, found only by DUMPING THE RAW GRID:** 5–6 beat holds with a sustained one-foot 8th
  stream under them. Two attribution corrections: (a) match the metric to the FELT property (a hold in a dense SECTION
  ≠ a stream UNDER a hold); (b) **"stamina is off by default" was FALSE** — `CANONICAL_DECODE["stamina_ceiling"]=50`,
  it was ON the whole time and thins by SALIENCE so it can't shed a LOUD stream (skill text corrected). Real fix
  (position-based `stamina_hold_bump`) DESIGNED + PARKED (`footspeed_floor_findings.md §5b`).
- **b_trip 0.7 vs 1.0**: measured 1.0 triples committed triplets (1.4→4%) but mostly 16th→triplet (pink barely drops)
  → by-ear **inconclusive / song-dependent** — the diagnostic that tipped the call: a GLOBAL phase knob can't be
  per-song-optimal because fine placement is note-context (the seq-onset ceiling), so tuning it is at its ceiling.
- **★ STRATEGIC PIVOT → SHIP.** /experiment-design read (user agreed): the "onset allocation undertuned" gap = the
  note-context placement CEILING (retrain-bound), not an untuned knob. Ship v1.0.0, deploy-swap v2, write the
  safe-settings guide; PARK the research (tolerance formula = already weak; GDL/equivariance = premature; seq-onset
  retrain = the correct-but-deferred fix). Un-park trigger: user says "the times have changed." Memory
  [[ship-mode-park-research]]. **This arc is now PARKED** (its GDL/meter-equivariance deep-math end).

## Session 4 (2026-07-06b) — the SHIP-facing end: b_trip auto-switch + safety envelope + a gate-bug fix
The arc is PARKED at its GDL/research end, but its SHIP-facing tail (checklist #2/#3 of [[ship-mode-park-research]])
was executed. Believed → learned:
- **b_trip AUTO-SWITCH built** (`--auto_b_trip`): per-song, apply the triplet band only where the AUDIO meter
  detector says triplet (`triple_pref>thresh`). Extracted the validated detector from `probe_meter_equivariant_sb.py`
  into `src/data/meter_detect.py` (single source; the probe now imports it, stays the validation reference).
- **Safety-envelope sweep** (`analyze_v2_envelope.py`, v2-aware): 5 arms × 12 songs. **Playability rock-solid
  across the whole range** (0 fast-jumps/flams, jack ≤2, no smear) → the safety zone EXISTS.
- **★ ATTRIBUTION CORRECTION (Rule 9, the arc's headline lesson this session):** the FIRST sweep ran on the narrow
  gate pool (2 triplet songs, both audio-clearly-triplet) and I wrote "auto STRICTLY dominates global." The gate
  fix (below) admitted 6 triplet songs and OVERTURNED it — the ρ+0.47 detector fires on only **3/6** chart-triplet
  songs (misses Sway tf0.61 / After The Rain / Parousia). So auto-vs-global is a real by-ear TRADEOFF. **The lesson
  is Rule 12 (stratify/sample before pooling) compounded with Rule 9: a 2-song "clean win" was small-n optimism;
  don't commit a "best default" from a sample that under-covers the regime the switch is FOR (triplet songs).** The
  fix that surfaced it was itself the diagnostic — the thin triplet coverage was a SYMPTOM of the gate bug.
- **★ A HARNESS BUG the user's question surfaced (HARNESS-first, the arc's recurring win):** user asked "isn't the
  bpm/relax-gate handling needlessly blocking songs?" → traced the code: `--relax_gates` was a **silent no-op on the
  v2 export path** (the `if subdiv==12` branch called `for_v2()` unconditionally with narrow TRAINING gates). Also
  found a THIRD stale gate (`max_simultaneous=2` rejecting hand/quad charts the 15-way head supports). Fixed via a
  shared `INFERENCE_GATES` (bpm[40,320]/len[30,600]/simul4/gimmick); widened is now the v2 export DEFAULT. Measured
  reach with the REAL parser (not a hand-rolled gate re-impl — Rule 2): 532→822 val songs (+55%). Verified
  end-to-end on the exact blocked songs (See Me Now 206s, SHINY DAYS 220bpm+quads).
- **Method keeper — validate the detector on the KNOWN cases before trusting it, and re-validate on the HARD ones:**
  the detector looked clean on the easy 9-song set (+0.81) but its true chart-triplet hit-rate is ~50%; the
  6-triplet sweep was the fair test. A metric that separates the easy cases is not validated on the hard ones.
- **Current state:** the switch + gate fix are SAFE + shipped (uncommitted, `feat/governor-subdiv-recalib`). Open
  fork = the auto-vs-global by-ear verdict (pack `~/sm-generated/v2byear_*`), then the deploy-swap. Notes
  `notes/v2_safety_envelope_findings.md`.

### Session 5 (2026-07-06) — LOW-DIFFICULTY verification + the 16th-grid SNAP (`notes/grid_snap_findings.md`)
Verified the v2 deploy candidate at the LOWER difficulties (it had only ever been by-ear'd on Hard). **v2 generates
Beginner/Easy/Medium coherently** (no degeneration, density tracks, critic reads low, sparse songs ~100% on-grid ==
originals), EXCEPT busy low-diff songs place **8–23% of notes on pure-48th cells `{1,5,7,11}`** the human originals
NEVER use (real 48th-usage ~0% at all low/mid diffs).
- **Attribution WIN (Rules 7–9):** hypothesised the 16th-unlock; **A/B REFUTED it** (unlock OFF didn't drop off-grid;
  the note-count decomposition showed the unlock moves ON-grid density, the 48th count is independent). Testing the
  suspected fix killed it before it shipped. **True cause = the 48th grid's double edge** — the same beat-synced
  sub-16th capability that gives v2 its triplet win also admits 48th jitter on busy DUPLE songs (governor gates
  spacing/jumps, not isolated grid POSITION).
- **Decomposition that set the design (Rule 1):** the off-grid excess is `{1,5,7,11}` (pure-48th noise), NOT the
  triplet family `{2,4,8,10}` (gen ~1% ≈ real) → the fix must PRESERVE triplets (keep-triplets), and Medium's real
  triplet usage (君のハート orig 6%) proves it.
- **FIX = `grid_snap` (decode-only):** `decode_defaults.grid_snap_offset` (−30 logit veto ridden through the
  exporter `harm_off_t` slot → single-sourced into tau + decode; v1 no-op by construction). Off-grid 6.6%→0%,
  density PRESERVED+improved, INERT on already-clean songs (byte-identical) — the ideal targeted knob.
- **WIRED TO CANONICAL (per user ship directive; BY-EAR PENDING):** `--grid_snap auto` (keep-triplets 48th-veto for
  difficulty ≤ Medium — no per-diff threshold needed since real 48th-usage is 0 at all three — OFF at Hard where
  fast 48th runs are legit + the v2 win lives) + `--auto_b_trip` flipped default→True. Validated: v1 byte-identical
  (deployed regime untouched), v2 auto snaps @Easy (48th→0, triplets kept), does NOT snap @Hard. Guard 21→**25 ✓**.
- **Open fork:** by-ear the installed `~/sm-generated/v2_low_*` vs `*_snap`; the Hard boundary is UNTESTED (left on
  canonical). Sub-thread memory [[low-diff-gridsnap]].

## Skills in play
`experiment-design` (this arc is a WIN case — mechanism-grounded metric predicts the ear; harness/units bugs caught
first) · `conditioning-mechanics` §6 (the `t%4` phase grid this questions; the refactor re-indexes it) ·
`generation-defaults` (canonical config the by-ear set replicated; the filter-relaxation touches the inference path) ·
`playtest` (the binding by-ear gate).

## Tooling
`probe_meter_equivariant_sb.py` (rotation-invariant DFT meter detector + meter-equivariant SB) ·
`probe_flip_secondfactor.py` (the 2nd-factor permutation-null hunt, shared with the tolerance downgrade) · census +
displacement + sub-16th checks are inline one-off scripts (reproducible; described in `meter_4_4_assumption_scope.md`).
Notes: `meter_4_4_assumption_scope.md` (the thread), `constraint_relaxation_roadmap.md` (data-layer-v2 bundling),
`playtest_log.md` (the ear verdict). Playtest set `~/sm-generated/meter_triplet_test/`.
