# SCOPE — data-layer v2 (beat-synchronous re-grid: finer subdivision + variable BPM)

**Opened 2026-07-04**, out of the 4/4-grid meter thread (`notes/meter_4_4_assumption_scope.md`, memory
[[meter-4-4-grid]]). Status: **SCOPED, greenlight PENDING** (the user's call). This doc is the actionable arc
scope for the BIG surgery; it does NOT re-derive the diagnosis (that lives in the meter scope note). It promotes
section (D) of that note into a standalone plan a future implementation session can execute from.

## Why now (the justification is banked)
- **The triplet tax is BY-EAR CONFIRMED + severe** (meter scope note step-3; `playtest_log.md`): plain-canonical
  triplet songs read off-time ("badly timing everything"), severity ∝ triplet concentration. The offline 33 ms
  floor-to-16th displacement (chart-triplet ρ+0.83) PREDICTED the ear. It's a REPRESENTATION cap, not a decode
  bug — no knob reaches it.
- **Size:** ~7% structural triplet tax (3.3% triplet-dominant); odd-METER is negligible (0.1%). So v2's payload is
  a **duple-16th SUBDIVISION fix**, not a time-signature fix (do NOT scope bar-level odd meter as a priority).
- **The roadmap gate is plausibly MET** (`constraint_relaxation_roadmap.md`: "defer until musicality plateaus /
  data quantity becomes the limiter"): the remaining defects (triplet tax, BPM/length reach) are DATA-LAYER, not
  musicality. CAVEAT: the chaos×onset anchoring gate (the seq-onset retrain, PARKED) is the still-open MUSICALITY
  cliff fix — under the roadmap's own sequencing that arguably precedes a data-layer expansion. v2 and the gate are
  COMPLEMENTARY (new vocabulary vs anchored placement), not substitutes.

## What v2 IS / IS NOT
- **IS:** ONE coordinated **beat-synchronous re-grid** — frames follow the real beat timeline (full `#BPMS`/stops
  map) at a FINER subdivision, replacing today's single constant-tempo hop `hop = sr·60/(avg_bpm·4)`. The
  variable-BPM relaxation and the finer-subdivision fix are the SAME surgery (both: beat-referenced frames).
- **IS NOT:** a parser patch. It re-indexes the phase vocabulary the whole stack is built on (see the re-index
  surface) → a coordinated `conditioning-mechanics §6` + `generation-defaults` **version bump**, plus a full
  model retrain. Nothing crashes at parse time; the damage is silent mis-alignment if done piecemeal.
- **IS NOT** required for BPM/length REACH — that was the cheap decoupled win, now SHIPPED (see sequencing §1).

## The proven core (already written, validated)
`probe_meter_equivariant_sb.py:54-77` — `bpm_map(txt)` builds the piecewise-constant `(start_beats, start_times,
bpms)` map from the FULL `#BPMS`, and `time_to_beat(t, bm)` does drift-free time↔beat over a multi-minute
variable-tempo song. This is EXACTLY the primitive v2's re-grid needs; it caught the "first-BPM-only drifts to
noise" harness bug already. Port it into `src/data/` as the shared timing spine (parser + feature extractor both
call it). `#STOPS`/`#WARPS`/`#DELAYS` are NOT yet handled by it — add them here (a stop = a beat with zero tempo /
inserted dead time).

## The GRID-DESIGN decision (open — pick before implementing)
The subdivision multiplier is the cost driver. Options:

| grid | subdiv/beat | resolves | frame cost vs today | 2-min song frames |
|---|---|---|---|---|
| today | 4 (16th) | duple only | 1× | 1440 |
| **48th (recommended)** | **12 = LCM(4,6)** | **16ths + triplets + sextuplets** | **3×** | **4320** |
| 96th | 24 = LCM(8,6) | + 32nds alongside triplets | 6× | 8640 |

- **DECISION: the 48th grid (12/beat) — CONFIRMED 2026-07-04 (user + the two hardening checks below).** It's the
  MINIMUM that resolves the CONFIRMED triplet tax (and the §C sub-16th expressiveness garnish: 32nd/triplet fills,
  ~10% of the densest real decile). 96th doubles cost for a rare 32nd-triplet combination — defer.
- **Alternative to a fixed finer grid: a meter-ADAPTIVE grid** (detect each song's subdivision via the equivariant
  detector, grid duple songs at 16th and triplet songs at 12th). Cheaper per-song frames, but variable frame
  semantics across songs complicate `t % subdiv` phase conditioning + the model's positional encoding. The fixed
  48th grid is simpler and uniform; prefer it unless the 3× sequence cost proves prohibitive.

### CHECKS (2026-07-04) — A1 CONFIRMED (fixed 48th), the two hardening probes ran
The A1-vs-A2 call hinged on two unknowns; both now measured (probes `probe_v2_context_fit.py`,
`probe_v2_grid_emptiness.py`):

- **Fit (does 3× context train on the RTX 3060?) — YES, with huge headroom.** On the DEPLOYED
  `LayeredTypedChartGenerator` (d128/4dec/2enc, the `train_motif_figure.py` lineage — NOT `train_factorized.py`;
  the autotune skill is stale, [[autotune-skill-stale]]), a real forward+backward at the 48th/4320-frame context:
  - ⚠️ CORRECTION (phase-4 setup): `probe_v2_context_fit.py` measured a NO-MASK forward and was OPTIMISTIC (bf16
    B16 @ 4320 = 3.4 GB). The REAL TRAINING-shaped memory (causal `mask` + teacher-forced decoder + AdamW) is far
    tighter — the O(T²) decoder self-attention dominates: at T=4608, B2 = 8 GB and **B4 OOMs**. ALWAYS measure the
    masked training forward, not a bare `model()` call.
  - FITTED retrain config (the authoritative training-shaped sweep): **T=3072 (256 beats = v1 gen_motif_figure's
    musical coverage, NOT 4608 — same span, ¼ the attention memory), batch 4 = 7.2 GB / 0.355 s per step** (B6=10.8
    GB tight; B8 OOMs). At ~3000 train samples that's ~5 min/epoch → a **~1.5 h warm-started retrain**.
  - The cost is WALL-CLOCK step time (O(T²) attention), and **bf16 (AMP) is mandatory** — it buys the finer-grid
    penalty back (the autotune headline lever). Affordability CONFIRMED; the batch is just smaller than the naive
    no-mask probe implied.
- **Emptiness (how much of the uniform 48th grid is USED?) — 4.2% payload, but affordable.** Over 6034 corpus
  charts: **4.2% of all notes are triplet-family** (land on the NEW cells the 48th grid adds); per-song mean 3.4%,
  median 0.16%. **49% of songs have ZERO triplet notes** (pay 3× context for pure representation-waste), 6.1% are
  structural beneficiaries (≥15% — matches the 7% census). 48th-grid occupancy ~20% (sparse). So A1's waste is real
  and proportionally large (half the corpus gains nothing) — but the fit check shows the 3× it costs is CHEAP.

**VERDICT — A1 (uniform fixed 48th) confirmed.** The cost that would have justified A2's complexity (the 3×
blowup) is affordable on this GPU in bf16; so A1's decisive advantage stands — it needs NO deployment-time meter
detector (can never mis-grid an unseen song, the exact failure we're fixing) and keeps the phase vocabulary a clean
uniform re-index. Two retrain riders the checks surface:
1. **Train in bf16** (mandatory for the throughput, per the fit check) — validate a comparable val metric vs fp32.
2. **The onset target is ~3× SPARSER** on the 48th grid (occupancy 20% vs ~61% at 16th) → the onset head's
   positive base rate drops → revisit the onset `pos_weight`/class weighting on retrain (don't inherit the 16th-grid
   value blindly).

## The RE-INDEX surface (the sharp part — everything keyed on `t%4`)
A finer grid changes what a frame index MEANS, so every consumer of the phase grid must be re-derived, not just
the parser. Grep-confirmed consumers of the hard-4/4 `t%4` / `t%16` grid:
1. **Parser** (`stepmania_parser.py`): `timesteps_per_beat=4`; `ts = floor(beat·4)` quantization (L560 — the
   triplet-FLOOR that IS the tax); `beats_per_line = 4.0/len(lines)` hard 4-beats-per-measure (L420/489/556);
   `hop_length = sr·60/(avg_bpm·4)`. NEW: read `#BPMS`/`#STOPS`, grid on `time_to_beat`, quantize to `beat·12`.
2. **Audio features** (`audio_features.py`): the frame hop (variable now — one hop per BPM segment, not per song);
   `_metric_phase` (L431-444) encodes `t%4` (beat-phase) + `t%16` (measure-phase) → becomes `t%12` + `t%48`.
3. **Model conditioning / decode** (`conditioning-mechanics §6`, carries the ⚠️ 4/4 flag): `metric_phase`, the
   `onset_phase_calib` **16th-UNLOCK** (`(b8,b16)` offsets on 8th/16th frames → must become the 48th-grid phase
   bands), `phase_shares` (quarter/8th/16th → quarter/8th/16th/triplet/…).
4. **Metrics / analysis:** **SB (strong-beat fraction)** and the **tolerance formula** are defined on `t%4` (the
   equivariant SB prototype already generalizes this — it's the meter-correct successor). `probe_*`/`diag_*` that
   compute `t%4` phase shares.
5. **Governors** (`§8`): per-frame `frame_hz = bpm·4/60` becomes segment-local (`bpm·12/60`); beat-referenced taus
   (`fatigue_tau` in beats, `stamina_tau` in beats) SURVIVE unchanged — that's the design payoff of beat-refs.

## Cost / constraints
- **Sequence length:** 48th grid → 3× frames. `config/model_config.yaml max_sequence_length=1440` and the model's
  `pos_encoding max_len=2048` (typed_model.py:37) are BELOW the 4320 a 2-min song needs → raise `max_len` to ~4608
  and `max_sequence_length` to ~4320. **MEASURED (the fit check, see CHECKS above): memory is a non-issue (bf16 B16
  = 3.4 GB / 12.6); the cost is step-time, and bf16 recovers it (bf16 @ 3× ≈ fp32 @ 1×).** So B1 (raise context) +
  bf16 is the answer; B2 (cap length shorter) is NOT needed and would regress reach; B3 (bucketing) is a free
  throughput extra, not required.
- **Retrain:** full model retrain on the re-gridded cache (all features re-extracted). ~3× the per-song compute +
  the longer context. The `autotune` skill is the tool for batch-size/AMP/bucketing under the new sequence length.
- **Deployment:** the equivariant meter detector (`probe_meter_equivariant_sb.py`, ρ+0.47) moves onto the critical
  path IF v2 grids an unseen inference song by detected meter; for a fixed 48th grid it's not needed (grid
  everything at 12/beat). Refine the detector only if the adaptive-grid option is chosen.

## Phased plan (proposed)
0. **Grid-design decision** (fixed 48th vs adaptive; the `max_len` strategy). Cheapest, gates everything.
   → ✅ DONE: A1 fixed 48th CONFIRMED (grid-design section + CHECKS); `max_len` ~4608 + bf16 (fit check).
1. **Timing spine:** port `bpm_map`/`time_to_beat` to `src/data/`, add `#STOPS`/`#WARPS`. Unit-test against the
   probe's validated outputs (drift-free beat↔time on a variable-BPM song).
   → ✅ DONE: `src/data/timing.py` (`TimingMap`: beat↔time + STOPS + `frame_beats`/`frame_times(subdiv=12)`),
   `tests/test_timing.py` (9 tests: hand-computed BPM/variable/stops, probe-equivalence, 48th grid). `#WARPS`
   (negative/skip time) NOT yet handled — rare; add in phase 2 if the corpus needs it.
2. **Parser re-grid:** quantize to `beat·12`, hop per BPM segment. Verify the triplet DISPLACEMENT metric
   (chart-triplet vs floor error, currently ρ+0.83 / 33 ms) collapses to ~0 on the triplet set — the concrete
   success criterion for the representation fix.
   → ✅ 2a DONE (finer-grid quantization): centralized `_beat_to_ts` helper + `round_quantize` flag +
   `StepManiaParser.for_v2(subdiv=12)` (round-to-nearest 48th); legacy 4-grid floor path byte-identical (87
   tests pass). **Success criterion MET** (`probe_v2_displacement.py`, 6034 charts): triplet-note displacement
   **0.1263 → 0.0009 beats (50.5 → 0.3 ms @150BPM); structural triplet songs 19.4 → 0.8 ms; ρ+0.808 → +0.344**
   (residual = genuinely-sub-48th nesting, ~0.002 beats, musically nil — the probe reproduced the meter thread's
   +0.83 on the deployed grid). Unit tests `tests/test_v2_quantize.py`.
   → ⬜ 2b PENDING (variable-BPM audio re-grid): replace the single avg-BPM `hop_length` in `_calculate_audio_
   alignment` + `audio_features.py` with per-frame `TimingMap.frame_times(subdiv=12)`. Separable from 2a (2a fixes
   fixed-BPM triplet DISPLACEMENT — chart-space; 2b fixes tempo-CHANGE songs — audio-space). `#WARPS` if needed.
3. **Feature re-grid + `_metric_phase` → `t%12`/`t%48`.** Rebuild the highres cache (new cache key/version).
   → ✅ PLUMBED + DE-RISKED: `highres_v2` feature spec (`decode_harness._FEATURE_SPECS`, `timesteps_per_beat=12`,
   cache `cache/samples_v3_48th`, still 42-dim — only hop density + metric_phase period change; `_metric_phase`
   re-indexes to `t%12`/`t%48` AUTOMATICALLY via config, so phase-5's metric_phase piece is free). Alignment
   DE-RISK (`probe_v2_alignment.py`) on real chart+audio: v2 audio frames == v2 chart timesteps, exactly 3.00×
   v1, dim 42 — the "piecemeal drift" risk is closed. Side benefit: v2 admits MORE difficulties (fewer floor-
   collision false-hands rejections in `validate_pattern_quality`). MUST pair `highres_v2` with
   `StepManiaParser.for_v2()`. → CORPUS RE-EXTRACTION command (into `cache/samples_v3_48th`, ~5.2 GB, ~4.5 h on
   4 cores; verify done via `ls cache/samples_v3_48th/{train,val} | wc -l` vs the split counts train 4452/val 954,
   minus audio-not-found skips): `python experiments/generation_typed/warm_cache_v2.py --data_dir data --audio_dir
   data --v2 --workers 4 --cache_dir cache/samples_v3_48th` (OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 to avoid BLAS
   oversubscription). Cache is msl-keyed at 5400 (the v2 sequence length).
4. **Model retrain** at the new sequence length (config bump + `autotune` for throughput).
   → ✅ TRAINER READY + DE-RISKED (`train_motif_figure_v2.py`): version bump of `train_motif_figure.py` — same
   arch/heads/conditioning, only the data grid changes. WARM-STARTS from the deployed `gen_motif_full_fixed`
   (every learned weight transfers; only the `pos_encoding.pe` buffer is filtered — 2048→5504 shape, rebuilt fresh
   = correct; verified loads clean, unexpected=[]). bf16 + T=3072/B4 fitted config baked in. Onset loss unchanged
   focal_bce — the 48th onset target is ~3× sparser, so WATCH val_onset epoch 1 and retune gamma/pos_weight ONLY
   if recall collapses. → ⬜ REMAINING: launch it once `cache/samples_v3_48th` finishes (`python experiments/
   generation_typed/train_motif_figure_v2.py --data_dir data --audio_dir data`).
5. **Re-index the phase vocabulary:** `onset_phase_calib`, `phase_shares`, SB/tolerance, the governors'
   `frame_hz`. Update `conditioning-mechanics §6` + `generation-defaults` in lockstep (version bump).
6. **Validate:** by-ear on the SAME triplet set (`~/sm-generated/meter_triplet_test/`) — the binding gate that
   opened this thread; the limp should be GONE.

## Risks / mitigations
- **Destabilizing the current H4/anchoring grid** (roadmap's explicit warning). Mitigation: v2 is a version bump,
  not an in-place edit — keep the 16th-grid model deployed until v2 validates by-ear.
- **Piecemeal drift:** re-gridding parser but not features (or vice-versa) silently mis-aligns audio↔notes.
  Mitigation: the shared timing spine (phase 1) is the single source; both call it.
- **Sequence-length blowup** un-budgeted (see Cost). Mitigation: decide the `max_len` strategy in phase 0.
- **Scope creep into odd-meter/bar-level:** NOT justified (0.1%). Keep v2 to subdivision + variable-BPM.

## Sequencing vs the cheap win
1. **✅ SHIPPED (2026-07-04, decoupled, zero grid risk):** relax length + widen BPM (gimmick-guarded) on the
   INFERENCE/export path. `StepManiaParser.for_inference()` (BPM `[40,320]`, length `[30,600]s`, gimmick guard on
   raw `#BPMS` events > 400) + `export_typed_samples.py --relax_gates` + `scripts/generate.py` warning band. Pure
   reach to songs `generate()` can already chart; training path byte-identical (guard default off).
2. **⬜ v2 (this doc):** GATED on greenlight. By-ear justification is banked; the open question is INVESTMENT
   PRIORITY vs the parked seq-onset anchoring retrain (the musicality cliff). Both are big; the user's call on
   which goes first.

## Links
Diagnosis: `notes/meter_4_4_assumption_scope.md` (census → damage → critic-blind → by-ear → §C 16th-ceiling →
§D scope seed). Lineage: `experiment_lineage/meter-grid-arc.md`. Roadmap: `notes/constraint_relaxation_roadmap.md`
(fixed-BPM + 16th-resolution bundled as data-layer v2). Memory: [[meter-4-4-grid]]. Skills to version-bump in
lockstep: `conditioning-mechanics §6` (the `t%4` phase grid), `generation-defaults §1` (canonical config).
Complement (not substitute): the chaos×onset anchoring gate / seq-onset retrain (`notes/chaos_onset_gate_scope.md`,
the musicality cliff). Proven core: `probe_meter_equivariant_sb.py:54-77`.
