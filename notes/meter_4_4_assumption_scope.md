# SCOPE — the whole pipeline is hard-wired to 4/4 (does the metering need a refactor?)

**Opened 2026-07-04** (out of the tolerance-formula thread, verifying the `strong-beat : 4/4` hypothesis).
Status: SCOPING. Two linked sub-threads: (A) the meter-equivariant strong-beat feature — PROTOTYPING now;
(B) **the generator-forced-to-4/4 quality probe — the user's framing, NOT yet run, scoped below.**

## The durable finding (verified at code level, 2026-07-04)
The 4/4 assumption is not just in the tolerance metric `SB` — it is baked through the ENTIRE pipeline:
- **Parser** (`src/data/stepmania_parser.py`): `timesteps_per_beat=4` (each beat = 4 sixteenths, simple/duple
  subdivision) and `beats_per_line = 4.0/len(lines)` at L420/489/556 = **hard 4 beats per measure**. There is
  **NO `#TIMESIGNATURES` handling at all** — grep is empty; every song is force-parsed as 4/4.
- **Audio features** (`src/data/audio_features.py`): the frame hop is `sr·60/(BPM·timesteps_per_beat)` so the grid
  is metrical 16th-notes, but `_metric_phase` (L431-444) encodes beat-phase `t%4` and measure-phase `t%(4·4)=t%16`
  → the model's own conditioning is hard 4/4.
- **Corpus reality:** genuine non-4/4 songs EXIST (e.g. a `#TIMESIGNATURES:0=3=4,135=2.5=4.5,140=3=4` song — 3/4
  with a 2.5/4.5 bar) and ~**10% of charts (489/5068) are triplet-heavy** (chart-derived triplet-note fraction >2%,
  max 0.66). All are silently mapped onto a 4-beat/16th grid.

## (B) The user's framing — deserves its own probe
> "the generator is forced to write in 4/4 regardless of the music's time signature, so any song generated in a
>  different time signature is of dubious quality regardless of settings."

**This is a REPRESENTATION cap, not a decode problem.** If a 3/4 or 6/8 or triplet-feel song is force-gridded to
4/4, the model's barlines/measures and its 16th subdivision do not align with the music's real pulse → notes are
placed at metrically-wrong positions → quality is capped BEFORE any conditioning/decode knob applies. The tolerance
formula (and every other decode result) sits ON TOP of this; a non-4/4 song's "tolerance" is measuring distance from
an already-wrong grid.

**The probe (to run, cheapest-decisive first — no retrain for the diagnosis):**
1. **Size it (no generation):** count corpus songs by meter class — pure-4/4-simple, triplet-heavy (audio triple-comb
   wins / chart triplet-frac high), explicit non-4/4 (`#TIMESIGNATURES`). Is it a 5% curiosity or a 15%+ tax?
2. **Is meter a hidden QUALITY axis (like BPM was)?** Regress per-song generator quality (the graded critic /
   distance-to-real from [[quality-feature-attribution]]) on a meter-mismatch score, controlling for BPM/density.
   The BPM defect (r=−0.68) only surfaced after denoising + ICC — do the same discipline here (Rule: check target
   reliability before concluding no-signal). Prediction: triplet/odd-meter songs score systematically worse, and —
   unlike BPM — the fix is NOT decode-side (`footswitch=False` won't touch it); it needs the representation.
3. **Playtest confirm (by ear):** generate a KNOWN triplet/compound song (First of the Year (Equinox), 0.41 triplet;
   subconsciousness 0.66) and listen — do the triplet runs land on-beat or does the 4/4 grid shear them? By-ear is
   the binding gate (Rule 8).

**If it bites → the metering needs a refactor** (parser reads `#TIMESIGNATURES`; features grid on the song's real
subdivision; the model gets a meter token / the grid becomes meter-adaptive). That is a large structural change —
this probe decides whether it's worth it. **If it's a <5% minority with modest quality loss → document the
limitation and scope to 4/4-simple.** The probe SIZES the tax before anyone commits to the refactor.

## (A) Meter-equivariant strong-beat feature — prototyping (this session)
The GDL framing: `SB` is the projection of onset mass onto the "strong" coset of the cyclic group **Z/4** (simple
meter). Generalize = detect each song's metrical group and project onto ITS strong coset (equivariance to the
metrical group), using the FINE onset envelope (the 16th-hop `onset_env` cannot resolve a triplet). Prototype +
validation in `probe_meter_equivariant_sb.py` / findings appended here. This is the cheap no-generation half; it
also yields the meter-DETECTOR that sub-thread (B) needs to size the tax.

### (A) findings — the prototype RESOLVES (2026-07-04, `probe_meter_equivariant_sb.py`)
A rotation-invariant meter detector recovers a song's subdivision from audio alone, validated against the
INDEPENDENT chart-derived triplet fraction (n=18 spanning chart-triplet 0.0–0.66):
- **Detector: the DFT of a 12-slot beat-phase histogram of the FINE (128-hop) onset envelope.** Triplet energy =
  DFT magnitude at 3 & 6 cycles/beat, duple = 2 & 4; `triple_pref = (triple−duple)/(triple+duple)`. DFT magnitudes
  are rotation-invariant, so the (unreliable) beat OFFSET drops out; the downbeat for the SB readout is recovered
  from the fundamental's phase.
- **Result: `triple_pref` vs chart triplet_frac Spearman +0.47.** subconsciousness (0.66) → triple +0.79;
  First of the Year (Equinox) (0.41) → triple +0.22; all binary songs → duple (−0.3…−0.7). The geometry RESOLVES.
- **TWO HARNESS BUGS caught first (experiment-design Rule 7 — don't blame the concept for a rigged setup):**
  (1) aligning phase with only the FIRST `#BPMS` value drifts over a multi-minute song → histogram smears to noise
  (+0.01); fixed with the full per-segment BPM map (exact beat↔time). (2) fixed histogram cells + ignored `#OFFSET`
  made the statistic rotation-SENSITIVE → it measured the offset error, inverting to −0.58; fixed by going
  rotation-invariant (DFT magnitudes). Only the 3rd iteration is a real test.
- **HONEST caveats (Rule 9):** +0.47 is decent not decisive (n=18; one borderline miss, a "First of the year"
  variant at −0.09 — possibly a wrong audio/BPM match); the triplet songs are only 3/18 (under-sampled — the effect
  needs the triplet-RICH set). And **whether the equivariant SB predicts TOLERANCE better than the duple SB is NOT
  yet tested** — that needs triplet-rich flip labels (generation). The prototype proves the geometry DETECTS meter;
  it does not yet prove the meter-correct reading improves the tolerance formula. Note: SB_eq on triplet songs comes
  out LOWER (subconsciousness 0.61→0.36) — a dense-triplet song is genuinely LESS strong-beat-anchored on its OWN
  grid; whether that maps to higher or lower tolerance is the open question.
- **Reusable spinoff:** this detector IS the meter classifier sub-thread (B) needs to size the 4/4 tax cheaply.

### (B) step-1 CENSUS — the 4/4 tax, sized (2026-07-04, chart-derived over 5345 songs)
Song-level max-chart triplet fraction across the whole corpus (chart signal = reliable ground truth; the audio
detector is NOT needed to COUNT meter, only to detect it when a chart is absent):

| class | songs | % |
|---|---|---|
| pure simple 4/4 (tf<0.02) | 3758 | 70.3 |
| trace triplets (0.02–0.05) | 660 | 12.3 |
| minor triplet fills (0.05–0.15) | 552 | 10.3 |
| substantial triplets (0.15–0.30) | 196 | 3.7 |
| triplet/compound-dominant (≥0.30) | 179 | 3.3 |
| **≥0.15 (real structural triplet tax)** | **375** | **7.0** |
| explicit non-4/4 `#TIMESIGNATURES` | **5** | **0.1** |

**THE KEY REFRAME — the two halves of the 4/4 assumption have VERY different taxes:**
- **Measure-level (`beats_per_measure=4`, odd meter like 3/4, 7/8): tax ≈ 0.1% — NEGLIGIBLE.** This DDR-style
  corpus is genuinely 4/4 at the bar level. The user's "different time signature" concern is real in principle but
  almost absent in THIS data.
- **Beat-level (`timesteps_per_beat=4`, simple/duple SUBDIVISION): tax ≈ 7% structural / 3.3% dominant.** This is
  where the 4/4 assumption actually bites — 4/4 songs with triplet/compound-FEEL (shuffle, swing, 6/8-feel, triplet
  runs) that the 16th grid cannot represent. So "the generator is forced to 4/4" is more precisely **"forced to
  DUPLE-16TH subdivision"** — a ~3-7% song tax, not a time-signature tax.

**Assessment — detector refinement NOT needed now:** the census + the quality probe (B-step 2) both use the
reliable CHART triplet fraction as the meter label; the +0.47 audio detector is only on the critical path for
DEPLOYMENT-time gridding of an unseen song (a refactor-gated use). Its current fidelity already proves meter is
audio-detectable (feasibility), which is all we need pre-refactor. Refine it only if/when we commit to the refactor.

**Open decision (gates the refactor):** is a ~3.3% heavy / 7% structural tax, times its per-song quality loss, worth
a metering refactor? → run **B-step 2** (is meter a hidden quality axis) using chart-triplet labels + the graded
critic, ICC-denoised (the BPM precedent). If the ≥0.30 songs (179) are badly broken → worth it; if mildly off →
document + scope to duple-16th. NOTE the fix cost is large (parser/feature/model), unlike the BPM decode fix.

### (B) step-2 — is meter a hidden QUALITY axis? Verdict: YES at the REPRESENTATION level, INVISIBLE to the critic
Two findings, one structural and decisive:
- **The critic CANNOT see the tax (structural, `stepmania_parser.py:560`).** Parse quantization is `ts = floor(beat_position·4)`
  — triplet notes are FLOORED onto the 16th grid at parse time (0.333→cell 1, 0.667→cell 2). So training targets,
  the critic's inputs, AND generated output are ALL de-tripleted. The tax is **sub-grid** — below the 16th
  resolution every instrument operates at. The cheap cache cut (`quality_features_hard_graded.csv`) confirmed no
  critic signal (deficit vs triplet ρ+0.13, p0.38) — BUT it was also underpowered (1/48 songs triplet>0.05, `p_gen_sd`
  =0 so no reliability, critic saturated at p_gen≈0.05). Do NOT read the null as "no tax"; the instrument is BLIND by
  construction. A 16th-grid critic compares a de-tripleted real to a de-tripleted gen — the lost triplet feel is
  invisible on both sides.
- **The representation DAMAGE is real + quantified (chart-derived, n=597).** triplet_frac vs floor-to-16th timing
  DISPLACEMENT = **Spearman +0.83**. Magnitude: pure-4/4 0.0008 beats (nil) → triplet-dominant **0.0376 beats mean**,
  individual triplet notes up to **0.083 beats ≈ 33 ms @150BPM = 2–3 judgment windows**. It is timing DISTORTION
  (limping), NOT note loss (collision loss only 0.4–2.2%). So triplet songs' charts are systematically shoved
  off-beat by the grid — felt-range, on ~7% of the corpus.

**VERDICT:** meter IS a quality axis, but a REPRESENTATION cap (a sub-grid timing distortion), not something the
current critic/decode stack can measure or fix. "Is the generator worse on triplet songs" is unfalsifiable with a
16th-grid critic; the FELT impact is a by-ear question (step-3). What's PROVEN: the grid displaces triplet notes by
felt-range amounts on ~7% of songs; the fix is necessarily the metering refactor (finer/adaptive grid), not decode.

### (B) step-3 — the binding gate: BY-EAR — ✅ CONFIRMED 2026-07-04 (the tax is felt + severe)
Played `~/sm-generated/meter_triplet_test/` (First of the Year 94% triplet-measures / My Christmas list 80%, 39%
near-pure), canonical defaults, plain generation. **User verdict (`notes/playtest_log.md`):** First of the Year
"felt a little off… it's dubstep, wanted that extra spice"; My Christmas list "really off, i was **badly timing
everything**." **H-meter CONFIRMED:** the 33 ms displacement is FELT; severity tracks triplet CONCENTRATION (Christmas
3× the near-pure measures → dealbreaker vs "a little off") — which a global sync bug would NOT do, so it IS the meter
tax. Both are notated 4/4 (pure duple-16th SUBDIVISION tax, not time signature). The offline 33 ms measurement
PREDICTED the ear (a hard representation fact, not a taste proxy → metric and ear agree). ⇒ **the finer-grid refactor
is JUSTIFIED by ear.** ROI: ~7% of songs but a SEVERE (near-unplayable-timing) defect on them + the sub-16th
intensity vocabulary (§C). The roadmap's "defer until musicality plateaus" gate is met AND the need is now confirmed.

### (C) Does the 16th ceiling CAUSE the backbone cliff? (user hypothesis, 2026-07-04) — partial, not established
User: the 16th ceiling may cause the tolerance/backbone cliff — cranked intensity with no sub-16th release valve
forces the degenerate global off-beat shift; a finer onset space would let the model pack intensity legitimately.
Cheap real-reference + flip-data checks (chart parsing only):
- **Saturation version REFUTED:** the flip is phase reallocation at HELD density ~0.40 (40% of 16th cells) — the
  grid is not near full. Not literal saturation.
- **Missing-vocabulary version REAL but MODEST:** real charts DO use sub-16th more as they intensify (⚠️ after a
  units-bug fix — `16 % d != 0` is the correct "finer than a 16th" test; `d∉{1,2,4}` wrongly counted eighths/16ths).
  Corrected: chart-level density vs GENUINE sub-16th frac Spearman **+0.65** (32nd-family +0.56, triplet +0.54);
  densest real decile ~**10%** sub-16th vs **0%** median; overall mean only 2.9%. So there IS a 32nd-burst/triplet-fill
  intensity device the 16th-capped model emits ZERO of — but it's a ~10% garnish at the extreme, not the main
  intensity mechanism.
- **Causal link NOT established (underpowered null):** real sub-16th frac vs flip g₀ = −0.06 (no relationship), vs
  g₀-residual +0.00 — BUT only 1/28 flip songs is burst-heavy (First of the Year again; the flip set is burst-poor).
  Can't confirm OR refute; a burst-RICH flip set is needed.
- **Competing mechanism (the weight of evidence):** the DOMINANT real intensity mechanism is MORE ANCHORED 16ths
  (chaos→density +0.68, anchoring 0.41→0.73, Rule-5 `real_phase_reference_findings.md`) — already ON the 16th grid,
  available to the model, which SMEARS instead of anchoring. ~90% of the intensity headroom is on-grid and
  mishandled ⇒ the cliff is primarily an ANCHORING/conditioning failure (H4 / the chaos×onset gate's target), NOT the
  grid ceiling.

**VERDICT:** the finer onset space is justified for the METER tax (proven) + a real sub-16th EXPRESSIVENESS
vocabulary (bursts/rolls the model structurally cannot emit) — but is NOT established as the backbone-cliff fix. Grid
refactor and the anchoring gate are COMPLEMENTARY (new vocabulary vs anchored placement), not substitutes; keep the
gate as the cliff lever. Cross-links [[good-settings-region]] (the cliff/tolerance parent) + `real_phase_reference_findings.md`.

### (D) REFACTOR SCOPE — finer grid bundles with variable-BPM + the strict filters (2026-07-04)
User: while refactoring the grid, relax the fixed-BPM + length requirements too (the dataset is stricter than
`generate()`). Rule-0 hit: `notes/constraint_relaxation_roadmap.md` ALREADY bundles "fixed BPM" + "16th-resolution
(triplets/swing)" as **data-layer v2** — the same beat-synchronous re-gridding surgery. Confirmed + scoped:

- **The gates are DATASET-only; `generate()` is filter-free.** BPM range `[60,200]` + length `[75s,130s]` live in
  `StepManiaParser._validate_phase1_requirements` (training-data construction). `generate()` consumes a precomputed
  audio tensor + scalar `bpm` and validates neither. So INFERENCE is already free of them; the exporter only inherits
  them via the dataset path. (User's "stricter than generate()" = correct.)
- **Three separable animals, very different cost:**
  1. **Length `[75,130]`** = a PURE filter, nothing downstream assumes it → trivially relaxable (guarded by model
     `max_len` truncation, already exists). Do anytime, independent.
  2. **BPM range `[60,200]`** = a filter with a JOB: screens GIMMICK/misparsed BPMs (2467/1431/441 = notation tricks,
     not felt tempo) that would feed `hop=sr·60/(bpm·4)` garbage → degenerate tiny-hop features + governor rate
     (`frame_hz=bpm·4/60`) blow-up. Widen to ~`[40,320]` + a gimmick guard (display/mode BPM); don't just delete.
  3. **"Fixed BPM" (tempo CHANGES within a song)** = NOT a filter — the `hop=sr·60/(avg_bpm·4)` GRID assumption. THE
     refactor, and IDENTICAL surgery to the finer/triplet grid (both: frames follow the real beat timeline, not a
     constant-tempo hop). The `bpm_map` time↔beat machinery already written in `probe_meter_equivariant_sb.py` is the
     proven core.
- **What actually BREAKS in data-layer-v2 (nothing crashes; it RE-INDEXES the phase vocabulary):** parser + audio
  features re-grid off the full `#BPMS`/stops map; model retrain (~3× sequence length for 48ths → max_len/memory/AR
  cost); governors get per-frame `frame_hz` (beat-referenced taus survive); and — the sharp part — `t%4`
  (quarter/8th/16th) is baked into `metric_phase`, the `onset_phase_calib` 16th-UNLOCK, `phase_shares`, **SB, and the
  tolerance formula itself** → a finer grid re-indexes ALL of them. So it ripples through `conditioning-mechanics §6`
  + `generation-defaults` = a coordinated grid + phase-vocabulary version bump, NOT a parser patch. Roadmap warning:
  "doing it now would destabilize the exact grid the current H4 work depends on."
- **The roadmap's gate ("defer until musicality plateaus, data quantity becomes the limiter") may now be MET** — the
  2026-07-03 shareable milestone + the fact that the remaining defects (triplet tax, BPM reach) are DATA-LAYER not
  musicality. So the timing instinct is defensible, BUT hinges on the by-ear verdict (step-3): if the triplet set
  doesn't limp, the finer grid loses its strongest justification.
- **Recommended sequencing:** (1) NOW, decoupled + low-risk: relax length + widen BPM (gimmick-guarded) on the
  inference/export path (pure reach, zero grid risk). (2) GATE the big refactor on the by-ear result; if it limps,
  finer-grid + variable-BPM v2 is one coordinated surgery.
  - **UPDATE 2026-07-04 — step (1) SHIPPED:** `StepManiaParser.for_inference()` (BPM `[40,320]`, length
    `[30,600]s`, gimmick guard rejecting raw `#BPMS` events >400) + `export_typed_samples.py --relax_gates`
    (forces `cache_dir=None`) + `scripts/generate.py` warning band widened. Training path byte-identical (guard
    default off, narrow gates unchanged); 7/7 guard/bounds cases + 12 parser tests pass. Step (2) — the by-ear
    result is IN (limps, confirmed above) → the big refactor is now SCOPED as its own arc:
    **`notes/data_layer_v2_scope.md`** (grid-design decision, re-index surface, cost, phased plan). Greenlight
    pending the user's investment-priority call vs the parked seq-onset anchoring retrain.

## Links
Parent thread: [[good-settings-region]] / `tolerance_formula_findings.md` (SB is the tolerance predictor whose 4/4
frame this questions). Method: [[quality-feature-attribution]] (meter-as-quality-axis reuses its critic + ICC
denoise discipline; BPM is the precedent for a hidden per-song quality driver). Discipline: `experiment-design`
(size/ICC before concluding; by-ear gate).
