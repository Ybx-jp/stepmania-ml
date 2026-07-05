# Lineage — The 4/4 grid assumption: meter tax + data-layer-v2 refactor (2026-07-04 →)

**One line:** the whole pipeline is hard-wired to 4/4 duple-16th subdivision; triplet/compound songs are
mis-gridded (notes floored onto 16ths, ~33 ms off), CONFIRMED felt + severe by ear → the finer-grid + variable-BPM
"data-layer-v2" refactor is justified (greenlight pending). A model METHODOLOGY WIN thread: a mechanism-grounded
offline measurement (not a taste proxy) PREDICTED the ear, and two of my own harness bugs + a units bug were caught
before any wrong conclusion committed.

**Status:** ACTIVE. Binding gate (by-ear) CLEARED — refactor JUSTIFIED, decision pending. Spun off
[[good-settings-region]] by verifying SB's 4/4 frame. Primary note `notes/meter_4_4_assumption_scope.md`; memory
[[meter-4-4-grid]].

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
- **DEPENDS-ON [[quality-feature-attribution]]:** B-step 2 (meter-as-quality-axis) reuses its graded critic + ICC
  discipline; BPM is the precedent for a hidden per-song quality driver (and the BPM RANGE filter is part of this
  refactor's scope).

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
