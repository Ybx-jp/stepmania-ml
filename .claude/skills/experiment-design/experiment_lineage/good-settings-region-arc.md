# Lineage — Good-settings region: tolerance(song) = f(features) (2026-07-03 →)

**One line:** from a SHAREABLE-quality milestone (`chaos=0.9,g=3.0` via `--style`), chase the user's question —
"what song features determine the region of good decode settings? a formula to be derived" — toward a per-song
**tolerance** = how far a song can be cranked before it leaves the REAL high-chaos phase envelope. This arc is a
CASE STUDY in exp-design LOSSES: three metric misreads + one pooled-OOD claim, each caught by the user's ear.

**Status:** ACTIVE. Metric REFEREED by ear (taste_grid): anchoring = an OVERLOAD DETECTOR, not a quality ranker;
tolerance = the anchoring<~0.3 cliff. Actionable rule: crank CHAOS, keep GUIDANCE gentle. **Open fork = the
chaos×onset GATE (conditioning-mechanism ceiling-raiser).** Feature lead: denser songs = lower tolerance (marginal).

**Memory:** roots in the 2026-07-03 milestone (`notes/playtest_log.md`) + `geometry_feasible_region.md` (the
map-the-region ancestor). DEPENDS-ON the chaos-conditioning arc (H4/H14) and [[quality-feature-attribution]]
(graded critic + best-of-N/ICC method). Corroborates [[taste-critic-transfer]] (critic floors on OOD).

## The hypothesis chain (what we believed → what we learned)
1. **Milestone (playtest):** `--style "chaos=0.9,voltage=0.7,air=0.5,freeze=0.5" g=3.0` (manifold snap, realized
   chaos ~0.44) = "exactly what I wanted", SHAREABLE. Opened: map good settings AS f(song features) — "the matrix
   of influential features × conditioning interactions; a formula to be derived."
2. **Scoping:** descriptive-first + graded-critic best-of-N (user choice). **Smoke (`probe_goodregion_sweep.py`)
   killed the scorer:** the graded critic's good-region (LOW chaos, gentle guidance) is the INVERSE of the loved
   corner → it measures REALISM, not taste. *(Later Rule 0: this is the DOCUMENTED H14 "critic floored on any
   off-the-song forced style", not a novel realism-vs-taste gap. I'd re-derived it.)*
3. **User mechanistic Q: why does the backbone flip 1/4→1/16 past tolerance — the governor?** `probe_backbone_phase.py`
   ablation ladder (FULL / GOV_OFF / CALIB_OFF / BOTH_OFF). **Result: NOT the governor (0%, FULL≡GOV_OFF to the
   digit); CFG-amplified chaos ~70% + 16th-unlock calib ~30%; phase REALLOCATION at held density.**
4. **LOSS #1 (Rule 1/8) — user overturned:** I called the flip "the taste axis / user loved it." WRONG — the charts
   the user LIKED **retained** the backbone; the flip is DEGRADATION.
5. **Metric evolution, LOSS #2 (Rule 1) — user overturned AGAIN:** quarter-SHARE (0.00, looked collapsed) →
   quarter-REPRESENTATION coverage with a **±1 window** (0.93, looked robust). Both wrong. The ±1 window miscounted
   a 1/16-OFFSET spine as downbeat coverage. **Settled only by dumping the ASCII GRID (Rule 8):** at g=3.0 Deja loin
   VACATES downbeat+8th and puts every note on a regular `_x.x` 1/16-offset grid — a phase-SHIFTED spine (regular,
   not a smear; not a density flood). *Lesson: look at the artifact FIRST, not after three scalar misreads.*
6. **Rule 0 (user directive "use /experiment-design"):** the phenomenon is DOCUMENTED — `h4_offbeat_signal_findings.md`
   (chaos = global off-grid shift; no local off-beat signal; 16th under-commitment; two retrains: NOT a feature
   problem) + `h14_guidance_sweep_findings.md` (guidance floods off-beats; critic floored on OOD). I built 2 probes
   before checking. Genuinely NEW: governor exoneration + the 70/30 split.
7. **Rule 5 (real reference, `probe_real_phase_reference.py`, real Hard n=176):** real charts get chaotic by
   **ADDING density on a PRESERVED, better-ANCHORED backbone** — chaos→density **+0.68** (≡ H4 +0.63), on_grid
   0.99→0.85, s16 bounded **≤0.15**, anchor **0.41→0.73**. Reframed tolerance = **distance from THIS envelope**
   (metrics: on_grid ~0.85, anchoring ~0.73; both →0.00 in the smear). **Anchoring NAMES the H4 defect:** real =
   ANCHORED coherent runs; generated = UNANCHORED global shift.
8. **LOSS #3 (Rule 9+12) — user caught:** I wrote "the whole `chaos=0.9,g=3.0` regime is OOD" into TWO durable notes
   from **n=2 songs** (Deja loin collapsed) — POOLED a song-STRATIFIED phenomenon and CONTRADICTED the user's ear
   ground truth (they've played several songs there that were "fantastic"). Corrected in-place: it is SONG-DEPENDENT,
   and that dependence IS the subject. *The very claim broke the discipline I'd just invoked.*
9. **Real-anchored sweep (`probe_backbone_tolerance.py`, n=40):** the metrics DISCRIMINATE (anchoring 0.14–0.82,
   where the flawed n=24 quarter-rep gave nothing). First feature signal: **denser songs = lower tolerance**
   (`real_density` ρ≈−0.37, p≈0.02 on all 3 real-anchored metrics) — but marginal (uncorrected across ~30 tests) +
   collinear with onset-busyness. A LEAD, not the formula.
10. **The by-ear REFEREE (taste_grid, `notes/goodregion_findings.md`) — the metric's true role, nailed:** measured
    on_grid/anchoring on the EXACT rated charts (Rule 8). **Anchoring is a validated OVERLOAD DETECTOR, NOT a quality
    ranker** — the 2 cells the user called "overload" are the 2 lowest-anchoring (0.17 vs 0.89, clean), BUT
    Spearman(metric,rating)≈0 and the PEAK (rated 6.5) has only MEDIUM anchoring 0.52. **Inverted-U:** it locates the
    failure BOUNDARY; expressiveness (unmeasured) ranks the good ones ("necessary-not-sufficient", ear-confirmed).
    **Actionable: crank CHAOS, keep GUIDANCE gentle** (peak=chaos0.9/g1.5, overload=chaos0.9/g3.0 → guidance is the
    OVERLOAD lever, H14 ear-confirmed). *Methodology win: don't force a boundary-detector to be a ranker — measure
    what it's FOR (the cliff), let ears own the quality gradient.*
11. **OPEN FORK (ceiling-raiser):** the overload = CFG-amplified GLOBAL chaos shift (root H4: no local off-beat
    signal). Fix = a **chaos×onset GATE** (conditioning-mechanism, tie off-beats to LOCAL perc/onset transients) so
    high chaos ADDS anchored off-beats instead of smearing — retrain/architecture, NOT decode (2 decode retrains
    failed, H4 §6). = the user's "harness it completely". Next thread.

## Methodology LOSSES to learn from (this arc is the cautionary case)
- **Rule 1 (metric must SEE the property) — failed 3×:** quarter-share, then ±1-window quarter-rep, both blind to a
  phase-SHIFTED spine. Mirror of the SKILL's own Rule-1 evidence ("blind to the quarter backbone dissolving").
- **Rule 8 (ground in the artifact) — applied too LATE:** one 8-measure ASCII onset dump settled what 3 scalar
  iterations couldn't. Dump the grid at the FIRST sign of a share/coverage ambiguity.
- **Rule 0 — skipped:** H4/H14 already characterized the phenomenon; 2 probes were partial re-derivations.
- **Rule 9+12 — committed a POOLED conclusion from n=2 against ear evidence.** Song-dependence was the whole point.
## Methodology WINS
- **Rule 5 real-reference** converted an arbitrary threshold into a grounded target AND exposed the (per-song) OOD.
- **Clean one-variable governor ablation** (FULL≡GOV_OFF) decisively exonerated the user's first hypothesis.
- **User-as-ear-referee (Rule 8)** caught all three misreads — the value of a ground-truth oracle in the loop.

## Cross-arc corroboration
- **DEPENDS-ON chaos-conditioning arc (H4/H14, `conditioning-mechanics` §2/§6):** the parent phenomenon (chaos =
  global off-grid shift; guidance floods off-beats). This arc's NEW piece is the real-envelope anchoring metric +
  the governor exoneration; the deeper fix (chaos×onset gate) is H4's conclusion, not decode tuning.
- **DEPENDS-ON [[quality-feature-attribution]]:** reused its graded critic, its best-of-N/ICC reliability method,
  and its feature-regression approach; its "check target reliability before concluding no-signal" applies here.
- **Corroborates [[taste-critic-transfer]]:** the critic's OOD-flooring (why it scores the loved corner 0.018)
  is the same ranking-not-graded limitation that arc found; explains why it can't referee this map.

## Skills in play
`experiment-design` (this discipline — and this arc is its cautionary case study) · `generation-defaults` (canonical
config the probes replicate) · `conditioning-mechanics` §2/§6 (manifold `--style`, CFG on the onset path, phase
metric) · `playtest` (the `taste_grid` by-ear referee).

## Tooling
`probe_goodregion_sweep.py` (critic smoke) · `probe_backbone_phase.py` (governor ablation) ·
`probe_real_phase_reference.py` (Rule-5 real anchor) · `probe_backbone_tolerance.py` (real-anchored per-song sweep).
Notes: `backbone_phase_findings.md`, `real_phase_reference_findings.md`. Data `cache/backbone_{phase,tolerance}.csv`.
