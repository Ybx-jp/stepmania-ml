# Lineage — Good-settings region: tolerance(song) = f(features) (2026-07-03 →)

**One line:** from a SHAREABLE-quality milestone (`chaos=0.9,g=3.0` via `--style`), chase the user's question —
"what song features determine the region of good decode settings? a formula to be derived" — toward a per-song
**tolerance** = how far a song can be cranked before it leaves the REAL high-chaos phase envelope. This arc is a
CASE STUDY in exp-design LOSSES: three metric misreads + one pooled-OOD claim, each caught by the user's ear.

**Status:** ACTIVE, but the formula's strength is now DOWNGRADED. **FORMULA EAR-CONFIRMED 3/3 (2026-07-04)** —
`env_strongbeat_frac` (SB, onset-envelope mass on strong beats) predicts the per-song FLIP GUIDANCE g₀≈0.77+1.62·SB
and passed the ear prospectively (the first ear result on this thread to AGREE with the offline metric) —
`notes/tolerance_formula_findings.md` + `playtest_log.md`. **BUT the expanded k=4 run (n=32) + a permutation-null
second-factor hunt CUT AGAINST the headline (2026-07-04 cont.):** `SB→g₀` fell from +0.72/R²0.44 (n=14, small-n
optimism) to **+0.29 clean / +0.39 censored (p=0.027), R²~0.09, LOO-CV R²≈0**; and **NO audio feature (84-dim
fingerprint) beat chance** as a 2nd factor. SB is a real but WEAK single-factor RANK predictor; the high-SB fork
(resisters vs early-flippers) is NOT audio-poolable → consistent with the parked gate's "placement is note-context."
Metric REFEREED by ear (taste_grid): anchoring = an OVERLOAD DETECTOR, not a quality ranker. Actionable rule stands:
crank CHAOS, keep GUIDANCE gentle — a per-song budget (low SB → tighter g ceiling), but a WEAK one. Open menu:
operationalize SB as an honest rank heuristic / chase the fork via note-context / scale n / declare + move to the
gate. Parked fork = the chaos×onset GATE (conditioning-mechanism ceiling-raiser).

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

## 2026-07-04 — the GATE fork investigated: decode EXHAUSTED → note-context retrain de-risked GREEN → PARKED
The referee named the ceiling-raiser (chaos×onset gate). This session ran it to a go/no-go. Full plan
`notes/chaos_onset_gate_scope.md`.
- **Believed:** tie off-beat placement to LOCAL AUDIO (perc/onset dim35/41) so high chaos ADDS anchored off-beats.
- **Phase-0 decode probe (`probe_chaos_onset_gate.py`) — DECODE EXHAUSTED.** Built the gate as a per-frame
  `onset_logit_offset` (the `--harm_calib` path; single-sourced `decode_harness.chaos_onset_gate_offset`). Three
  arms, EACH one change vs canonical (unlock kept ON): ADD (additive) WORSENED the smear; DESMEAR (subtract in
  low-saliency zones) un-smeared the overload (HSL anchor .08→1.0) **but crushed the GOOD songs' loved 16ths
  IDENTICALLY** (GC s16 .44→.01 == HSL .95→.01). ⇒ **off-beat placement is NOT in audio** (H4 at the decode surface):
  audio saliency can't separate an expressive 16th from a smear 16th.
  - **ATTRIBUTION CATCH (Rule 11, user-caught):** my FIRST arm set turned the canonical unlock OFF + added the gate
    = TWO changes → I misread the good-song collapse as "the gate over-corrects" when it was the unlock removal.
    Corrected to one-change arms; the corrected run is the decisive one. A canonical-defaults + one-change violation
    — the 5th ear/user catch on this thread. **KEEP THE CANONICAL PALETTE FIXED; change one lever.**
- **PIVOT:** placement is in NOTE-CONTEXT, not audio → the retrain is the **seq-onset head revived with chaos** (not
  FiLM-on-audio). **Stage-1 de-risk GREEN (`probe_seqcontext_chaos.py`):** on HIGH-chaos Hard charts the frozen
  decoder's `h` predicts 16th placement at conv-readout AUC **0.862 ≈ ceiling 0.858 ≫ audio 0.618** (recovers 102%
  of the note-context gap; positive control fired; STRONGER than tame charts 0.771). A learned note-context gate CAN
  place high-chaos off-beats. Binding risk = free-run DRIFT (Stage-2, scheduled sampling; not settled — teacher-forced).
- **State: PARKED by user** ("not the right direction now"); plan docs intact. **Active reverts to the FORMULA
  derivation** (scale the ρ≈−0.37 density lead: more songs + partial correlations).
- **Method keeper:** de-risk a retrain CHEAPLY first (Rule 6) — the frozen-`h` chaos-stratified probe gave a clean
  go/no-go in minutes, vs H4's two BLIND feature retrains that burned cycles.

## 2026-07-04 (cont.) — FORMULA derivation: the DEPLOYABLE predictor found (audio strong-beat mass)
After parking the gate, reverted to the formula goal. Cheap-first (Rule 6) on the EXISTING n=40 CSV, then a
no-generation deployability probe. Full note `notes/tolerance_formula_findings.md`.
- **Partial-correlation pass (no gen):** the ρ≈−0.37 `real_density` lead is CLEAN — orthogonal to the audio-busyness
  block (correlates +0.01–0.26 with it), survives controlling for the whole block (−0.36/−0.37), only significant
  predictor; the HANDOFF's "density⊥busyness collinearity" worry was UNFOUNDED. **BUT `real_density` is the REFERENCE
  chart's density → not deployable on an unseen song, and no audio feature proxies it (best 0.45).**
- **Deployability check (user's call) — WON DECISIVELY (`probe_tolerance_audio_density.py`):** the audio-derivable
  **`env_strongbeat_frac`** (fraction of onset-envelope MASS on strong-beat frames t%4∈{0,2}) predicts tolerance at
  ρ≈**+0.63/+0.56/+0.63** (p<0.001) — ~2× density, R²≈0.33 from AUDIO ALONE (no reference chart, no model forward).
  **SUBSUMES density** (partial strongbeat|density +0.60; density|strongbeat →−0.25 n.s.) → the density lead was a
  SHADOW of audio on-grid-ness. LOO-stable [+0.61,+0.66], Spearman≈Pearson. Sign is mechanism-correct (on-grid audio
  energy resists the H4 global off-grid smear). **Rule-8 retrodiction:** the two EAR-caught smear songs (Deja loin,
  HSL) sit at the BOTTOM of the predictor; GC/NIM at MEDIUM = "great at g1.5, overload at g3.0" (matches the referee).
- **Honesty (Rule 9):** pre-registered hypothesis was the MODEL's p_onset strong-beat frac (NULL −0.09); the RAW
  ENVELOPE variant won (mechanism confirmed, operationalization corrected). n=40, k=2 label noise → ρ likely
  ATTENUATED (true effect stronger). **NOT yet PROSPECTIVELY ear-tested** — the binding next gate = generate the
  milestone crank on a fresh predicted-LOW vs predicted-HIGH SB song and play them (Rule 8).
- **FLIP-POINT experiment (user: "predict which guidance flips a song"; `probe_flip_point.py`):** Rule-8 on the n=40
  curves = 32/40 monotone anchoring CLIFFs (intuition confirmed). A focused DENSE 8-pt guidance × k4 sweep + a
  logistic-cliff fit (g₀=inflection) → **SB predicts the flip guidance g₀ at Spearman +0.72 (p=0.003), R²=0.44,
  resid ±0.28 guidance-units** (denser grid + k4 HALVED the coarse-data ±0.58; fits r²=1.00 = the cliff is literal).
  **Formula `g₀ ≈ 0.77 + 1.62·SB`** = a per-song guidance CAP. SB subsumes density again (+0.65 partial); BPM null.
  Cliff SHARPNESS w is an INDEPENDENT axis (SB predicts WHERE not HOW SHARP). Caveats: n=14 clean fits, in-sample
  band, no taste_grid ear-anchor in the subset, outliers LOVE/BUMBLE BEE (SB ~44% of variance).
- **PROSPECTIVE EAR VALIDATION — the binding gate CLEARED (2026-07-04, `notes/playtest_log.md`):** 3 songs spanning SB
  × {below-g₀, above-g₀} at the milestone chaos spec (exported charts re-measured at the k4 anchoring means = faithful).
  **User CONFIRMED 3/3:** every SAFE chart coherent, every OVERLOAD degraded; the g=2.0 same-guidance-opposite-verdict
  cross-check landed (Heart Attack overloaded @2.0, Take It fine @2.0 / only "degraded not ruined" @3.0 = the high-SB
  shallow-cliff prediction). BONUS: Heart Attack g=1.0 MORE expressive than g=2.0 → inverted-U re-confirmed by ear,
  g₀ is a SAFETY CEILING (recommended setting sits gentle-side). **FIRST ear result on this thread that AGREED with the
  offline metric** (the prior 5 were ear-overturned misreads). Caveat: n=3 by ear.
- **State:** formula's CORE TERM (SB) derived + offline-validated + retrodicts the ear failures + predicts the FLIP
  GUIDANCE g₀ to ±0.28 + now PROSPECTIVELY EAR-CONFIRMED (3/3). Deployable: SB<0.40 low tolerance (keep g≤1.5),
  SB>0.65 high; g₀≈0.77+1.62·SB (a ceiling; recommend gentle-side). Open: a "recommended guidance" (the expressiveness
  peak below g₀, not the ceiling); optional out-of-sample / more-songs for a tighter band. User's call whether to push. **Method keeper:** the deployability check didn't just clear the bar — a
  mechanism-faithful audio feature BEAT and SUBSUMED the reference-chart lead. Reframe "predict density" → "predict
  the target directly" (we didn't need to proxy real_density; the audio feature predicts tolerance outright).

## 2026-07-04 (cont.) — EXPANDED run + 2nd-factor hunt: the formula DOWNGRADED, no audio 2nd factor (a methodology WIN)
The "optimize the fit" step ran to completion and **overturned its own prior headline** — captured here while fresh.
- **Expanded flip run (`cache/flip_point_v2.csv`, 32 songs, SB 0.07–0.84, dense 8-pt × k4):** `SB→g₀` did NOT tighten
  as predicted; it WEAKENED. +0.72/R²0.44 (n=14) → **+0.29 clean n.s. / +0.39 censored p=0.027, R²~0.09, LOO-CV R²≈0.**
  The 0.44 was **small-n optimism** (14 songs); more songs regressed it toward the true weaker value. The `fit_ok=0`
  fallbacks (high-tolerance resisters Abyss/Dead Heat/ONE TWO) must be kept as CENSORED, not dropped — dropping them
  is what pushed the reported clean number to n.s. **Method note: `probe_flip_point`'s default printout is
  pessimistically censored; the +0.39 censored figure is the honest one.**
- **Second-factor hunt (`probe_flip_secondfactor.py`) = CLEAN NEGATIVE, and it's a WIN for the discipline.** 84-dim
  pooled fingerprint vs the g₀ residual after SB, judged by **LOO-CV increment + a 500-perm NULL** (n=28 vs 84
  candidates → in-sample R² is guaranteed to "find" a winner; the null is the only fair judge). Best real ΔCV +0.267
  < null 95th pct +0.387, **p=0.23** (censored p=0.60). The **negative control fired** (`real_density`: in-sample↑
  +0.106, LOO-CV↓ −0.076) — the harness can distinguish signal from overfit, which licenses trusting the null.
- **Mechanistic read:** the weakening localizes to a HIGH-SB FORK (Take It/BUMBLE resist; MEANING OF LIFE/And Then We
  Kiss/LOVE flip early at the same SB). No pooled-audio feature separates them → the discriminator is NOT audio-poolable
  at n=32. This DOVETAILS with this arc's own Phase-0 gate finding (off-beat placement not audio-reachable) and the
  seq-onset arc's frozen-`h` result: the missing 2nd factor most plausibly lives in NOTE-CONTEXT.
- **The 3/3 ear result SURVIVES** — the 3 tested songs sit on the clean SB spine (v2 g₀ within ~0.27 of formula). The
  ear test was real; it just wasn't a fair sample of the fork. *(Rule 8/9: the offline downgrade does NOT retract the
  ear win; it bounds what SB explains — a rank bracket, not a variance model.)*
- **Methodology WIN (new):** this is the counter-example the arc needed — a disciplined LOO-CV + permutation-null
  design that *demoted its own favorite result* instead of an ear-caught misread. The prior wins were the ear catching
  the metric; this one is the STATS catching the small-n optimism BEFORE the ear had to. Small-n R² is the tell:
  always LOO-CV, and permutation-null any best-of-many-features selection.

## Cross-arc corroboration
- **SPUN OFF → `meter-grid-arc.md` ([[meter-4-4-grid]]):** the downgrade's unexplained high-SB fork prompted the user
  to question SB's 4/4 frame → the whole 4/4-grid meter thread. Meter was RULED OUT as the fork's cause (only 1/32
  flip songs triplet-heavy) but is a real correctness issue; that thread's data-layer-v2 refactor would re-index SB +
  the tolerance metric here. Reciprocal link in the meter arc.
- **DEPENDS-ON chaos-conditioning arc (H4/H14, `conditioning-mechanics` §2/§6):** the parent phenomenon (chaos =
  global off-grid shift; guidance floods off-beats). This arc's NEW piece is the real-envelope anchoring metric +
  the governor exoneration; the deeper fix (chaos×onset gate) is H4's conclusion, not decode tuning.
- **CONNECTS-TO / REVIVES seq-onset-arc.md:** the gate's need for a placement signal is answered by the seq-onset
  arc's frozen-`h` finding (note-context predicts placement). Phase-0 (decode) + Stage-1 (frozen-`h` chaos-stratified
  0.862) merge the two threads; the parked seq-onset build is the gate's Stage-2 substrate. Reciprocal link added there.
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
GATE fork (07-04): `probe_chaos_onset_gate.py` (decode gate, ADD/DESMEAR arms) · `probe_seqcontext_chaos.py`
(frozen-`h` chaos-stratified de-risk) · `decode_harness.chaos_onset_gate_offset` + `--chaos_onset_gate` flag.
Formula tooling: `probe_tolerance_audio_density.py` (the DEPLOYABLE predictor — base p_onset + env strong-beat/
occupancy features, merges with the tolerance CSV) → `cache/tolerance_audio_density.csv`; `probe_song_similarity.py`
+ `cache/{audio_fingerprints_highres,song_bpms}.npz` (per-song audio feature table). Notes:
`tolerance_formula_findings.md` (the predictor + the downgrade), `backbone_phase_findings.md`,
`real_phase_reference_findings.md`, `chaos_onset_gate_scope.md`. Flip-point: `probe_flip_point.py` (logistic-cliff
g₀ fit; expanded → `cache/flip_point_v2.csv`) + `probe_flip_secondfactor.py` (LOO-CV increment + permutation-null
2nd-factor hunt). Data `cache/backbone_{phase,tolerance}.csv`, `cache/tolerance_audio_density.csv`,
`cache/flip_point{,_v2}.csv`, `cache/chaos_onset_gate_v2.log`, `cache/seqcontext_chaos.log`.
