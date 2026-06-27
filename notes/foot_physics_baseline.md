# Foot-physics baseline generator (the fatigue model, run forward)

**Date:** 2026-06-27. **Branch:** `claude/arrowvortex-linux-compat-75ri8n`.
**Origin:** a look at ArrowVortex's auto-stream feature (`uvcat7/ArrowVortex`,
`src/Editor/StreamGenerator.cpp`) — a two-foot, pad-geometry, weighted-random
stream generator. The takeaway was NOT its mechanism (our per-foot fatigue
governor, §8b in `conditioning-mechanics`, already subsumes it: decay, BPM
coupling, footswitch grading, stamina/arc — none of which AV has). The takeaway
was the **framing**: AV runs a foot-physics cost as a *generator*; we only ever
ran ours as a *penalty on a learned head*. So there was one thing genuinely
missing — a learned-model-FREE generator built from the same physics.

## What was built

1. **`src/generation/foot_model.py`** — the shared, single source of truth for
   pad geometry (`PANEL_POS`, `PAD_DIST`) and the two-foot exertion cost
   (`FootState.eval_pattern` / `.commit`), in numpy. The math is a faithful
   transcription of the fatigue governor in `typed_model.generate`
   (notes/foot_fatigue_design.md): per note, assign arrow(s) to feet at MIN added
   exertion (crossovers when cheaper, no surcharge); stay-&-re-hit costs
   `jack_weight·rate`, move costs `travel_weight·dist·rate`; per-foot exertion
   decays exponentially; a pattern's cost is `min over footings of max(E_L, E_R)`;
   footswitch graded 2-free / 3-penalize / 4-hard-cap.

2. **`typed_model.generate` refactor** — the governor now imports `PANEL_POS`
   from `foot_model` instead of an inline literal. Behavior-preserving (verified:
   `test_fatigue_penalty_*` and all of `tests/test_generation.py` still green).

3. **`FootPhysicsBaseline`** (`src/generation/baselines.py`) — the generator.
   `generate(onsets, difficulty, bpm, rng) -> (T,4)`. Takes **onsets as given**
   and only chooses WHICH arrows, by scoring every candidate placement with
   `FootState` and sampling `P ~ softmax(-beta·E_eff + jump_bias·[jump])`,
   hard-forbidding the unplayable (`E_eff ≥ fatigue_cap`) and over-long jacks
   (`max_jack_run`). Import: `from src.generation.baselines import FootPhysicsBaseline`.

## Why this design (experiment-design alignment)

- **Rule 14 (shared, code-enforced infra):** the governor and the baseline MUST
  agree on the cost math, or a comparison between them measures copy-drift, not
  the hypothesis. They now share `foot_model`. **Drift guard:**
  `tests/test_foot_model.py::test_foot_state_matches_governor_formula` pins
  `FootState`'s cost to an independent transcription of the governor's formula —
  it fails if either side's geometry/weights/footswitch grading changes.
- **Rule 11 (isolate the variable):** the baseline consumes the SAME onsets the
  learned model would place (e.g. `model_chart.any(axis=1)`), so a head-to-head
  isolates **panel choice** (learned pattern head vs hand-coded foot physics),
  holding density/rhythm fixed. It deliberately does NOT decide density.
- **Rule 15 (baseline against the capability you already have):** difficulty here
  is a COARSE knob (nudges jump rate + fatigue ceiling), NOT the radar's
  calibrated conditioning. The baseline's headline use is **pattern realism**
  (same-panel / jump-stream run-length vs REAL, stratified by difficulty —
  the metric the existing `calib_foot_fatigue.py` / `diag_foot_fatigue.py` use),
  not difficulty conditioning. Don't overclaim it.

## What it gives us

- A non-learned floor for the comparison grid alongside `NGramChartModel`
  (audio-blind bigram) and `PerFrameMLP` (audio, no history): foot-physics is
  history- and geometry-aware but has **no learned pattern head**. The question it
  answers: *given identical onsets, does the learned pattern head produce more
  realistic footwork than the easiest-footing physics policy?*
- The **critic/oracle** anticipated in `foot_fatigue_design.md` (lines ~60-68):
  the same simulator that governs decode can now score/seed charts, the bridge
  toward a v2 equivariant fatigue-aware generator.

## Next (not yet done)

- Wire a comparison cell/script: feed real-chart onsets (and model-generated
  onsets) to `FootPhysicsBaseline` and the learned model, run the existing
  run-length-vs-REAL diagnostics, stratified by difficulty.
- Calibrate `beta` / `jump_bias` on the EGREGIOUS rich-Hard set (where the
  governor calibration already lives), not the mild val set.
- Spot-check by EAR (export to .sm via `charts_to_sm`) — metrics are blind to
  musicality (Rule 8).
