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

## Comparison harness (2026-06-27 — BUILT)

- **`src/generation/playability_metrics.py`** — the ONE importable home for the
  footwork metrics (`chart_metrics`, `same_panel_run_lengths`, `run_length_shares`)
  that the `experiments/generation_typed/*` scripts each re-implement inline
  (`metrics` is copied across ≥6 of them — the drift Rule 14 warns about). New code
  routes through here. **Drift guard:** `test_playability_metrics.py::
  test_chart_metrics_matches_calib` pins it to `calib_foot_fatigue.metrics`, so the
  shared numbers stay comparable to the fatigue-calibration history. (The 6 existing
  copies are left untouched — migrating them is a separate, low-risk cleanup.)

- **`experiments/generation_typed/compare_foot_physics.py`** — the harness. Feeds
  the **REAL onset mask** to every generator so density is held and only PANEL
  CHOICE varies (Rule 11):
    * `real` (human reference) · `foot_phys` (FootPhysicsBaseline) ·
      `model_raw` (learned head, `onset_override`, NO governor) ·
      `model_gov` (learned head + fatigue governor).
  Reports `chart_metrics` + same-panel run-length shares vs `real`, **stratified by
  difficulty** (Rule 12), a distance-to-real summary, and `--export_sm` for an
  ear-check (Rule 8). Verified end-to-end on a random-init model + synthetic song:
  the isolation check holds (identical density across all four generators).

  Run: `python experiments/generation_typed/compare_foot_physics.py --songs 12 --export_sm 2`

## Harness validation pass (2026-06-27 — experiment-design review + first real run)

Ran the experiment-design checklist on the harness BEFORE trusting it, fixed six
gaps, then ran it for real on `gen_motif_full_fixed` + local `data/` (8 rich songs).
**Verdict: the first result is NOT committable — it's confounded. Do the fair test.**

Fixes applied to `compare_foot_physics.py` (all green: 18 drift-guard tests pass):
- **G1 (isolation leak):** onset mask was `(real != 0).any(1)`, which counts hold
  TAILS (symbol 3 = release) as onsets — so foot_phys planted a tap on every hold
  release, worst on the hold-rich charts `by='rich'` selects. Now press-only:
  `np.isin(real, ACTIVE_SYMBOLS).any(1)`.
- **G2 (pinned metric):** `max_jack_run` / `jack≥4%` are cap-pinned at 2 for ALL
  generators (real uncapped), so they can't move or match real. Dropped from the
  `dist→real` scalar; table columns marked `*`.
- **G3 (headline metric absent):** same-panel run-length shares (the stated headline)
  were pooled, not stratified, and not in the distance. Now stratified by difficulty
  and the free len2/len3 shares are folded into per-difficulty `dist→real`.
- **G4 (unfair baseline):** foot_phys ran at default `beta`/`jump_bias`. Added
  `--beta`/`--jump_bias` passthrough so it can be calibrated before any conclusion.
- **G5 (isolation unverified):** added a hard runtime check that each generator's
  PRESS frames == the onset mask. **It immediately fired** — the model paths leak a
  few frames/song (hold-aware rebinds some forced presses into hold bodies/tails).
- **Nit:** clarified `model_raw` = governor-free but still mandatory-playability.

**G6 — the decisive confound the first run surfaced (HARNESS, not model) — NOW FIXED + RE-RUN.**
The charts are SELECTED radar-rich (voltage/stream/freeze), but the model was generated
with difficulty + audio + onset_override and **no `radar=`** → a generic-"Medium" model
asked to reproduce a hand-picked high-voltage human chart (`real` jump% ≈ 44, model ≈ 1).
The fix feeds each chart's OWN measured radar (`s['groove_radar']` — the exact 5-vec the
model TRAINED on, `dataset.py:208`; a full real point, so NO `build_target` conditional-fill
is needed — that's for partial user specs). Onsets stay overridden, so radar steers only
the pattern head; density isolation is untouched.

**Result — the fair test REVERSED the conclusion (experiment-design Rules 7–9 in action).**
`dist→real` over footwork dims (mJumpStrm, len2, len3; jump%/jacks excluded per G2/G6):

| radar | foot_phys | model_raw | model_gov |
|-------|-----------|-----------|-----------|
| OFF (confounded control) | 1.041 | 1.208 | 1.266 |
| **ON (fair)** | 1.041 | **0.603** | 1.117 |

Clean A/B: `foot_phys` is identical in both (it never gets radar — the isolated variable,
Rule 11); the only change is model conditioning, and it flips model_raw from LOSING (1.21)
to clearly WINNING (0.60). **So once conditioned on the same vibe, the learned pattern head
produces footwork closer to the human chart than the min-exertion physics policy** — the
"physics beats the head" reading was a pure missing-conditioning artifact. (Jump% confirms
the mechanism: model_raw 1→25% Medium / 1→34% Hard with radar on.)

G5 (decomposed): DROP=0.00% all generators → density isolation HOLDS; the model's 2.0–2.6%
"reshape→hold" is holds it used where foot_phys can't (freeze-rich set) — a foot_phys caveat,
not a leak.

## G4 calibration — REVERSED the verdict (Rule 15 risk realized)

Built `experiments/generation_typed/calib_foot_physics.py` (numpy-only sweep of foot_phys's
beta/jump_bias, reuses the harness loader+distance — Rule 14). Tuned foot_phys to MINIMIZE its
OWN dist→real (deliberately optimistic FOR the baseline). Best = **beta=2.0, jump_bias=0.0**
(default jump_bias=-2 starved jumps on this jump-heavy set); halved foot_phys's distance.

Final head-to-head, 16 songs, calibrated baseline — **OVERALL dist→real**:

| | foot_phys | model_raw | model_gov |
|--|-----------|-----------|-----------|
| default knobs, 8 songs | 1.041 | 0.603 | 1.117 |
| **calibrated β=2 jb=0, 16 songs** | **0.699** | 0.870 | 1.177 |

Per-diff (calibrated): Medium(11) fp **0.55** < raw 0.88; Hard(4) fp **0.17** < raw 0.62;
Easy(1, outlier) raw 1.11 < fp 1.38. **The earlier "learned head wins" was NOT robust — it
relied on an uncalibrated baseline.** Calibrating flipped it: the tuned physics policy is
closer to real footwork on the 15 Medium+Hard songs.

**⚠️ RETRACTION (2026-06-27, same session) — the "model over-produces jacks" finding was a
HARNESS artifact, the 3rd onset/conditioning attribution slip this session.** I claimed the
learned head over-produces long jacks (maxRun 17–24, ≥4 share 9–14%). The user flagged this is
already-solved, onset-head-mediated work (`foot_exertion_findings.md`, shipped 2026-06-25). Ran
the FAIR test — the model on its OWN onsets + governor (`calib_foot_fatigue.py`, native deploy
mode, NO onset_override):

| regime | maxJackRun | jack≥4 share |
|--------|-----------|--------------|
| real (target) | 3.5 | 0.8% |
| model native, OFF | 6.2 | 2.1% |
| **model native, λ=2 (deployed)** | **3.9** | **0.7%** |
| model under my onset_override (this harness) | 14–24 | 9–14% |

So on its OWN onsets the governed model MATCHES real jacks. The maxRun-24/inflated-≥4 is
`onset_override` forcing the dense REAL rich onset stream (dens 0.39–0.44) through a pattern
head trained to co-operate with its OWN onset head (which sets dens ~0.32 via the manifold) —
an OOD regime. The long-jack tail is an ONSET-HEAD / density condition, not a pattern-head defect.

**This invalidates the head-to-head VERDICTS for the learned model.** onset_override distorts the
model's whole run-length distribution (≥4 share 0.7%→9–14%), and the comparison distance is built
from len2/len3 shares — so BOTH "head wins 0.60" and "foot_phys wins 0.70" measured the model OOD.
Neither is trustworthy. `foot_phys` (native onset-taker) is unaffected; the LEARNED model is the
one put OOD. The G2 "include the jack tail" idea is moot — the tail is a harness artifact, not a
model trait, and re-scoring would just sharpen an artifact.

## Where this leaves the comparison (post-retraction)
The `onset_override` head-to-head is **methodologically invalid for the learned model** (it forces
the pattern head OOD). Don't trust either verdict. Options if we still want "is the head's footwork
more realistic than a physics policy":
1. **Native-mode comparison:** run the model in its DEPLOYED mode (own onset head + radar + governor,
   manifold target density) and compare its footwork DISTRIBUTION to real; give foot_phys the model's
   OWN onsets (so both sit at the deployed density), not the real rich onsets. Densities won't pin
   per-frame, but run-length SHARES are the comparison target anyway.
2. **Repurpose foot_phys as the CRITIC/oracle** (its other intended role, foot_fatigue_design.md ~60):
   score real vs generated charts with the shared cost — no onset_override needed.

The standalone `foot_phys` generator + shared `foot_model`/`playability_metrics` + drift guards are
sound and worth keeping regardless. The CALIBRATION (beta=2, jump_bias=0) is real and reusable.

## NATIVE-MODE comparison — BUILT + RUN (`compare_native.py`, the valid harness)

The deployed model (own onset head, radar-conditioned, density matched to REAL's press density so the
COUNT is held but the model picks POSITIONS = in-distribution; governor+playability on; NO override).
foot_phys (calibrated beta=2/jb=0) takes the MODEL's own onsets → isolates panel choice at a realistic
density. 16 songs. **maxRun is sane now: model 4–9 (real 4–5), NOT the override 17–24 → retraction confirmed.**

OVERALL dist→real (mJumpStrm, len2, len3): **foot_phys 0.731 < model_raw 0.946 < model_gov 1.134.**
Per-diff: Medium fp 0.43 < raw 0.85; Hard fp 0.38 < raw 1.02; Easy(n=1) raw 0.97 < fp 1.38. Density
isolation holds (drop 0.00%; model reshape→hold ~2.8%, benign).

**So even in the FAIR native regime foot_phys is closer — a REAL result, not an artifact. But decompose
WHY before reading it as "physics footwork > learned head" (claim-precision):**
- **Jump-streams dominate the distance and are the KNOWN under-jump gap.** real Hard mJumpStrm 15.0;
  model_raw 6.2 (under-jumps, even with radar); foot_phys 18.2 (its jump_bias was tuned to match). The
  mJumpStrm term is the single biggest distance component for the model — and §8d / conditioning-mechanics
  say the model under-jumps for SEPARATE air/density reasons; do NOT score footwork against it.
- **Same-panel jacks: BOTH miss real, opposite directions.** real ~87% len2 / ~7% len3 / small ≥4; the
  learned head is MORE jack-heavy than human (len3 18–24%, ≥4 ~10–14%); foot_phys is degenerately
  len2-only (99% / 0% len3 / 0 ≥4 — it NEVER makes the 3-jacks real does). foot_phys wins the L1 metric
  only because its len2-heavy error is smaller, NOT because it's human — it can't produce real's jack tail.

**Honest verdict:** at a realistic density, the learned head's panel-choice STYLE is modestly more
jack-heavy than human and under-produces jump-streams (the latter a known separate gap); the physics
policy matches real's len2-dominance + jump-rate better but is itself degenerate (no len3). The metric,
dominated by the under-jump gap, FLATTERS foot_phys. Neither is a clean human match.

## AXIS-SPLIT (`compare_native.py`, jumpDist vs jackDist) — the pooled foot_phys "win" was the jump gap

Split the distance into **jumpDist** (jump_rate + jump-stream = the KNOWN under-jump gap, §8d) and
**jackDist** (full same-panel run-length dist: len2+len3+≥4 = the footwork-STYLE question). 16 songs.

OVERALL (mean over difficulties): jumpDist fp **0.62** / raw 1.06 / gov 1.44; jackDist raw **0.40** /
gov 0.48 / fp 0.48. So the pooled foot_phys advantage was ENTIRELY the jump axis (the model under-jumps,
a separate thread). On the footwork-STYLE axis the three are close.

**But the jackDist "overall" is an equal-weight mean over strata and an n=1 Easy OUTLIER flips it
(Rule 12).** Per stratum jackDist: Easy(n=1) raw 0.41 << fp 0.94 (a bizarre Easy chart: 35% ≥4 runs);
**Medium(n=11) fp 0.23 < raw 0.31; Hard(n=4) fp 0.26 < raw 0.48.** On the reliable bulk (Med+Hard, n=15)
**foot_phys is modestly closer** on footwork style; the overall raw<fp is the single Easy chart getting
1/3 weight. Read the distribution, not the scalar:
- real: ~87% len2, ~5–9% len3, small ≥4 (Med 2.9 / Hard 9.2%).
- foot_phys: 99% len2, ~0 len3, 0 ≥4 — nails the len2 DOMINANCE, makes NO tail (degenerate).
- model_raw: 62–72% len2, 18–24% len3, 9–14% ≥4 — RIGHT structure (it makes the tail real has) but
  WRONG proportions (~2× real's len3, ~3–4× real's ≥4): the head is genuinely more jack-heavy than human.

**Verdict (claim-precise):** they err in OPPOSITE directions and neither is human. foot_phys is too
len2-only; the learned head is too jack-heavy. On Med+Hard, foot_phys's all-len2 sits closer to real's
len2-dominance; the head captures real's jack tail but overshoots it. The governor doesn't improve
jackDist natively (gov ≈ raw, mixed per-diff). The headline "physics footwork > head" is WRONG — pooled
it was the jump gap; split, it's a near-tie where each is unrealistic in a different way.

## Next
- The Easy stratum is n=1 (rich set skews Med/Hard). Don't trust the Easy/overall jackDist; lean on
  Med+Hard. More non-rich / Beginner songs would be needed to characterize the full difficulty range.
- The head's jack-heaviness (len3 ~2× real): INVESTIGATED → notes/jack_heaviness_findings.md. Answer = BOTH
  heads: pattern_temperature (0.7 too greedy, a decode lever, coherence-capped) + the onset head's blocky
  rhythm (8th-heavy, 16th-absent, rest-poor, onset runs ~2× real → jack opportunity; a training thread, NOT
  a temperature — onset_logit_scale is a no-op under thresholding).
- PLAY the native exports (Rule 8): `outputs/compare_native/`. Head jack-heaviness vs foot_phys no-3-jacks.
- Optional: migrate the 6 inline `metrics` copies to `playability_metrics`.
