# Backbone phase flip (1/4 → 1/16) under cranked conditioning — attribution

**Question (user, 2026-07-03):** when conditioning is cranked past a song's tolerance, the generated chart
"sacrifices a 1/4 backbone for a 1/16 backbone." Is that the fatigue/stamina governor? Probe it.

**Answer: NO — the governor is a bystander (0% effect). The flip is onset-side: CFG-amplified chaos
conditioning owns ~70%, the 16th-unlock calib ~30%, and it is a PHASE REALLOCATION at held density, not a flood.**

> **UPDATE (2026-07-03, Rule-0 + Rule-5 reconciliation — read this first).** The *phenomenon* here is NOT new: it
> is the documented **H4/H14 chaos degeneracy** ("guidance floods off-beats; chaos = a global off-grid shift the
> model renders uniformly because it has no local off-beat signal"; `h4_offbeat_signal_findings.md`,
> `h14_guidance_sweep_findings.md`). This note's genuinely-new contributions are narrow: **(a) the governor is
> exonerated** (H4/H14 predate it) and **(b) the ~70/30 CFG-vs-calib decomposition.** Frame it as *confirms +
> decomposes*, not *discovers*.
> **Bigger correction (Rule 5, `probe_real_phase_reference.py`, real Hard n=176):** `chaos=0.9,g=3.0` is
> **SONG-DEPENDENT, not OOD as a regime** — the user has PLAYED several songs there that were "fantastic" (ear
> ground truth). SOME songs collapse to a degenerate off-grid smear far outside real (Deja loin: s16~1.0 vs the
> real Q4 max ~0.15); OTHERS stay real-like. Whether a song goes degenerate at a given crank is exactly the
> per-song **tolerance** this thread maps — do NOT pool it into "the regime is OOD" (my earlier claim broke
> Rules 9+12: committed a pooled conclusion from n=2, against the ear evidence). Real charts get
> chaotic by **ADDING density on a PRESERVED backbone** (chaos→density +0.68, on_grid only 0.99→0.85, and the
> added 16ths become **more anchored** 0.41→0.73). So Deja-loin's "quarter backbone dissolves" is a per-song OOD
> collapse, and the
> right metric is **distance from the real high-chaos envelope** — primarily **on-grid share** (real Q4~0.85) and
> **16th-anchoring** (real Q4~0.73), both →~0.00 in the generated smear. The real-vs-degenerate gap is exactly
> **anchored coherent runs vs an unanchored global shift** (the H4 mechanism, now measurable). The `q-share`/
> `quarter-rep` framing below is SUPERSEDED by these real-anchored metrics.
> **Also:** the graded-critic 0.018 at this corner is the **documented OOD-flooring** ("the taste critic is floored
> on any off-the-song forced style", H14), NOT a novel realism-vs-taste gap.

## Method
`probe_backbone_phase.py`. Fix the milestone HIGH-chaos `--style` spec `chaos=0.9,voltage=0.7,air=0.5,freeze=0.5`
(manifold-snapped, realized chaos ~0.44), sweep CFG `guidance ∈ {1.0,1.5,2.0,3.0}`, and measure the generated
backbone's phase shares (`decode_harness.phase_shares`: quarter t%4==0, 16th-offbeat t%4∈{1,3}) under a
one-variable ablation ladder. 2 Hard songs (Deja loin, ヤマト), k=3 gens/cell, shared RNG across arms.
Real-Hard backbone anchor ≈ (q 0.71, s16 0.04).

Arms (calib baked into BOTH tau and decode per arm, per generation-defaults §3):
- **FULL** deployed (governor ON, `onset_phase_calib=(0,1.0)` ON)
- **GOV_OFF** `fatigue_penalty=None, stamina_ceiling=None`
- **CALIB_OFF** `onset_phase_calib=(0,0)`
- **BOTH_OFF** governor + calib off

## Results — the flip develops monotonically with guidance (q drops, s16 rises)
Deja loin (real q=0.73, s16=0.00):
```
arm         g1.0        g1.5        g2.0        g3.0
FULL        q0.49/s0.15 q0.25/s0.59 q0.04/s0.93 q0.00/s1.00
GOV_OFF     q0.49/s0.15 q0.25/s0.59 q0.04/s0.93 q0.00/s1.00   ← IDENTICAL to FULL
CALIB_OFF   q0.53/s0.00 q0.48/s0.02 q0.37/s0.12 q0.04/s0.65
BOTH_OFF    q0.53/s0.00 q0.48/s0.02 q0.37/s0.12 q0.04/s0.65   ← IDENTICAL to CALIB_OFF
```
ヤマト (real q=0.71, s16=0.03): FULL s16 0.37→0.48→0.58→0.71 (gentler; retains q=0.20 at g3.0).

**Attribution (Δs16 = s16@g3.0 − s16@g1.0, mean over songs):**
| arm | Δs16 | Δquarter | reading |
|---|---|---|---|
| FULL | +0.595 | −0.379 | the full flip |
| GOV_OFF | +0.598 | −0.381 | **≡ FULL → governor owns 0%** |
| CALIB_OFF | +0.419 | −0.323 | raw CFG+chaos alone = **70% of the flip** |
| BOTH_OFF | +0.419 | −0.322 | ≡ CALIB_OFF → governor still 0% |

Calib contribution = FULL − CALIB_OFF = **+0.176 ≈ 30%**, concentrated at high guidance (finishes Deja 0.65→1.00).

**Density is pinned ~0.40 across ALL guidance & arms** → the flip is a **phase REALLOCATION of fixed note-mass**
off the quarter grid onto 16th-offbeats, NOT a density flood. (0.40 = the manifold's target density for this spec;
tau holds it constant, so only WHERE notes land moves.)

## Mechanism
1. **Chaos conditioning + CFG amplification (dominant, ~70%).** The chaos target's learned onset effect is "move
   mass off the quarter grid" (H4). CFG blends `uncond + g·(cond−uncond)`; at g=1.0 the raw conditioning barely
   flips (CALIB_OFF s16≈0.00), but g>1 amplifies the off-beat delta until quarters lose — steeply for Deja
   (near-complete by g2.0), gently for ヤマト.
2. **16th-unlock calib (~30%, compounding).** The fixed +logit on 16th frames matters more as CFG suppresses the
   quarters: more 16th frames clear the density-quantile `tau`. Song-dependent (adds more on busier ヤマト).
3. **Governor: none.** It gates density/workload via `tau`, not phase; with density already reallocated it has no
   quarters left to prune. FULL ≡ GOV_OFF to the digit.

## Connections
- **Refutes** the governor hypothesis for this phenomenon (the durable, validated result). This is the **H4/H14
  chaos mechanism** (see `h4_offbeat_signal_findings.md`, `h14_guidance_sweep_findings.md`) resurfacing at the
  decode extreme — chaos as a global off-grid shift, amplified by CFG past where the on-beat structure survives.
- **The GOOD signal is staying inside the real high-chaos envelope — measured as on-grid share + 16th-anchoring**
  (real Q4 ~0.85 / ~0.73; the generated smear →~0.00), NOT the `quarter-share`/`quarter-rep` used in this note's
  body (those either conflate 16th-dilution with backbone-loss, or — with a ±1 window — miscount a 1/16-offset
  spine as coverage; both were misleading, settled only by dumping the actual note grid, Rule 8). The user's
  "backbone retained" ⇔ real-envelope membership. See `probe_real_phase_reference.py`.
- **The failure mode, seen in the grid:** at `chaos=0.9,g=3.0` Deja loin vacates the downbeat AND 8th and puts
  every note on a **1/16-offset grid** — a *regular but phase-shifted spine* (`_x.x` repeating), not a random
  smear and not a density flood. on_grid 0.00, anchoring 0.00 → far off the real manifold.
- **Song-dependent tolerance = the good-region boundary (the [[goodregion]] thread).** ヤマト retains more on-beat
  structure than Deja loin at the same crank. Mapping tolerance = f(song features), scored by *distance from the
  real envelope* (not an arbitrary threshold), is `probe_backbone_tolerance.py` (real-anchored rerun 2026-07-03).
- **Graded critic:** its 0.018 at this corner is the documented **OOD-flooring** (H14: "floored on any off-the-song
  forced style"), not a realism-vs-taste gap. Pending `taste_grid` ratings still referee what "good" means by ear.

## Tooling
`probe_backbone_phase.py` (ablation ladder + phase metric; reuses `decode_harness`, `radar_manifold`,
`probe_quality_features` song loaders). Data `cache/backbone_phase.csv`. Real reference:
`probe_real_phase_reference.py`. Real-anchored tolerance sweep: `probe_backbone_tolerance.py`.
