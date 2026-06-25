# Per-foot fatigue model — design (decode-time governor that GENERALIZES the jack governor)

**Date:** 2026-06-25. **Branch:** `gen/foot-fatigue` (off `gen/h19-retrain`, which has the jack governor).
**Origin:** playtest — the jack governor ([[foot_exertion_findings]]) solved unnatural jacks but DISPLACED mass
into jumps (measured: jump rate 15.7→23.2%, longest jump-dense section 11→59). User: "we need a similar
mechanism for consecutive jumps, but with more nuance." The design conversation turned it into a unified
**per-foot fatigue simulator** where jacks and jumps are the same thing seen through two feet.

## The model
Track **two feet** with state, simulate how a player would dance each candidate pattern, accumulate per-foot
exertion that DECAYS over time, and penalize patterns that fatigue a foot. A jack = "one foot stays and
re-hits"; a jump = "two feet placed"; **one model**, so there is no jack-budget vs jump-budget to launder
between (dissolves the escape valve: `jack jack jump jack` keeps the same foot's exertion climbing).

State per sequence (batched):
- `foot_panel[2]` = panel under (left, right) foot (or none). This IS the body orientation — crossed vs not.
- `E[2]` = per-foot exertion accumulators (left, right), **exponential decay** with time constant `tau`.
- `t_last[2]` = frame of each foot's last hit (for rate).

Per note event (single or jump) at frame `t`:
1. **Assign** the arrow(s) to feet (policy below) — pick the assignment that MINIMIZES added exertion given
   current foot positions. (LOCKED: assume crossovers when they're lower-exertion, NO crossover surcharge — the
   user wants more crossovers; a chart that admits an easy crossed footing should read as easy.)
2. Per foot used, with rate `r = frame_hz / gap_frames` (gap to THAT foot's last hit; frame_hz = BPM*4/60):
   - **stay & re-hit** (same panel): `+= JACK_W * r`   (a 2-footed jack at high BPM is hard; ~0 at low BPM)
   - **move** panel→panel': `+= TRAVEL_W * dist(panel,panel') * r`   (foot-speed; distance × rate)
   - LOCKED: `JACK_W > TRAVEL_W` at equal r — a stay-and-re-hit is harder than a one-panel travel (no momentum
     to ride). dist = Euclidean on the cross (L=(-1,0) R=(1,0) U=(0,1) D=(0,-1)); adjacent=√2, L↔R=2.
3. **Decay then add:** `E[f] = E[f] * exp(-(t - t_last[f]) * frame_dt / tau) + cost[f]`; `t_last[f] = t`.
   LOCKED: exponential decay (NOT a hard window). `tau` configurable, default ≈ a half-measure (so a foot
   recovers across a phrase). NOTE: a single between jumps is NOT relief — the foot still travels to it AND time
   decay is the only relief, so a hard mid-stream single correctly stays costly.
4. **Penalty:** subtract `lambda_fat * max(E_left_after, E_right_after)` (the more-fatigued foot gates) from the
   candidate pattern's logit. Escalates with run length (E accumulates) and rate; eases with spacing (decay +
   the /gap rate term) — matching every nuance the user raised.

### Why this captures the user's nuances (each maps to a mechanism)
- LR→LR is a 2-footed jack at high BPM, free at low BPM → both feet "stay & re-hit", cost = JACK_W*r (rate-scaled).
- per-foot: LD→LU / LD→LR = L jacks (stay) while R travels D→U / D→R → assignment leaves L on L (stay cost) and
  moves R (travel cost); the two feet fatigue independently.
- LD→UR relieving at 1/4 vs 1/8 → travel cost ∝ r, and decay over the wider gap → cheaper at 1/4.
- LD→DR body-turn vs foot-swap by history → min-exertion assignment given foot_panel state: when the feet sit
  for a rotation it picks the body-turn (both feet move 1 step, low total); when a prior note pinned a foot it
  picks the L→R crossover (high). Body orientation = the foot-position state; no special-casing.

## Foot-assignment policy (the crux — LOCKED)
At each note, enumerate assignments (single: 2 feet; jump: 2 ways) and pick MIN added exertion given
`foot_panel`. No crossover surcharge. This makes the simulator assume the EASIEST footing — so the penalty
reflects the chart's best-case difficulty, which is the right thing to gate on (a chart is only as hard as its
easiest valid footing).

## Decode integration (plan)
Replace the single-foot jack governor in `typed_model.generate` with this two-foot model (jack governor = the
special case). Needs `bpm` (already threaded). Per frame, for each of the ≤15 candidate patterns, compute the
prospective `max(E)` after assignment and subtract `lambda_fat * that`. Cost: ≤2 assignments × 15 patterns × B,
cheap. Keep the hard `max_jack_run=2` backstop. New params: `fatigue_penalty` (lambda_fat), `fatigue_tau`,
`jack_weight`, `travel_weight`. Validate like the jack governor: same-panel + jump-stream + per-foot run
distributions vs real, density held, decay sanity. Then an A/B playtest.

## STRATEGIC — decode-now vs learned-v2, and why building now is not wasted
The user flags this may be **v2 territory**, and that the learned geometry/symmetry ([[geometry_feasible_region]]
— pad-symmetry equivariance) might let the MODEL reason about fatigue rather than a hand-coded decode simulator.
Resolution: **build the decode simulator now; it doubles as the fatigue CRITIC/oracle for v2.**
- NOW: a controllable decode-time penalty that ships value immediately (same proven pattern as the jack governor).
- V2: train a fatigue-aware generator (equivariant to the pad's L↔R/U↔D symmetry — fatigue is symmetric under
  mirroring, so an equivariant net learns it from far less data) using THIS simulator as the training signal /
  reward / eval. The hand-coded foot-physics becomes the differentiable-ish objective the learned model distills.
- So the simulator is the BRIDGE: decode governor today, training oracle tomorrow. Not throwaway v2 scaffolding.

## RELEASE note (the user hasn't forgotten)
The CURRENT release candidate (clean-retrain `gen_motif_full_fixed` + jack governor λ=1.5, both validated) is
assessable INDEPENDENT of this. The per-foot fatigue simulator is a fast-follow / v2 headline, not a release
blocker. Recommendation: assess the release on the current state; let foot-fatigue land on its own branch/PR.

## Open / to calibrate (not blocking the build)
- `JACK_W : TRAVEL_W` ratio + `lambda_fat` (sweep like the jack λ).
- `tau` default (half vs full measure) — expose configurable.
- dist metric (Euclidean vs a hand-tuned foot-travel table for U/D vertical awkwardness).
- whether `max(E)` or `sum(E)` gates (more-fatigued foot vs total).
