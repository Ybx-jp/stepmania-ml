# Choreography spot-check metrics — findings

*2026-06-21. `experiments/realism_critic/choreography_metrics.py`.* Geometric metrics on the ARROW patterns
(the which-arrow / choreography axis, H1) — our metrics are otherwise almost all timing. Starter battery,
validated against structure-destroying nulls (ioi-shuffle, panel-shuffle), on 40 val songs.

## (1) Space-time movement (velocity) — coupling hypothesis FAILED; weak mean signal
Pad as 2D (L,D,U,R); note frame -> centroid; per consecutive notes dist & ioi; velocity=dist/ioi.

| source | mean_vel | corr(dist,ioi) |
|---|---|---|
| REAL | 0.275 | 0.043 |
| REAL ioi-shuffled | 0.275 | 0.021 |
| REAL panel-shuffled | 0.267 | -0.015 |
| gen_stage1 | 0.239 | 0.015 |

- **`corr(dist,ioi)` is a DUD:** REAL 0.043 ≈ nulls — the "big moves get more time" ergonomic coupling
  essentially doesn't exist in DDR (crossing the pad on fast notes IS the difficulty). Drop it.
- `mean_vel`: gen 0.239 < real 0.275 (gen moves ~13% smaller/slower) — mild real signal, weak. The
  **velocity DISTRIBUTION (fast-burst fraction)** is the better cut to try next, not mean or correlation.

## (2) Panel transition matrix — HIT: generator choreography is ~RANDOM
P(next panel | cur) over single-note transitions (4x4).

| source | L↔R symmetry | KL(·‖REAL) |
|---|---|---|
| REAL | 0.937 | 0.000 |
| gen_stage1 | 0.974 | **0.024** |
| REAL panel-shuffled | 0.976 | **0.025** |

- **KL(gen‖real) 0.024 ≈ KL(shuffle‖real) 0.025: the generator's panel transitions are no closer to real
  than a random panel-shuffle.** Real has structure the model misses — it favors **opposite-panel
  back-and-forth** (L→R 0.08, R→L 0.09, D↔U 0.09); gen is uniform/random-like (its symmetry 0.974 ≈ null
  0.976; real is LESS symmetric = it has directional preference).
- First concrete read on the which-arrow axis (H1): rhythm ~right, **choreography unstructured.** No timing
  metric saw this.
- **DECODE hypothesis TESTED & REJECTED (`choreography_temp.py`, 24 songs):** transition-KL is FLAT
  across pattern_temperature 0.3–1.2 (0.060–0.071), and low temp did NOT collapse panels. So — unusually
  for this project — the transition gap is NOT a decode-temperature artifact; it's in the pattern head.
- **The concrete gap = HORIZONTAL BIAS:** gen over-uses L/R (29/29%) and under-uses D/U (20–21%); real is
  balanced (25/26/25/23). A real but MODEST H1 (which-arrow) finding, model-level not decode. (Not obviously
  tied to felt musicality — needs the meta-correlation below to know if it matters.)

## (3) BIPEDAL kinematics — HIT, and it agrees with the hands (`bipedal_metrics.py`)
The distance metric was wrong geometry (user: with two alternating feet, L,R,L 16ths is easy). Correct
model = two feet alternating, a HOLD pins one foot; the awkward signal is one foot streaming across panels
while the other is pinned (L-hold + U,D,U). Assign feet (alternation + hold-pin automaton), per-foot
consecutive moves:

| source | per_foot_vel | fast_cross% | hold_burst% | hold_moves |
|---|---|---|---|---|
| REAL | 0.130 | 1.1% | **4.0%** | 136 |
| gen_stage1 | 0.124 | 1.1% | **6.9%** | 531 |
| REAL panel-shuffled | 0.140 | 1.1% | **7.0%** | 136 |

- per_foot_vel / fast_cross = DUDS (flat; alternation makes same-foot fast crosses rare by construction).
- **hold_burst = HIT:** gen 6.9% ≈ random null 7.0%, real 4.0% → real choreography AVOIDS one-foot-stream-
  during-hold; gen produces it at random rates. And gen has ~4× more during-hold note-events (531 vs 136)
  → ~7× more awkward bursts in absolute terms.
- **★ AGREES WITH THE HANDS:** this is exactly the B4U playtest complaint ("crossovers and jacks with one
  foot during a hold", `playtest_log.md` 2026-06-20). First geometric metric shown to PREDICT a play-feel
  verdict — the meta-validation the battery needed.
- **Points to a cheap DECODE fix:** `no_jump_during_hold` blocks JUMPS during a hold but allows single-tap
  STREAMS on other panels → the free foot fast-crosses. Extend the hold-aware automaton to also suppress
  fast free-foot crosses during an open hold (and/or curb the during-hold note over-generation).

## Battery status
- KEEP: panel transition matrix (KL + symmetry) — validated, discriminative, diagnostic.
- DROP: corr(dist,ioi). REFINE: velocity -> distribution/fast-burst fraction.
- TODO: formal syncopation (LH&L), spatial motif n-grams, chart↔audio self-similarity (H5).
- META: once ≥3 discriminative metrics exist, check which correlate with the taste critic / playtest
  verdicts — that's the battery that predicts the hands.
