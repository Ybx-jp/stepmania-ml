# Groove-radar manifold — realizability probe (for manifold-aware conditioning)

*2026-06-22. `experiments/generation_typed/diag_radar_manifold.py`.* Foundation for the user's
manifold-aware conditioning surface (named axes = traversal dims; [[sequence_aware_onset_plan]] piece 3).

## The manifold (4606 charts; correlation measured separately)
The 5 radar dims are NOT independent knobs — they're ~rank-2:
- **Intensity cluster** stream/volt/air/chaos, pairwise r **0.71–0.92** (denser ⇒ faster, jumpier, more
  syncopated, together).
- **freeze (holds) is orthogonal**, r ~**0.3** with all — a genuine second axis.
- chaos↔stream = **0.80** → "high stream, low chaos" is the natural CONTRADICTION (the chaos OOD bug:
  cranking chaos while pinning density requested an r≈0 point real data never visits).
- Settles the open question: stream↔freeze = **0.33** (0.18 Hard) — weak POSITIVE, not inverse.

## The surface (decode-time, NO retrain)
User gives a PARTIAL/loose spec over named axes (low/mod/high = 0.15/0.50/0.85 real quantiles, per
difficulty). Then:
- **(b) conditional-fill** unspecified dims via the real Gaussian conditional `E[free | fixed] = μ_free +
  Σ_fc Σ_cc⁻¹ (fixed − μ_fixed)` → a coherent full 5-vec.
- **(c) project** the whole point onto the covariance ellipsoid (shrink along the off-manifold Mahalanobis
  ray to ≤ the real 90th-pct distance) — only bites for extreme corners.
The resulting 5-vec feeds the EXISTING `radar_proj` + CFG conditioning. Generalizes `--match_radar` (which
conditions on a whole coherent source radar = an on-manifold point) to arbitrary user targets.

## Result — Hard manifold (n=1086; mean s.36/v.32/a.21/f.31/c.14; real Mahal median 1.77, 90th-pct 3.20)
Each seed style: conditional-filled, its Mahalanobis distance (off-manifold-ness), and how populated its
neighborhood is. **NONE of the named styles are empty, and none exceed the 90th-pct → none even need
projection** (the realizable envelope is generous at the 0.85 quantile):

| style (set dims) | filled (s/v/a/f/c) | Mahal d | real Hard within Mahal<1.0 | nearest real |
|---|---|---|---|---|
| Chaos flood (chaos↑ stream↑) | .44/.38/.31/.37/.23 | **0.93** (most typical) | 22 | Top The Charts, MAXIMIZER |
| Freeze storm (freeze↑ stream↑ air↑) | .44/.38/.35/.65/.20 | 1.27 | 10 | First of the Year, Codename APRIL |
| Pure minimal (all low) | .27/.23/.05/.00/.05 | 1.33 | **111** | Boom Boom Dollar, Silent Hill |
| Hold ballad (freeze↑ stream↓ chaos↓) | .27/.26/.12/.65/.05 | 1.63 | **44** | BRIGHT STREAM, DYNAMITE RAVE |
| Glitch tech (chaos↑ air↓ stream~) | .34/.34/.05/.30/.23 | 1.95 | 8 | LOGICAL DASH, Condor |
| Power jumps (air↑ volt↑ stream~) | .34/.42/.35/.46/.19 | 2.41 | 4 | Starry HEAVEN, I Don't Like You |
| **Stream machine** (stream↑ chaos↓ air↓) | .44/.34/.05/.21/.05 | **2.63** (least typical) | 7 | Hear me now, exotic ethnic, ビビットストリーム |

## Reads
- **The "contradictory" request is realizable.** Stream machine (high-stream/low-chaos, the r=0.80 tension)
  is the FURTHEST off-manifold (d 2.63) yet still has 7 real neighbors — a real pocket of dense-but-on-beat
  charts exists. Conditional-fill softens the contradiction (high=0.85-quantile, not max), keeping it inside
  the envelope. Pushing to MAX/MIN quantiles is where projection would finally engage.
- **Mahalanobis d ranks realizability cleanly** and matches intuition: the correlated direction (Chaos
  flood) is most typical; the contradictory one (Stream machine) least. A natural UI "how unusual is this
  combo" meter.
- **Conditional-fill reveals hidden couplings the user doesn't have to know:** Power jumps auto-fills
  freeze 0.46 (jumpy Hard charts also hold), Stream machine fills freeze 0.21 (dense-simple charts hold
  less). The user steers 2-3 axes; the rest come out real.
- **We have a REAL reference chart per style** → can A/B what each target SHOULD feel like, and the model
  has in-distribution training signal for every one of them (the chaos-OOD failure mode is avoided by
  construction).

## Source-chart-free DENSITY (the key for new songs)
The eval exporter pins onset density to the SOURCE chart — fine for A/B, but a brand-new song has no chart.
Density must come from **difficulty + style**, not a source. The manifold now carries per-chart density
(frac of frames with a note) and exposes `target_density(radar, difficulty) = E[density | radar]` via the
joint [radar, density] Gaussian. So a style's stream/chaos level **derives its own density** (stream is now
a real knob, not blunted), with NO source chart. Per-difficulty densities persisted in `radar_manifold.npz`.

Source-free density per seed style (Hard; real Hard mean ~0.31), ordered by intensity — sensible:
Hold ballad 0.289 < Power jumps/Pure minimal 0.305 < Freeze storm 0.329 < Chaos flood 0.345 < Glitch tech
0.354 (chaos couples to density). **Verified end-to-end** (`--style "chaos=high,air=low,stream=mod"`, 2
songs): both got gen_dens **0.354** (the manifold target) regardless of their own chart density (0.276 /
0.306) → density is driven by difficulty+style, source chart unused for the decision. Two pipelines stay
distinct: EVAL pins to source (for A/B); PRODUCTION (new song) + the staged quota-free default are source-free.

`export_typed_samples.py --style` density priority: `--target_density` > manifold style density > source
(legacy eval). Wired, tested (`tests/test_radar_manifold.py`, 7 pass incl. density tracks stream + graceful
None).

## Next
- Generate 2–3 seed styles + their nearest real chart (A/B), playtest: does the model RENDER the on-manifold
  target with the patterns/musicality the style implies? (open Q: on-manifold target helps, but rendering
  fidelity is the separate test — the manifold makes the ASK coherent, not automatically the OUTPUT.)
  Use `--pattern_temperature 0.7` (the exporter's 1.0 default trips the H2 coherence guard).
- Integrate manifold steering into the new-default STAGED generator (its onset isn't radar-conditioned yet —
  panels only); + let density EMERGE around the manifold prior (vs the current fixed per-style value).
- Stress projection later with MAX/MIN extreme corners (where d > 3.20) to confirm the snap-back behaves.
