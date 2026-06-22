# Phase-aware onset threshold — unlocking the model's own 16ths (chaos)

Context: chaos radar = 0 because the generator places ~no 16ths. Localized (diag_no16ths) to: the model's
16th onset confidence exists (p_on@16th ~0.4 with the high-res feature) but a single density-quantile
threshold buries it — 16th frames are out-budgeted by the more-confident 8ths (~0.55), so only 0.4–1.3%
of placed notes land on 16ths (real ~4–5%). See [[chaos_mechanism_plan]].

## The early-stop detour (and what it taught us)

User asked the right question: v4 "early stopped at epoch 4" but the BEST checkpoint was epoch 1 — best
at epoch 1 is suspicious when you're trying to learn a new representation.

- **`val_total` is blind to this experiment.** It's ~88% pattern loss; the onset head we change is ~7.5%,
  and even val_onset is dominated by abundant quarter/8th/empty frames, so the rare 16ths we optimize
  barely move it. It was flat (1.264→1.271) across the whole run while 16th behavior changed underneath.
- v5 added a **target-aligned selection metric** (`val_f16`: precision+recall at 16th-phase frames, at the
  val-set operating threshold). On THAT metric, epoch 4 beat epoch 1 (0.253 vs 0.173, +46%). So
  early-stopping on aggregate loss DID stop short of the per-metric optimum.
- **BUT `val_f16` itself was a flawed proxy** — I computed it with a *global* threshold; real generation
  uses a *per-song* threshold. They diverge. Under real decode, the val_f16-selected checkpoint (v5,
  epoch 4) actually placed FEWER 16ths than v4 (0.4% vs 0.8%) with a lower 16th posterior (0.381 vs
  0.415). More training sharpened the abundant 8ths and pushed 16ths FURTHER below the cut.

**Lessons:** (1) when optimizing a rare subpopulation, early-stop/select on a metric aligned with that
subpopulation, not aggregate loss; (2) and make sure that metric's *thresholding matches decode* or it
becomes its own artifact. `val_f16` with a per-song threshold may still be worth it — parked, not killed.
Conclusion that mattered: **training longer is not the lever; the decode threshold is.**

## The fix: phase-stratified allocation (decode-time, no retrain)

Keep the SAME note budget N = (p_on > tau).sum() the single threshold gives, but split N across the three
16th-grid phase bands (quarter t%4==0, 8th t%4==2, 16th t%4 in {1,3}) by a target note distribution
(default real's 0.707/0.252/0.041), picking the top-p_on frames WITHIN each band. Each band gets its own
implicit threshold, so the model's own 16th ranking chooses which 16ths instead of losing globally to 8ths.

Implemented as `generate(onset_phase_alloc=(q,8th,16th))` (src/generation/typed_model.py) — prototyped
first via `onset_override` (no model change) in `experiments/generation_typed/diag_phase_threshold.py`,
then wired in with a unit test (`test_onset_phase_alloc`) and an exporter flag (`--onset_phase_alloc`).

### Result (diag_phase_threshold.py, 20 songs; identical decode)
```
  source / mode               quarter    8th     16th   density  crit_adj
  REAL                          70.2%   24.7%    5.1%
  v4 / global (single tau)      62.8%   35.9%    1.3%    0.270    1.000
  v4 / alloc  (phase-aware)     70.5%   25.4%    4.1%    0.267    1.000
  v5 / global                   70.1%   29.0%    0.8%    0.270    1.000
  v5 / alloc                    70.5%   25.4%    4.1%    0.267    1.000
```
16ths 1.3%→4.1% (real-matched), **same density**, crit_adj holds at 1.0. Export spot-check: Deja loin
(the original "zero 16ths" case) 0.0%→4.2% at identical note count (n=384).

### What the metrics CANNOT settle (→ playtest)
- crit_adj=1.0 only says the redistribution didn't break difficulty — NOT that 16ths land on musically
  right frames.
- `alloc` forces the target share on EVERY song, even ones with little real 16th content. Within a song it
  picks the most-confident 16th frames (good), but the flat per-song budget could smear 16ths into
  sections that shouldn't have them. Only play-feel decides musical-vs-noise.

## Playtest handoff (installed)
A/B, same model (gen_highres_v4), same 6 rich Hard songs, threshold off vs on:
- `~/sm-generated/phase16_single` — single threshold (1.3% 16ths)
- `~/sm-generated/phase16_alloc`  — phase-aware (4.1% 16ths, real-matched)

Question for the hands: do the added 16ths feel musical (right spots, real chaos) or noisy/smeared? If
musical → phase-aware threshold is the chaos knob; if smeared → the model's 16th *ranking* is the next
target (it has the confidence, but maybe not the right placement), and a per-song/section-adaptive share
(not flat 4.1%) is the follow-up. See [[playtest_log]].
