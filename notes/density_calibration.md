# Density Calibration — the Stage 2 generator's core weakness

*Written 2026-06-17, after Stage 2. Context: deciding what to fix next in the AR transformer.*

## The observation

Real charts sit at ~0.20 onset density (fraction of 16th-note frames carrying a step).
The Stage 2 transformer's density is controlled **only** by decoding temperature, and
no setting gives both correct density and good onset placement:

| decoding | density | onset_F1 | onset_P | onset_R | critic_adj |
|---|---|---|---|---|---|
| greedy | 0.000 | 0.000 | — | — | 0.469 |
| temp 0.7, top_k 2 | 0.308 | 0.245 | 0.21 | 0.33 | 0.984 |
| temp 1.0 | 0.536 | 0.300 | 0.21 | 0.58 | 0.562 |

Greedy places nothing; temp 1.0 places **2.7× too many** notes. The best onset_F1 (0.30)
comes drenched in spurious notes. Precision is pinned at ~0.21 regardless — when it fires
it's only onset-aligned 21% of the time. Temperature buys recall (0.33→0.58) purely by
carpet-bombing, not by getting better at *where*.

## Root cause

**1. The single 16-way softmax conflates two decisions.** Each frame predicts one of 16
panel-states in one distribution, bundling a 1-bit *onset* decision (is there a step here?)
with a 15-way *panel* decision (which arrows, given a step). These have very different base
rates and difficulties — onset is the rhythmically hard, high-value decision; panel choice
is comparatively easy given an onset. Bundled, nothing in the loss or decoder can address
"how many" separately from "which arrows."

**2. The class weighting that fixed greedy-collapse broke calibration.** Empty dominates
~80% of frames, so to stop greedy emitting all-empty we used inverse-sqrt class weights. But
class weighting doesn't recalibrate a *generative* distribution — it reshapes the gradient so
the model assigns more probability to non-empty states than their true frequency warrants. The
per-frame distribution **systematically overstates P(note)**. At greedy you still get empty
(empty is still argmax most places even inflated); the instant you sample, that inflated
note-mass floods steps everywhere. That's why there's no good middle ground — we traded
calibration for escaping collapse.

So temperature is one global knob doing two jobs: raise it to commit to notes and you
simultaneously raise entropy over panel choice *and* over spurious positions.

## Why it matters beyond aesthetics

A chart at 2.7× density is a *different difficulty*. The critic column shows it: temp 1.0
(over-dense) reads as the right difficulty only 56% adjacent vs temp 0.7's 98%. Density and
difficulty-fidelity are coupled — you currently can't have good onset recall AND correct
difficulty at once, because the only control trades them. Fixing density is what unlocks
higher-recall operating points without wrecking difficulty conditioning.

## Fixes, ranked

1. **Factorized onset-then-panel head** (attacks the root cause). Head A: binary onset
   probability per frame. Head B: 15-way panel pattern, trained/decoded only where onsets
   occur. Density becomes an explicit onset threshold you set to the target difficulty's
   empirical density, independent of panel sampling. Decouples the two bundled decisions.
2. **Focal loss** instead of class-weighted CE (lighter). Fights the empty imbalance by
   down-weighting easy/confident frames rather than uniformly inflating note probability, so
   calibration is largely preserved.
3. **Density-matched threshold decoding** (band-aid, but cheap and diagnostic). On the
   *current* model, replace temperature with a per-song onset threshold chosen to hit target
   density. Probe first: does the model "know where" once we fix "how many"?

## Honest caveat

Precision stuck at 0.21 might not be only calibration — it could reflect model capacity /
training limits (1.1M params, 20 epochs). Density calibration is necessary but maybe not
sufficient. But that's a reason to do the probe: it cleanly separates "places too many" from
"doesn't know where," which every current metric smears together.

## Experiment (this branch): the cheap probe

Goal: decide whether the failure is **calibration/decoding** (→ build the factorized head) or
**the model doesn't localize** (→ need capacity/training). On the trained Stage 2 checkpoint,
no retraining:

1. **Teacher-forced onset posteriors.** One forward pass per song; per frame compute
   `P(onset) = 1 - softmax(panel_logits)[empty]`. Report **ROC-AUC and PR-AUC** of P(onset) vs
   true onsets — the ceiling "does it know where" number, fully decoupled from density and AR
   sampling.
2. **Density-matched threshold decoding (free-running).** Pick a global onset threshold τ so
   generated density ≈ real density, then AR-decode emitting a non-empty panel (argmax of
   states 1..15) where `P(onset) > τ` else empty. Report onset_F1/P/R, density, critic — the
   honest generation number at correct density.
3. **Compare** both to the Stage 2 temperature baselines above.

Interpretation: if (2) at matched density beats temp-sampling onset_F1, density miscalibration
was the culprit and the factorized head is worth building. If onset-AUC (1) is near chance,
the model genuinely can't localize and the next move is capacity/training, not the head.

## Probe results (2026-06-17, 64 val songs, max_len 768)

| metric | value |
|---|---|
| target (real) density | 0.197 |
| onset **ROC-AUC** | **0.813** (chance 0.5) |
| onset PR-AUC | 0.469 (base rate 0.197) |

| decode (density-matched, τ=0.770) | onset_F1 | onset_P | onset_R | density |
|---|---|---|---|---|
| teacher-forced @ τ | 0.279 | **0.291** | 0.338 | 0.196 |
| free-running @ τ | **0.000** | 0.000 | 0.000 | 0.000 |
| *temp 1.0 (Stage 2)* | 0.300 | 0.210 | 0.577 | 0.536 |
| *temp 0.7 top_k 2 (Stage 2)* | 0.245 | 0.210 | 0.332 | 0.308 |

### What we learned

1. **Localization is NOT the bottleneck.** onset ROC-AUC 0.813 (PR-AUC 2.4× base rate): the
   model genuinely knows where notes belong. So capacity/training is *not* the first move.
2. **At correct density, threshold decoding has better per-note quality than temperature.**
   Teacher-forced @ τ: onset_F1 0.279 at density 0.196 with precision **0.291** — vs temp 1.0's
   0.300 F1 but at 2.7× the density and precision 0.21. Temperature's F1 was inflated by
   over-placement; honest density gives cleaner notes.
3. **The real failure is autoregressive drift (exposure bias), not just calibration.**
   Free-running @ τ **collapses to all-empty**. A fixed threshold calibrated on clean
   (teacher-forced) context can't survive self-generated context: once the model emits empty,
   its AR context becomes empty, which pushes P(onset) below τ everywhere → permanent empty.
   Temperature avoids collapse only because stochasticity keeps notes in the context — but
   that's what causes over-placement. The decode is bistable: collapse (deterministic) or flood
   (stochastic), with no stable middle.

### Revised recommendation

The factorized **onset-then-panel head** is still the fix, but the probe sharpens the design:
make the **onset head primarily audio-driven** (audio is fixed input, not self-generated), so
onset probability does not depend on the AR feedback loop that collapses it. Concretely: predict
onset per frame from the audio memory (+ position/difficulty), largely or fully independent of
previously generated step tokens; condition only the *panel* choice on AR history. That breaks
the exposure-bias loop and makes density an honest, stable threshold. Secondary: scheduled
sampling / teacher-forcing decay during training to reduce exposure bias for the panel stream.

