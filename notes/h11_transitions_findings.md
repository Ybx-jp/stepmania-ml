# H11 — transitions: the AR pattern head drifts (confirmed)

*2026-06-21. Playtest seed (`playtest_log.md`, deja loin): "does well continuing a sequence but struggles
to adapt to transitions" — awkward at the intro and the bridge/breakdown.* Mechanism pinned down with two
probes; section boundaries from a Foote-novelty detector on a timbre(MFCC)+harmony(chroma) self-similarity
matrix.

## Probe 1 — teacher-forced loss at boundaries (`diag_transitions.py`): FLAT
Teacher-forced per-frame loss around audio section-boundaries, normalized to each song's mean (30 songs):
onset ≈ 1.00, pattern ≈ 1.01 in the [0,+8] window (faint pattern bump to 1.07 at the boundary, decaying).
→ **Given the correct context, the model predicts transitions ~as well as steady-state.** Transitions are
NOT a representational deficit. (Caveat: teacher-forcing FEEDS the real prior notes, so it hides any
free-running drift by construction.)

## Probe 2 — free-running transition responsiveness (`diag_transitions_freerun.py`): GEN UNDER-TRANSITIONS
How much the choreography changes across a position (L1 of [density, jump, L,D,U,R mix], window 8 beats),
at section boundaries vs random positions. **Boundary detection matters:** L=16 over-detected (~30
"boundaries"/song = local noise) and washed the signal out (real responsiveness +0.014). With L=32 (8-beat
kernel) + top-5 most-prominent boundaries/song (real sections):

| chart | @boundary | @random | responsiveness (bnd−rand) |
|---|---|---|---|
| real | 0.820 | 0.692 | **+0.128** |
| gen | 0.775 | 0.745 | **+0.030** |

**Real charts re-choreograph at section boundaries; the generator captures only ~23% of that** — it drifts
through transitions instead of adapting.

## Conclusion: exposure-bias / AR drift in the PATTERN head
Probe 1 (flat) + Probe 2 (under-transitions 4×) ⇒ the model CAN choreograph transitions given correct
context, but in FREE-RUNNING its **autoregressive pattern (which-arrows) head has momentum** and continues
the prior section's motif across a boundary. This is the exact failure the project already fixed for the
ONSET head (made non-causal/audio-driven to kill AR-drift collapse, see `factorized_head_findings.md`) —
the pattern head is still AR and still drifts. **Cold-start (awkward intro) is the t=0 special case**: a
transition with no prior context. Unifies the long-standing cold-start thread with mid-song breakdowns.

## Fix levers (untested)
1. **Scheduled sampling for the pattern head** — train on its own sampled outputs so it learns to recover
   from / adapt after drift (the classic exposure-bias fix; targets the mechanism directly).
2. **Reduce the pattern head's AR reliance / strengthen audio-conditioning** so it follows the audio change
   at a boundary rather than its own history (architectural; analogous to the non-causal onset head).
3. **Decode-time boundary reset** (cheap probe): at a detected section boundary, down-weight the prior-note
   context so the pattern head re-derives from audio. Risk: reintroduces cold-start-style awkwardness at
   each boundary — test whether it helps net.
Recommend (3) as the cheap first probe (decode-time, no retrain), then (1) if it warrants a retrain.

## Boundary-reset probe (2026-06-21, `boundary_reset_probe.py`): DIRECTION VALIDATED
`generate(boundary_reset=<frames>)` flushes the self-attn KV-cache + resets the decoder input to BOS at
each detected boundary (drops note-history momentum; keeps audio cross-attn). Transition responsiveness:

| chart | @boundary | @random | responsiveness |
|---|---|---|---|
| real | 0.820 | 0.692 | +0.128 |
| gen (no reset) | 0.769 | 0.790 | -0.021 |
| gen_reset | 1.011 | 0.827 | **+0.185** |

Resetting flips drift (-0.021) → re-choreography (+0.185), **overshooting** real (+0.128). So the model CAN
re-choreograph at transitions; AR momentum was suppressing it. The overshoot = the HARD cold-start is too
abrupt (boundary change 1.011 vs real 0.820) — exactly what a **warmup buffer** should soften toward real.

## Buffered-sectional generation (next): the playable version
Per the user's idea: generate each section independently with discarded WARMUP (absorbs cold-start) and
COOLDOWN (absorbs H5 end-fade) buffers, keep only the clean middle, concatenate. One decode-time mechanism
that addresses cross-boundary momentum (H11) + cold-start (H11 t=0) + end-fade (H5). `generate_sectional`
in `buffered_sectional.py`. Measure responsiveness (should land nearer real than the raw reset's overshoot)
+ playtest (the raw reset isn't playable — cold-start at each boundary; the buffers fix that).

## RESULT (2026-06-21, `buffered_sectional.py`, 12 rich Hard songs) — DOESN'T pan out offline; H11 effect not robust

| chart | @boundary | @random | responsiveness | (real +0.114) |
|---|---|---|---|---|
| real | 0.755 | 0.641 | +0.114 | |
| baseline | 0.809 | 0.703 | +0.105 | ≈ real |
| sectional | 1.343 | 0.831 | +0.512 | 4.5× real (overshoot) |

Two sobering findings:
1. **The free-running under-transition effect is NOT robust — it's song-set-dependent.** On rich Hard songs
   the BASELINE already matches real (+0.105 vs +0.114); the big gap reported in Probe 2 (gen +0.030 vs
   +0.128) was on a different 30-song Medium+ set. The responsiveness metric is noisy across song sets.
   (The teacher-forced "not a representational deficit" result still stands; the free-running *gap* and the
   *fixes* are the shaky part.)
2. **Buffered-sectional OVERSHOOTS badly** (+0.512; boundary change 1.343 vs real 0.755). Generating each
   section fully INDEPENDENTLY (cold start + own density) discards ALL continuity → adjacent sections are
   unrelated → jarring discontinuity, not real-like transition. Real transitions *evolve*; this *resets*.
   The warmup buffer didn't help — cold-starting into the new audio yields a fresh unrelated choreography.

**Verdict:** offline does NOT support the sectional approach (likely feels discontinuous). The robust H11
finding is only "transitions aren't a representational deficit"; the drift gap and the reset/sectional
fixes are inconclusive (noisy metric) or counterproductive (overshoot). If pursued, the fix would need to
PRESERVE continuity across boundaries (warm-start each section with the prior section's tail, not cold),
not hard-reset — but given the effect isn't robust, low priority. Playtest (esp. deja loin, the original
observation, in the exported set) is the arbiter; temper expectations (sectional may feel jarring). Density
side-effect: per-section density targeting DOES give sections the real intensity arc (a real H5 bonus),
independent of the transition overshoot.
