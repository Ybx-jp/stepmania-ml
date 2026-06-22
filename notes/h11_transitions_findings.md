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
