# H16 — is guidance "harmonic"? NO (monotonic backbone-dissolution); the sweet spot is a KNEE

*2026-06-23. `experiments/generation_typed/diag_harmonic_guidance.py`.* Playtest (OH WORLD glitch): g3.5
GREAT, g4 degraded (1/8-biased, structure ruined), g4.5 partially recovered → user's hypothesis that CFG
guidance is HARMONIC (discrete coherent "nodes", garbage between). Per [[experiment-design]] rule 11, tested
the claim against SAMPLING NOISE before believing it: fine g sweep (3.0–5.0 step 0.25) × 3 seeds on OH WORLD,
measuring rhythm shares + a motif-repetition proxy.

## Result (OH WORLD, glitch, 3 seeds)
| g | quarter% | 8th% | 16th% |
|---|---|---|---|
| 3.00 | 36.6 | 62.5 | 0.8 |
| 3.50 | 24.4 | 62.9 | 12.7 |
| 4.00 | 8.6 | 61.6 | 29.8 |
| 4.50 | 1.3 | 59.4 | 39.3 |
| 5.00 | 0.5 | 55.5 | 44.0 |

Seed std ~0.3 everywhere. **Monotonic, not harmonic:** quarter% collapses 37→0.5 and 16th% rises 0.8→44 as g
climbs; the **8th line is PINNED at ~60% throughout**.

## Reads
- **H16 (harmonic) NOT SUPPORTED.** No reproducible non-monotonic feature exceeds the seed std; the rhythm
  distribution is smooth and monotonic in g.
- **The real mechanism: guidance dissolves the QUARTER BACKBONE into 16ths** (same as the chaos/H4
  backbone-collapse). The "1/8 bias" the user felt at g4 isn't new — the 8th line is ALWAYS ~60%. What changes
  is the quarter anchor: 24% at g3.5 → 9% at g4 → ~0 by g5. **g3.5 is special because it's the last operating
  point with a real quarter backbone (~24%) UNDER the 16th storm** — that anchor is the "structure/climax" the
  user heard. Above it, the quarters are gone and the bare 8th line feels unstructured.
- **The sweet spot is a KNEE, not a node:** "where the quarter backbone is still ~20–25%". Song-dependent (the
  curve shifts with the audio's own rhythm) — which is why Deja loin (1/4-dominant) needed a different point
  and still felt forced (H17 song↔style fit).

## Caveats (refutation NOT over-claimed)
- The user's **g4.5 > g4** observation is genuinely NOT in the rhythm distribution (g4.5 has *less* quarter).
  Most parsimonious: sampling — one seed per g played; an unlucky g4 / lucky g4.5 draw can't be separated from
  a real node by a single playthrough (seed std here is tiny, so seed-42 ≈ mean — but the FELT quality may
  ride on structure the rhythm metric can't see).
- **The structure/symmetry proxy (1-beat motif repetition) FLOORED at ~0** at every g — it cannot see the
  "varied symmetry" the user valued (motifs that recur WITH variation → exact-window match is rare by design).
  So quarter-backbone-share is an INDIRECT structure proxy. A real run/sweep-coherence metric is still missing
  (the standing "offline metrics can't see choreography" problem).

## Implication for the program
Guidance only TRADES backbone↔16ths along a monotonic curve — it cannot ADD the motif vocabulary that makes a
style's *character*. So the vibe lever is NOT more/cleverer guidance; it's **H15 (train the pattern head on
groove-correlated motifs)**. Guidance tuning = pick the backbone-preservation knee per song/axis (cheap, do
it); vibe = H15 (the model bet). See `notes/h15_motif_handoff.md`.
