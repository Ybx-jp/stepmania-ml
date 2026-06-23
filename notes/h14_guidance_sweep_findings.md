# H14 — guidance sweep: the conditioning "decision boundary" is real, but it's a CLIFF into the chaos smear

*2026-06-22. `experiments/generation_typed/diag_guidance_sweep.py`.* Playtest H14: manifold style
conditioning is coherent but too weak at g=1.5; the user wondered if there's a guidance threshold to cross.
Swept CFG guidance ∈ {1, 1.5, 2, 3, 5} × 3 styles × 3 Hard songs (Deja loin, IN BETWEEN, nightbird lost
wing). Metrics: the 4 generation proxies the radar dims summarize (density, jump, hold, off-beat-16th) +
taste-critic P(real). gen_style ckpt, base 23-dim features.

| g | style-separation | critic P(real) | glitch off16 | hold hold-rate | stream density |
|---|---|---|---|---|---|
| 1.0 | 0.075 (1.0×) | 0.044 | 0.00 | 0.09 | 0.34 |
| 1.5 | 0.102 (1.4×) | 0.006 | 0.00 | 0.15 | 0.33 |
| 2.0 | 0.094 (1.3×) | 0.011 | 0.01 | 0.14 | 0.33 |
| 3.0 | 0.116 (1.5×) | 0.008 | 0.07 | 0.13 | 0.34 |
| 5.0 | **0.496 (6.6×)** | 0.009 | **0.62** | 0.22 | 0.34 |

## Reads
- **The decision boundary is REAL and it's HIGH (between g=3 and g=5).** Separation is flat ~1.3–1.5× through
  g≤3, then JUMPS to 6.6× at g=5. So the user's intuition holds: the conditioning is roughly inert until a
  high guidance, then bites hard. g=1.5 (the playtest setting) is well below it → explains "too weak."
- **But crossing it OVERSHOOTS into the H4 chaos smear (for chaos).** glitch_tech's off-beat-16th rate goes
  0.00 → 0.07 (g=3) → **0.62 (g=5)**. Real charts are ~80–90% ON-beat (≤~0.15 off-beat); 0.62 off-beat is the
  known H4 degenerate "uniform off-beat smear," not tasteful syncopation. So g=5 buys *separation* by
  *breaking* musicality on the chaos axis — high divergence, wrong kind. **Guidance does NOT unlock glitch
  tech's vibe; it floods off-beats.** Reinforces H4/H6 (chaos is an objective/representation problem) and
  H15 (vibe needs motif vocabulary, not decode strength).
- **Axis-specific behavior:** freeze (hold_ballad) rises gracefully 0.09 → 0.22 with no smear signature →
  moderate guidance (~g=3) may safely strengthen the holds steer. stream density is correctly **pinned** to
  the manifold target (0.33–0.34, set by the threshold not CFG) — the source-free density coupling working
  as designed; stream is steered by the density TARGET, not guidance.
- **CAVEAT — the taste critic is FLOORED (~0 everywhere, even g=1).** It cannot localize the OOD cliff here
  (no dynamic range; experiment-design rule 11). Notable in itself: the audio-grounded critic scores ANY
  off-the-song forced style as low-realism (~0.04) regardless of guidance — consistent with "this audio
  didn't ask for this style." So the **off-beat rate vs real (~0.15) is the cleaner degeneracy signal**, not
  the critic. Needs an ear check to confirm g=5-glitch is the smear the number implies.

## Conclusion / next
- The conditioning IS controllable but the useful range is dim-specific: **freeze tolerates more guidance;
  chaos overshoots into smear well before it becomes tasteful** → guidance alone is the wrong lever for the
  glitch-tech "vibe" (→ H15 motif training).
- [ ] Ear-confirm: g=3 vs g=5 on one song per style (esp. glitch_tech) — is g=5 divergence the smear the
  off16=0.62 implies, and does freeze@g=3 strengthen cleanly? Small A/B set.
- [ ] Don't chase a global guidance default; if anything, a per-axis guidance (gentle for chaos, stronger
  for freeze) — but the real chaos/vibe lever is H15 (motifs), not strength.
