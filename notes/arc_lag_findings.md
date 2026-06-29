# Arc-lag probe — WHERE does the "lag adapting to phase changes" come from?

**Date:** 2026-06-28
**Probe:** `experiments/generation_typed/probe_arc_lag.py` → `outputs/arc_lag_envelopes.png`
**Trigger:** by-ear note on `unlock16_b20` (HSL, 2026-06-28): the chart is slow to react at the
piano-solo cold-start and slow to back off the intensity after a busy section.
**Prompted by:** my making the breathing arc default-on this session → is the new lag *my* change?

## Question / candidate sources (each makes a DIFFERENT prediction)

- **(A) breathe arc** = centered boxsmooth of `p_onset` over `stamina_breathe_win=96`f (~6 measures).
  A centered smoother is **zero-phase**: it can only blur symmetrically, it cannot create a one-sided
  lag (if anything it anticipates). Postdates H11, so H11 never tested it.
- **(B) onset head** intrinsic `p_onset` envelope (audio-only, non-causal → should not be causally late).
- **(C) AR pattern head** carrying the prior section's pattern across a boundary = causal, only-late = a
  true one-sided lag. **This is H11 (`notes/h11_transitions_findings.md`), ALREADY characterized**
  (teacher-forced boundary loss flat; free-run under-transitions via AR drift; "cold-start = t=0 special
  case"). H11's magnitude metric is blind to direction; its pooled free-run gap is noisy/song-set-dependent
  (`buffered_sectional.py`) → did NOT re-run a pooled average (Rule 11).

This probe is the cheap **no-generation, timing+direction** cut H11 lacked, on the **complaint song**
(HSL; Rule 5/11), isolating the one variable I changed (breathe). Reference = MFCC0 (audio log-energy).

## Result

| song | p_onset lags audio | breathe lags p_onset | human density lags audio |
|------|---|---|---|
| **High School Love** (complaint, bpm180) | **+1f** (+83ms) r=.90 | **+0f** (zero-phase ✓) | −1f |
| 突撃！…kneeso (contrast, bpm185) | +14f (+1135ms) r=.79 | **+0f** (zero-phase ✓) | +7f |

## Reading (pre-registered decision rule, Rule 9)

1. **Breathe is exonerated.** `breathe-vs-p_onset = +0f` on BOTH songs — the zero-phase prediction holds
   exactly. Making breathe default-on did NOT introduce a one-sided lag. A `stamina_breathe_win` knob
   would only change blur *width*, not lag. (One nuance: at the literal t=0 cold start the centered box's
   *edge padding* makes the ramp start a hair late — a small blur, not the felt 3.5-beat lag.)
2. **On HSL (the complaint), the onset head does NOT lag** (+1f, r=.90) — audio-driven intensity tracks
   the music tightly, ramping up out of the cold start and down at the outro *with* the audio energy.
3. **⇒ By elimination, HSL's felt cold-start/back-off is the AR pattern head = H11.** Already characterized,
   documented limitation, post-0.1.0 lever = `boundary_reset` (down-weight prior-note context at a detected
   boundary). No new mechanism, no retrain decision.
4. **Secondary (NOT the complaint population):** on kneeso `p_onset` lags +14f — but the *human* chart also
   lags +7f there (energy leads density by the song's own structure), so the metric carries a song-specific
   offset and the model only *over*-lags human by ~7f. The reference (loudness) is not clean onset-truth on
   this song; do not over-read the +14f. HSL is where the metric is clean and the answer is decisive.

## 0.1.0 decision

**Do not block.** H11 is a documented, scoped limitation (maps onto the README's existing "global,
whole-song phrase planning is the open frontier" honesty note — worth sharpening to name transition-lag).
The thing genuinely worth checking first — *my* breathe change — is cleared: zero-phase confirmed. Ship.

## Bugs fixed in the probe while running it (Rule 14 hygiene)

- `lag_of` assumed equal lengths; the chart note-tensor can be a few frames short of the audio → clip both
  to `min` length before corr.
- title printed empty: it lives on `m['chart'].title`, not `m['title']` (the `source_file` fallback hit a
  nonexistent attr). Mislabeled rows would have inverted the complaint-vs-contrast attribution — the whole
  reading hinges on which row is HSL.
