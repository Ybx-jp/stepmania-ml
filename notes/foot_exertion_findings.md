# Foot-exertion soft jack governor (decode-time) — graded, BPM-aware jack control

**Date:** 2026-06-25. **Origin:** user playtest of the trill A/B noted long jack streams the retrain didn't
touch (correctly — jacks are a DECODE behavior, not learned). Audit of the existing constraints found the
foot-speed cap had a **spacing blind spot**.

## The gap that was found
- `max_jack_run` (foot speed) only fires when `since_onset == 1` — i.e. **16th-adjacent** presses. It perfectly
  eliminates 16th jacks (measured: literally 0 in generated charts) but is **blind to 8th-note jacks**, which
  streamed through up to length 6–7. The user's "long jack streams" = 8th jacks. Confirmed: of generated
  jack-runs ≥3, **70/108 were 8th-spaced, 0 were 16th** (the cap had removed all 16ths).
- Foot DISTANCE: `no_cross_during_hold` is on (mandatory), but the general `no_crossovers` is off by default —
  left off ON PURPOSE (the user wants MORE crossovers/footswitches, the advanced-patterns thread).
- Real-chart reference (8th-spaced same-panel runs): len2 75.8% / len3 12.8% / len≥4 11.4%, tail to 8+.

## Design (user's spec) — a soft escalating penalty, not a hard cap
Hard per-spacing caps are the wrong abstraction: exertion is continuous in time, not bucketed by grid (a 6-note
8th jack and a 3-note 16th jack can be equally brutal at different BPMs). So:
- Accumulate **exertion** `E` per same-panel run: on each repeat, `E += (press_rate / jack_free_rate)`, where
  `press_rate = frame_hz / gap_frames` Hz and `frame_hz = BPM*4/60` (16ths/sec). E **persists across the empty
  frames** between presses and **resets** on a different-panel single or a jump (the foot gets relief).
- Penalty to EXTEND the run = `jack_penalty (λ) * (E + cost_of_this_press)`, subtracted from the jack-panel's
  single-pattern logit. Escalates with run LENGTH (E grows) and RATE (Hz). A short/justified jack (incl. a 2-note
  16th) has small E → survives if the onset is strong; a long or fast stream escalates and gets re-routed.
- One formula, BPM-aware, no 8th/16th split. `src/generation/typed_model.py::generate(jack_penalty, jack_free_rate=5,
  jack_max_gap=4, bpm)`.

## Validation (diag_jack_exertion.py, gen_motif_full_fixed, 8 songs, hard cap relaxed to 2)
```
jack_penalty  runs>=2  len2%  len3%  len>=4%  maxlen  density
   OFF          165     80.6   14.5    4.8       6     0.208
   1.5           61     93.4    4.9    1.6       4     0.208
   3.0           36     97.2    2.8    0.0       3     0.208
   5.0           15    100.0    0.0    0.0       2     0.208
```
- **Density IDENTICAL (0.208) at every λ** — the headline. The penalty re-routes long jacks to ALTERNATION, it
  does NOT delete notes (biomechanically correct: a human alternates feet instead of hammering). maxlen falls
  monotonically 6→2.
- The base model's avg long-jack rate (4.8% len≥4) is already BELOW human (11.4%) — the egregious streams were
  concentrated in the trill knob + jack-heavy songs, not everywhere. ⇒ a GENTLE default is right; over-penalizing
  makes charts LESS human than real. **Chosen default λ=1.5** (user-approved): maxlen 6→4, len≥4 4.8→1.6%.

## Shipped (user-approved 2026-06-25)
- `generate()`: `jack_penalty`, `jack_free_rate`, `jack_max_gap`, `bpm` params + the exertion accumulator + soft
  penalty. Regression test `test_jack_penalty_governs_long_runs` (never lengthens worst run, never empties,
  preserves onset count). 35/35 gen tests pass.
- **Mandatory contract change:** `MANDATORY_JACK_CAP` 1→2 (allow a justified 2-note 16th jack; 3+ still hard-
  forbidden). Updated `playtest_export.py` + the playtest skill table in lockstep.
- Exporter: `--jack_penalty` (default 1.5), `--max_jack_run` default 1→2, threads song BPM into `generate`.

## Next
- [ ] Playtest A/B (jack_penalty OFF vs 1.5) — does the governed feel read as more natural / less jacky?
- [ ] Tune λ by ear per the A/B; consider exposing jack_free_rate if the BPM normalization needs adjusting.
