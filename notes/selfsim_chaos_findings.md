# Self-similarity probe for section-level chaos — REFUTED; chaos is a conditioning problem, not a feature one

Question (from the [[phase_aware_threshold_findings]] ceiling): the model is strong frame-locally (16th AUC
0.742) but only ~0.4–0.46 correlated at the SONG/section level — it doesn't know WHICH sections deserve
chaos. Hypothesis: a global-structure (self-similarity / Foote novelty) feature would raise that, since
frame-local features can't see section identity ([[h5]]).

Cheap offline feature-informativeness probe, NO retrain (experiments/generation_typed/diag_selfsim_chaos.py).
Sliding windows across 60 songs (1061 windows, win=96 / ~6 measures); target = window's real 16th density.
Does each candidate predict it, and add R² over the model's current signal?

```
  feature          Spearman(.,real16)        incremental R^2 (target = real 16th density)
  model_p16           0.245                   model_p16                  0.059
  busy_local          0.191                   model_p16 + busy_wide      0.059   (wider context: +0)
  busy_wide           0.205                   model_p16 + ssm_homog      0.066   (self-sim: ~+0, noise)
  ssm_homog           0.051                   all features               0.066
  ssm_distinct       -0.046
```

## Findings
1. **Self-similarity is a dead end for chaos.** ssm_homog/distinct correlate ~0 with section 16th density
   and add ~0 R². Section repetition/identity (what SSM captures) does not predict 16th content. Don't
   build the feature; don't add a Foote-novelty input for chaos.
2. **Wider receptive field won't help either.** busy_wide (3× context) ≈ busy_local ≈ no incremental R².
   Not a "encoder can't see far enough" problem.
3. **Section 16th density is barely audio-predictable — R² ≈ 0.06.** The best signal (model's own) explains
   ~6% of window-level variance; nothing beats it. Yet frame placement is strong (AUC 0.742). So WHERE a
   16th goes is audio-driven (model nails it); HOW MANY a section gets is mostly charter style/artistic
   choice, weakly determined by audio. (model_p16 0.245 barely beats raw busyness 0.191 -> the model is
   already near the raw-audio ceiling.)

## Implication (loops back to the project thesis)
Chaos cannot be reliably INFERRED from audio, so the lever is CONDITIONING (specify chaos), not better
audio features. The ~0.46 song-level "ceiling" is close to the audio-information limit, NOT a defect to fix
with a self-similarity feature or wider encoder — this probe spent ~10 min proving both would fail before
any retrain. This is exactly "mastering chaos conditioning" (the standing thesis):
- decode-time: `onset_phase_calib` already gives variable per-song chaos (see [[phase_aware_threshold_findings]]).
- train-time: the chaos-radar conditioning input, and/or a PER-SECTION chaos target (the buffered-sectional
  idea), so the AMOUNT of chaos is dialed in rather than guessed from audio.

## Next (recommended)
Stop trying to make the model infer chaos from audio. Test/strengthen train-time chaos CONDITIONING: does
the chaos-radar input actually move section/song 16th density up and down (it smeared in H4/H6 pre-high-res;
re-test now with the high-res feature engaged), and is per-section conditioning the route to local control?
Cheap-confirmation-first as always. See [[chaos_mechanism_plan]], [[playtest_log]].
