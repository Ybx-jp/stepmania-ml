---
name: experiment-design
description: >
  Discipline for designing ML experiments/diagnostics/probes and interpreting their results so a
  rigged setup is never mistaken for a model defect. Use BEFORE running a probe or choosing a
  metric, and BEFORE attributing a surprising or negative result to "the model" — especially when
  conditioning inputs, holding a variable fixed, sweeping a control knob, picking an early-stop /
  selection metric, or concluding a capability is missing. Distilled from real attribution errors in
  this project (chaos work): the cause was repeatedly the harness, not the model. Consult it
  proactively when a result is about to change direction or get committed as a conclusion.
---

# experiment-design

A checklist discipline to stop the single most expensive mistake in this project: **blaming the model
for a failure the experimental setup caused.** It happened three times in the chaos work (see Evidence).
Each time a cheap, fair re-test overturned the "it's the model" conclusion.

**Default attribution order when something fails: HARNESS → DATA → MODEL.** Suspect your own setup
(conditioning, decode, metric, fixed variables) first. Only conclude "model defect" after the inputs and
measurement are verified fair and in-distribution. Phrase findings as *"under condition X, observed Y"* —
not *"the model can't Z"* — until the setup is cleared.

## PRE-FLIGHT — before running a probe / sweep / retrain

1. **What property am I actually testing, and does my metric see it?** A summary statistic can move the
   "right" direction while the thing you care about breaks. Name the property in plain terms; confirm the
   metric is sensitive to *it*, not a correlate. Decompose aggregates (a share can shift because the
   denominator grew, not the numerator).
2. **Does the metric match deployment conditions?** Selection/eval thresholds, decode settings, and
   sampling must mirror how the model is actually run. A metric computed under different conditions than
   generation is its own artifact.
3. **Are my control inputs in-distribution and internally coherent?** If the knobs you set are correlated
   in real data, do NOT set one extreme while pinning another at the mean — that's a combination the model
   never trained on; failure there says nothing about capability. **Check pairwise correlations of the
   variables you're setting** before decorrelating them.
4. **Am I holding fixed only what the real generating process holds fixed?** Pinning a variable the real
   data lets vary forces degenerate solutions. Match the real joint structure.
5. **What does REAL data do in this regime?** Bin/condition the real dataset the same way and read off the
   reference distribution. "What does real do here?" is the cheapest, sharpest target and sanity check.
6. **Cheapest decisive version first.** Prefer no-generation / no-retrain probes (read posteriors,
   correlate features, bin real data) before expensive runs. A cheap probe can kill a direction in minutes
   — but only if its setup is fair (a rigged cheap probe gives a confidently-wrong answer fast).

## POST-RESULT — before believing or committing a conclusion

7. **If the result blames the model, run the FAIR version first.** Re-test with in-distribution, coherent
   conditioning and deployment-matched metrics. Only a failure that survives a fair test is a model defect.
8. **Ground stats in the artifact.** Spot-check the actual output (and play-feel where it exists). Metrics
   here are blind to musicality by design — a clean number is not a clean chart.
9. **Don't commit attribution prematurely.** Write notes/commits as conditional observations until the fair
   test clears the harness. Overturning a committed "model defect" conclusion is a tell you skipped step 7.
10. **State what would change the conclusion.** If a confound is still untested, say so and test it before
    escalating to expensive model work — the confound often changes *which* fix is needed.

## Evidence (this project's chaos work — why each rule exists)

- **Rule 1 (metric sees the property):** called the chaos posterior shift "specific, not a smear" because
  quarter *share* fell and 8th/16th rose — but that stat was blind to the quarter *backbone* dissolving.
  The user's ears caught what the number hid.
- **Rule 2 (deployment match):** `val_f16` (16th-phase F1) was computed with a GLOBAL threshold while
  generation uses a PER-SONG threshold; the checkpoint it selected placed *fewer* 16ths under real decode.
  (`val_total` separately was blind to rare 16ths — 88% pattern loss.)
- **Rules 3+4 (coherent, in-distribution; fix only what real fixes):** cranked the chaos knob while leaving
  density at the dataset mean (chaos↔density correlate +0.63 in real data) AND pinned density — an OOD
  request that *forced* the backbone to collapse. Concluded "the model has no protected backbone" and
  committed it. The fair `--match_radar` test (full coherent real high-chaos profile + real density)
  overturned it: backbone survival 0.83. The real defect was far narrower (16th under-commitment).
- **Rule 5 (real reference):** binning real charts by chaos revealed chaos is *defined* as an off-beat sum
  and that real charts raise chaos by ADDING density (corr +0.63) on a preserved backbone — which explained
  the artifact and gave the true rhythm-share targets.
- **Rule 6 (cheap first):** the self-similarity probe (no retrain) ruled out a feature AND a wider encoder
  for chaos in ~10 min (R² 0.06); the onset-posterior sweeps were fast and decisive.
- **Rules 7–9 (fair test before committing):** two committed "it's the model" conclusions were overturned
  by fair re-tests. Cost: wasted exports + a misleading paper trail, narrowly avoided a misdirected retrain.
