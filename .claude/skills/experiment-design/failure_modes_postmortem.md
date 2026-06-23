# Failure-mode post-mortem — chaos / 16th-placement session (2026-06-22)

A catalog of how experiments went wrong across one long session (~24 probes + 4 retrains v4–v7), and the
concrete check that prevents each. Read it as the "what bites in practice" companion to SKILL.md. The
failures generalize; the examples are this session's.

## The meta-pattern (the two roots almost every failure traces to)
1. **Coarse / global / aggregate default framing.** The instinct reached for GLOBAL stats, SINGLE metrics,
   POOLED populations, and AUDIO-only / per-song views. Reality was LOCAL / SEQUENCE / STRATIFIED / per-frame.
   The user had to redirect this ≥4 times (audio→note-sequence; per-song→per-frame; global-quota→emergent;
   pooled-difficulties→bucketed). **When you pick a measurement or a control, ask: is the phenomenon actually
   at this granularity, or am I defaulting to the coarsest one?**
2. **Premature conclusions from rigged tests.** Model-defect / ceiling conclusions were COMMITTED and then
   OVERTURNED by the very next (fair) probe — twice. The fair test always existed; the only error was running
   it second instead of first. **Every overturned conclusion was caught by a cheap test we eventually ran.**

## Catalog: failure mode → session example → betterment

### 1. The metric was BLIND to the target
- `val_total` (88% pattern loss) couldn't see rare 16ths. `s16err` watched only 16th-share (missed the 8th
  collapse) AND averaged over mostly-trivial low-chaos songs. Per-song phase-L1 was too coarse — v7 *passed*
  it yet PLAYED awkward. The taste critic was confounded by the v4 panel fingerprint.
- **Betterment:** before trusting a metric, write the sentence "this could pass while the real goal fails
  IF ___." Decompose aggregates (a share moves because the denominator grew). Spot-check the metric against
  the actual artifact / play-feel on ≥3 examples. If the target is a rare subpopulation, the metric must
  ISOLATE it and be WEIGHTED to it.

### 2. The metric / test didn't match DEPLOYMENT
- `val_f16` used a GLOBAL threshold while generation thresholds PER-SONG → it selected a checkpoint that
  placed *fewer* 16ths in real decode. The refinement de-risk used synthetic corruption that *kept half the
  real 16ths* in context → optimistic (0.865) vs the real audio-only seed (0.666).
- **Betterment:** the metric's thresholding / sampling / INPUT distribution must mirror generation. If you
  de-risk on a proxy input, run a TRANSFER gate (does the proxy ≈ the real input?) before believing it.

### 3. The setup was RIGGED — OOD, wrong-thing-fixed, or wrong VARIABLE
- Pinned density while raising chaos (real charts never do; chaos↔density corr +0.63) → forced backbone
  collapse. Cranked chaos with the density dims at the dataset MEAN → out-of-distribution combo → collapse.
  The architecture de-risk varied AUDIO context, NOT the hypothesized NOTE-SEQUENCE context → produced a fake
  "audio-ambiguity ceiling."
- **Betterment:** (a) hold fixed only what the real generating process holds fixed — CHECK pairwise
  correlations of the variables you set; (b) keep conditioning IN-DISTRIBUTION and internally coherent;
  (c) VERIFY the probe varies the *hypothesized* variable, not a correlate of it.

### 4. Premature MODEL-DEFECT attribution
- Committed "the model has no protected backbone" (overturned: backbone survival 0.83 under coherent
  conditioning) and "16th placement is audio-ambiguity-bound" (overturned: note-sequence AUC 0.935). Both
  from rigged tests (#2/#3).
- **Betterment:** default attribution order **HARNESS → DATA → MODEL**. Treat "I'm about to conclude the
  model can't do X" as a TRIGGER to run one more fair (in-dist, deployment-matched, variable-isolated) test
  FIRST. Phrase notes/commits as "under condition X, observed Y," never "the model can't Z," until cleared.
  Overturning a committed model-defect conclusion is the tell you skipped this.

### 5. CONFOUNDED comparison (didn't isolate the variable; no dynamic range)
- Staged-vs-v4 critic eval: both used v4 PANELS, so the critic (which fingerprints panel style) crushed both
  to ~0.28 — the *placement* difference under test was unmeasurable. The shuf16 baseline was uninformative
  (no chaos filter → few 16ths to break).
- **Betterment:** change ONE thing, hold all else IDENTICAL, and verify the metric has DYNAMIC RANGE on the
  varied thing (if A and B crush to the same value, the metric is blind to it — pick another).

### 6. POOLED heterogeneous populations
- A single global confidence threshold calibrated across ALL difficulties → wrong for Hard (expert charts
  confounded by beginner charts). `s16err` averaged over a population dominated by trivial songs.
- **Betterment:** STRATIFY / bucket by the relevant covariate (difficulty, rating/foot, chaos) before
  calibrating or averaging. Ask "are these units comparable?" before pooling them into one number.

### 7. The GLOBAL-QUOTA anti-pattern (project finding H12)
- `onset_phase_alloc` (flat 16th share), fixed-density chaos, and the oracle per-phase budget — every imposed
  global COUNT damaged LOCAL pattern coherence (and forced notes into unplayable hold spots).
- **Betterment:** prefer LOCAL / EMERGENT mechanisms (per-frame confidence thresholds) over imposed global
  counts. A control that forces a per-unit AMOUNT is a coherence/feel risk — flag it and prefer letting the
  amount emerge (calibrate a threshold, not a count).

### 8. Standing INFRA not carried forward
- A bespoke export path re-implemented generation and silently DROPPED a mandatory pad-playability constraint
  (`no_cross_during_hold`) → unplayable streams-during-hold → confounded a user playtest.
- **Betterment:** standing constraints live in a SHARED, CODE-ENFORCED layer (e.g. `enforce_playability`);
  never re-implement a generation/eval/export path ad-hoc. New paths route through the shared helper;
  deviations require explicit user approval. If you find yourself copy-pasting a `generate()` call, stop.

### 9. Operational / tooling traps (burned time, not findings — but they confounded monitoring)
- `pkill -f <pattern>` matched its OWN launching shell (the command line contained the pattern) → exit 144,
  twice. A `python` heredoc ran before `conda activate` → "python: not found," twice. Block-buffered stdout
  hid progress (couldn't tell a hung run from a slow one — the user caught a "stuck" run by GPU utilization).
  `chroma_*` segfaulted (numba env instability). Recurring tuple-unpacking bugs (`NameError 'a'`,
  `bitwise_and` on a float array).
- **Betterment:** never `pkill -f` a pattern that matches your own command — track and kill saved PIDs.
  Activate the env BEFORE any `python` in a compound command. Run with `python -u` (+ `flush=True`) so
  progress is monitorable. Isolate a segfault with a 1-file repro before assuming a logic bug. Re-read every
  comprehension's unpacking against what the body uses.

## The single highest-leverage habit
Before you BELIEVE or COMMIT any result, ask **"what would make this conclusion wrong?"** and run THAT test
first. Cheap probes are the right tool — but a cheap probe on a rigged setup gives a *confidently wrong*
answer fast (every overturned conclusion here was a cheap probe whose setup hadn't been cleared).

## A 30-second pre-commit gate (run before writing a conclusion in a note/commit)
1. What is this metric BLIND to? (decompose; spot-check vs the artifact)
2. Does it match DEPLOYMENT (threshold/sampling/input/population)?
3. Did I vary the HYPOTHESIZED variable, holding the rest in-distribution & identical?
4. Are the units comparable (stratified, not pooled)?
5. If this blames the MODEL — have I run the fair test that could exonerate it?
6. Did I route generation through the shared enforced infra (no ad-hoc paths)?
