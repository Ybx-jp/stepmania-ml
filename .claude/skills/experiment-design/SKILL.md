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

📎 **Resource: `failure_modes_postmortem.md`** — a catalog of HOW experiments went wrong across a full
session (~24 probes), each with the concrete preventive check, plus a 30-second pre-commit gate. Read it
when designing a metric/comparison or before committing a conclusion; it's the "what bites in practice"
companion to the rules below.

📒 **Resource: [`experiment_lineage/INDEX.md`](experiment_lineage/INDEX.md)** — one **lineage file per
investigative thread** (hypothesis chain → probes → findings → pivots → current state, cross-linked to the
`notes/*_findings.md`, the skills, and the threads it corroborates). This is **Rule 0 at the ARC level**: read the
relevant lineage file BEFORE starting/continuing a thread so you don't re-derive ruled-out work or repeat a
known-invalid setup. **You are expected to MAINTAIN these — see "Experiment lineage — maintain these" below.**

🔧 **Pairs with the `conditioning-mechanics` skill — the MECHANISM half.** This skill is the discipline (don't
blame the model); conditioning-mechanics is the exact deployed math. Rules 2–4 (match deployment / in-distribution
coherent inputs) are *satisfied in practice* by replicating that skill's §1–§8 — radar via `build_target` not
mean-pin, `tau` from guided logits, the in-loop stamina onset gate, the right per-knob metric. Consult it the
moment a probe touches a generator knob, so "match deployment" means "match the documented code," not a guess.

## PRE-FLIGHT — before running a probe / sweep / retrain

0. **Has this already been done? Check the NOTES + skills FIRST.** Before designing ANY probe, search
   `notes/` (especially `*_findings.md`, `INDEX.md`, the briefs) and the `conditioning-mechanics` skill for the
   phenomenon you're about to investigate — the answer, the mechanism, or a prior characterization is often
   already written down. This is the cheapest decisive step of all (it precedes even Rule 6). Concretely:
   `grep -ri` the key terms (the behavior, the knob, the metric) across `notes/` and read any hit before
   writing code. Two failure modes this prevents: (a) **re-deriving solved work** (e.g. the long-jack tail was
   already traced to an onset-head/density condition and SHIPPED a governor fix — a probe that "discovers" it
   wastes a cycle); (b) **mis-attributing to the model a behavior the notes already explain as a harness/decode
   regime** (e.g. `onset_override` forces the pattern head OOD → inflated jacks; the notes say the model on its
   OWN onsets matches real). If a note already answers it, CITE it and stop; if a note contradicts your planned
   setup, fix the setup. Treat "I'm confident I know this" as the trigger to check, not to skip.
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

## Added 06-22 (session post-mortem — see `failure_modes_postmortem.md` for examples)

11. **Isolate the variable; confirm dynamic range.** When comparing A vs B, change ONE thing, hold all else
    IDENTICAL, and check the metric can actually MOVE on that thing — if A and B crush to the same value
    (e.g. a critic dominated by a shared panel fingerprint), the metric is blind to what you're testing.
12. **Stratify before you pool.** Don't calibrate/average over a heterogeneous population — bucket by the
    relevant covariate (difficulty, rating/foot, chaos) first. A global threshold across difficulties
    confounds expert charts with beginner ones; an average over trivial cases hides the regime that matters.
13. **Beware the GLOBAL-QUOTA anti-pattern (H12).** Imposed per-unit COUNTS (flat shares, fixed density,
    oracle budgets) consistently damaged LOCAL coherence and forced unplayable spots. Prefer LOCAL/EMERGENT
    mechanisms (per-frame confidence thresholds) — let the amount emerge; calibrate a threshold, not a count.
14. **Route through shared, code-enforced infra; never ad-hoc.** Standing constraints (e.g. pad-playability)
    live in one enforced helper. A bespoke generate()/export path that re-implements them WILL silently drop
    one and confound a playtest (it did). New paths call the shared helper; deviations need explicit approval.

## Added 06-23 (motif-gate post-mortem — see `failure_modes_postmortem.md` §10)

15. **Baseline against the capability you ALREADY have — not a strawman subset.** When the question is "does
    a NEW lever add value," the control must be the FULL set of things already conditioned/deployed, not a
    convenient slice. Baselining motifs against *density alone* (a slice of the radar we already condition on
    in full — density, jumps, AND holds) let an already-SOLVED capability (freeze→holds is a shipped radar
    knob) masquerade as a novel motif finding. **Red-flag sanity trip:** if a conclusion recommends pursuing
    something the project ALREADY does (e.g. "start with holds"), the control was wrong — stop and re-floor.
16. **Manipulate the HYPOTHESIS variable itself; don't let an auxiliary variant answer an easier, different
    question.** The hypothesis was note-PATTERN motifs (which-panel figures). I added a frame/RHYTHM variant,
    it predicted groove, and I credited *that* to "the vibe lever" — but rhythm/timing is a DIFFERENT,
    already-settled axis (H4/H16). The manipulated variable drifted off the hypothesis. If you extend a probe
    past the spec'd hypothesis, LABEL the extension and report results **against the original hypothesis**
    separately; never silently redefine the hypothesis to fit what the apparatus happened to surface. When a
    handoff/spec defines the hypothesis and its operationalization, deviating from it needs the same explicit
    flag as deviating from shared infra (Rule 14) — and a negative result must be stated against the ORIGINAL
    definition, not the substitute.

## Experiment lineage — maintain these (DIRECTIVE)
Keep a **lineage file per distinct investigative thread** under `experiment_lineage/`, indexed in
[`experiment_lineage/INDEX.md`](experiment_lineage/INDEX.md). A thread = a coherent line of inquiry (a capability,
a knob, a defect), not a single probe. This makes Rule 0 work at the arc level: the cheapest way to avoid
re-deriving ruled-out work or repeating a known-invalid setup is to read the thread's history first.

- **WHEN:** create the file when a thread produces its 2nd–3rd `notes/*_findings.md` (or the moment a thread
  PIVOTS or overturns a prior conclusion — capture the correction while it's fresh). Update it at every checkpoint
  (new finding, pivot, attribution correction, ship). Reading it is Rule 0; writing it is part of finishing a step.
- **WHAT it holds:** the hypothesis CHAIN (what we believed → what we learned), each probe + its verdict, the
  attribution corrections (the methodology wins/losses — what would have made each conclusion wrong), the current
  state, and the open fork. It THREADS the primary `notes/*_findings.md` together; it does not duplicate them.
- **CROSS-REFERENCE liberally:** link the source notes, the skills in play, the memory nodes (`[[name]]`), and —
  crucially — the OTHER lineage threads it **corroborates** or **depends on** (shared evidence, a finding in one
  arc that explains an observation in another). Make the links RECIPROCAL: when arc A cites arc B as corroborating,
  add the back-link in B. (Example: the governor arc and the onset-phrasing arc reference each other on
  "the breathe arc is the onset-side density mechanism that carries transition responsiveness.")
- **MAINTAIN THE INDEX:** every new/under-construction thread gets a row in `INDEX.md` (✅ written / 🟡 stub / ⬜),
  with its one-line hook and primary notes. The index is the map; keep it current and prominently linked here.

The seed: `experiment_lineage/onset-phrasing-calibrator-arc.md` is the worked example (it also catalogs how Rule 0,
the density-dropped-metric Rule 1 cut, and Rule 8's artifact-smell each changed an attribution this arc).

## The single highest-leverage habit
Before you believe/commit ANY result, ask **"what would make this conclusion wrong?"** and run THAT first.
Every overturned conclusion this session was caught by a cheap fair test we eventually ran — the only error
was running it second instead of first.

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

### From the fatigue/stamina governor work (Stage 2/3; cf the `conditioning-mechanics` skill §8)
- **Rule 1 (metric sees the property), governor edition:** the stamina relief is REDISTRIBUTION — the density
  MEAN barely moves, so it's blind. The effect only shows in the paired peak/rest density-window selectivity
  (~20:1), and for holds the *window* metric DILUTED it (a "hold window" is ~85% non-hold frames) until the probe
  went FRAME-level (pinned vs non-pinned-dense). Pick the metric at the resolution of the property.
- **Rule 16 + Rules 7–9 (manipulate the real hypothesis; don't commit a strawman):** a prior session invented a
  "reach/affordability veto," BUILT it on accumulated fatigue, watched it hole-punch density, and committed
  "the local layer is near-vacuous" to the design note — but the user's ACTUAL design (onset hold-aware + per-foot
  effort) was never the thing tested. The auxiliary substitute answered a different, easier question and a NEGATIVE
  result got recorded against it. Caught only by re-reading the spec against the build. Refute the hypothesis as
  stated, not a convenient proxy.
- **Rules 5+11 (right population / dynamic range):** the percussion-bias hypothesis was probed on the default
  song order (not the complaint songs) → a noisy non-answer; re-run on the ACTUAL songs (HSL/japa1) it REFUTED the
  hypothesis (p_onset tracks harmonic energy fine) and redirected the gap to the onset head. The population that
  exhibits the effect is the only fair test of it.
