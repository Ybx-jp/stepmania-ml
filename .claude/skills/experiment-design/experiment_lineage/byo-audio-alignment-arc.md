# BYO-audio alignment arc (`scripts/generate.py`, 2026-07-07/08)

**Seed:** user played the shipped v2 personal charts and said "a little off" / "sus" — a by-ear signal against a
set I'd reported as good. Isolating WHY produced three failure modes and three attribution corrections.

## Hypothesis chain (believed → learned)
1. **Believed:** the two entry points might diverge → "run the val song through both scripts and diff."
   **Learned (user redirect):** wrong frame — they're *designed* to differ (exporter parses the real chart; generate.py
   is source-chart-free: stub timing, manifold density, estimated BPM). Diffing them measures the design gap, not a bug.
2. **Believed:** the misalignment is an OFFSET problem. **Learned — ATTRIBUTION CORRECTION #1:** offset is a RED
   HERRING here. generate.py extracts features and writes playback from the SAME `t=0` anchor + hop, so a note plays at
   exactly the audio time the model saw the onset → audio-synced regardless of offset (offset only moves measure
   LINES, cosmetic). **BPM is the alignment lever.** Verified by grading estimated BPM (read from each generated
   `chart.sm`'s `#BPMS`) against the source `.sm` oracle: **10/26 mis-estimated**, fast hardcore at a **2:3 metric
   error** (true·⅔, pulled to librosa's ~120 prior; 7 songs collapsed to exactly 112.3).
3. **Believed:** an audio-only concentration metric can auto-correct the BPM. **Learned — ATTRIBUTION CORRECTION #2:**
   NO. (a) Herfindahl peakiness on a 24-bin histogram = a discretization artifact (biased to the highest BPM — caught
   because the "winners" were all the top candidate at ~uniform 0.042). (b) DFT-harmonic fit = **net-negative: fixed
   6/10 but BROKE 7 good songs.** The tempo octave/metric ambiguity isn't quick-metric-resolvable → a corrector that
   fixes 6 and breaks 7 is worse than nothing. **Don't ship a net-negative fix; validate BEFORE building.** Decision:
   require `--bpm` + warn (`3d5639d`).
4. **Believed (user catch):** I didn't check Stereo Sayan (variable BPM). **Learned — ATTRIBUTION CORRECTION #3a:** my
   first BPM check read only the FIRST `#BPMS` value → rubber-stamped variable-BPM songs "ok". Heroes (13 BPMs) +
   Stereo Sayan (2+stop) are unsupported by a single-BPM tool, no `--bpm` fixes them. Left as documented limitation.
5. **Believed (user hypothesis):** "longer charts are harder for the model." **Learned — ATTRIBUTION CORRECTION #3b:**
   it's not model quality decay — the chart literally STOPS. The absolute sinusoidal PE was a hard length cap →
   **23/34 charts truncated to silence**, some to <half; WORSE on v2 (48th grid = 458 beats vs v1's 512). Density
   over the *charted* portion mostly RISES (the difficulty arc), doesn't degrade.
6. **Scoping probe (does the model extrapolate past its trained context?):** load at checkpoint PE size, swap in a
   longer sinusoid buffer (a bigger BUILD size-mismatches — the checkpoint stores `pe`), generate a 2×-context song.
   **Graceful extrapolation:** panel-usage entropy ~full to 410s (no jack/dead-zone collapse), density thins only
   ~28% (partly the stamina governor). Why: the decoder leans on audio cross-attention + periodic `metric_phase`, not
   absolute position. → **cheap fix strictly dominates truncation** (`75cffaf`, extend PE to cover the whole song).

## Verdicts / current state
- BPM: require `--bpm` (shipped). Variable-BPM: unsupported (documented). Truncation: FIXED (PE-extend, shipped).
- Deliverable: 34 v2 Hard charts (`~/sm-generated/v2_personal_hard`); the 8 gross-BPM re-genned true-BPM; the 23
  truncated re-genned full-length (verify the re-gen `DONE` marker — transient at write time).
- **Open fork:** a re-indexed **sliding-window** generator (positions always in-distribution) would erase the ~28%
  late thinning — the quality-ceiling refinement over the cheap PE-extend. Not built.

## Method keepers
- Match the frame to the question (don't diff two tools built to differ). Use an oracle (source `.sm`) to GRADE an
  audio-only estimator without making the tool depend on it. **Validate a proposed auto-fix against the oracle BEFORE
  wiring it** (6-fix/7-break killed it). Read ALL of a multi-value field (`#BPMS`), not just the first. Separate "the
  chart STOPS" (coverage) from "the chart degrades" (quality) — different fixes.

Primary notes: `notes/byo_audio_alignment_findings.md` · memory [[byo-audio-bpm-footgun]] · depends-on the
`meter-grid` arc (v2 grid = why truncation is worse) + `generation-defaults`.
