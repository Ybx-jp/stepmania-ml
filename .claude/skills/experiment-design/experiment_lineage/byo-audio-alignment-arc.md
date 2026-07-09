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

## Chapter 2 — the CHOREOGRAPHY/OFFSET overturn (2026-07-08)
**Seed:** user played the re-genned personal set (incl. Toulouse, BPM-"corrected" + length-fixed) and said "really
bad, no choreography, like the model is deaf." A quality complaint that SURVIVED the Chapter-1 fixes → the real
disease was still unfound.
7. **Believed (my first move):** the "randomly hard" is a DENSITY problem (stamina/section density, user's hypothesis).
   **Learned:** partly — found a REAL units bug (generate.py never got the exporter's `×4/subdiv` manifold-density
   correction, so v2 BYO charts ran ~2× real-Hard density; `generate.py:272` fixed). BUT the real-chart oracle showed
   Toulouse's quiet:loud RATIO already matched real (0.61 vs 0.64) — placement was fine, only the COUNT was inflated.
   **ATTRIBUTION: the density fix was the WRONG axis** (count≠placement); lowering it just EXPOSED the deafness. Match
   the fix to the felt property. (Grounded via `probe_density_quiet.py` + a 12-song real-chart reference.)
8. **Believed (from Chapter-1):** offset is a red herring, only BPM matters. **Learned — OVERTURNS CORRECTION #1:**
   that was PLAYBACK-only reasoning. The model chores on `metric_phase`, so an audio↔beat-grid shift → phantom-grid
   choreography = "deaf." TWO misalignments generate.py never handles: (a) **BPM** — Toulouse was charted at the
   librosa estimate **129.199** (the "129.2 prior"), NOT the true **128** from the user's reference chart; +0.9%
   drifts a full beat every ~51s (so "≤3% cosmetic" from Ch.1 was ALSO wrong for choreography). Corroboration: at
   129.199 the `auto_b_trip` detector false-fired TRIPLET +0.18; at 128 it correctly read duple −0.49. (b) **OFFSET** —
   `build_stub_chart` hardcodes `offset=0.0`; frame 0 should be the first beat. **The "red herring" call was a
   playback-sync argument mistaken for a choreography one.**
9. **NEW ASSET (user reminder):** `~/sm-personal` holds the user's OWN hand-authored charts (true BPM+offset) — never
   recorded (now memory [[personal-reference-charts]]). Toulouse ref = `#BPMS 128.000; #OFFSET -0.281`. Regen at true
   128 confirmed the BPM half of the disease (by-ear pending).
10. **Auto-offset detector — can we fill #OFFSET from audio?** Validated an onset-envelope beat-phase detector against
   the reference charts + training packs as a REGRESSION ORACLE. **Full-band pulse-train + a 31ms latency calibration
   WON:** ~80% of songs nailed to ~7ms (a fifth of a 48th-cell), ~20% half-beat SLIP (genuine beat/offbeat ambiguity).
   Three "principled" upgrades were **oracle-refuted**: DFT-phase-at-beat-freq (WORSE — spike train isn't sinusoidal),
   kick-band (worse), kick half-beat tiebreak (HARMFUL — wrecked good answers). `beat_track` DP segfaults here.
   **METHOD KEEPER: the oracle killed 3 confident-but-wrong hypotheses cheaply — measure, don't theorize the mechanism.**

## Verdicts / current state
- **Ch.1:** BPM require `--bpm` (shipped); variable-BPM unsupported; truncation FIXED (PE-extend). Deliverable
  `~/sm-generated/v2_personal_hard` was built BEFORE Ch.2 → still deaf (wrong BPM/offset + 2× density).
- **Ch.2:** density `×4/subdiv` fix applied (`generate.py:272`). Offset detector chosen = full-band pulse-train +
  latency cal + a confidence flag for the ~20% (user: ship it as the UNIVERSAL offset source, not reference-inheritance).
- **✅ RESOLVED (2026-07-09):** The anchoring BY-EAR GATE — Arm A `toulouse_bpm128` (frame0=t=0) vs Arm B
  `toulouse_anchor_beat` (frame0=true downbeat) — came out **Arm B WINS: anchor-to-beat via the EXTRACTION anchor**
  (skip-to-first-beat), a deliberate DEVIATION from training's negative-offset-t=0 convention. The detector is now
  **productionized (`src/data/offset_detect.py`) + WIRED into generate.py as the default:** extraction skips the
  detected within-beat phase (positive stub offset → `extract_from_chart` skip) and the `.sm` writes `#OFFSET=−phase`
  (untrimmed audio stays synced). Re-validated vs the `~/sm-personal` oracle: **median 4.6ms, 19/23 ≤40ms, Toulouse
  7.1ms** — reproduces the original validated result, so the productionized detector IS the validated variant. The
  confidence flag is WEAK (2/4 slips; the misses slip on a syncopated off-beat, not a clean half-beat rival) → the
  real safety net is the `--offset` override, not the flag. `tests/test_offset_detect.py`.
- **Ch.1's sliding-window refinement:** BUILT + BY-EAR PASSED (`toulouse_win_anchor` "decent enough to cut v1"). The
  harder decoder-side windowing stays parked (not needed).
- **Current state:** Ch.2 CLOSED for the ship. Feeds the v1.0.0 cut (regen the personal set with the wired detector;
  the deploy-swap to v2-default is DONE — see `meter-grid-arc.md`).

Primary notes (Ch.2): `notes/byo_offset_detection_findings.md`; memory [[personal-reference-charts]].

## Method keepers
- Match the frame to the question (don't diff two tools built to differ). Use an oracle (source `.sm`) to GRADE an
  audio-only estimator without making the tool depend on it. **Validate a proposed auto-fix against the oracle BEFORE
  wiring it** (6-fix/7-break killed it). Read ALL of a multi-value field (`#BPMS`), not just the first. Separate "the
  chart STOPS" (coverage) from "the chart degrades" (quality) — different fixes.

Primary notes: `notes/byo_audio_alignment_findings.md` · memory [[byo-audio-bpm-footgun]] · depends-on the
`meter-grid` arc (v2 grid = why truncation is worse) + `generation-defaults`.
