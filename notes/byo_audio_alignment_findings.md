# BYO-audio alignment findings (`scripts/generate.py`, 2026-07-07/08)

The single-song bring-your-own-audio CLI (`scripts/generate.py`) had several charts read "sus" by ear. Root-caused
to THREE independent failure modes; two fixed, one left as a documented limitation. (All probes ran in scratchpad —
gitignored — so this file is the durable record. Deliverable set: `~/sm-generated/v2_personal_hard`, 34 v2 Hard
charts for `~/sm-personal`.)

## Failure mode 1 — BPM octave / 2:3 metric error (FIXED: require `--bpm`)
`generate.py` grids the whole chart from BPM (`hop = SR·60/(BPM·subdiv)`, and the model's `metric_phase` channel
assumes each frame IS that beat-fraction). When `--bpm` is omitted it estimates via `librosa.beat.tempo`, which is
**unreliable**: on the 34-song set, **10/26 gradable songs mis-estimated** (graded against the source `.sm` `#BPMS`
as an oracle — the tool stays audio-only). Signature: fast hardcore (165–175 BPM) came out ~112–117 = a **2:3 metric
error** (true·⅔, pulled toward librosa's log-normal ~120 prior; tell-tale: 7 songs collapsed to *exactly* 112.3,
others to 129.2 = the prior showing through). A wrong BPM = whole chart mis-gridded → "sus."

- **Offset was a RED HERRING** (attribution correction #1). `generate.py` hardcodes `#OFFSET:0.0`, but it extracts
  features and writes playback from the SAME `t=0` anchor + hop, so a note at frame `f` plays at `f·hop/SR` = exactly
  the audio time the model saw the onset → **notes stay audio-synced regardless of offset** (offset only shifts where
  measure LINES fall, cosmetic). **BPM is the alignment lever, not offset.** Corollary: a *small* BPM error (≤~3%) is
  also cosmetic (mild `metric_phase` OOD + measure lines); only gross errors (the 2:3, ~65% ratio) are catastrophic.
- **Audio-only auto-corrector: TRIED + REJECTED** (attribution correction #2 — don't ship a net-negative fix).
  Scored metric-alternate tempos (`×{⅓,½,⅔,¾,1,1.5,2,3}`) by beat-phase concentration. (a) Herfindahl peakiness on a
  24-bin histogram = BROKEN (discretization artifact biased toward the highest BPM). (b) DFT-harmonic fit (bins
  2,3,4,6 / DC on a 12-bin histogram, the `strong_readings` metric) = **net-negative: fixed 6/10 but BROKE 7 good
  songs** (rated the wrong 3:2 tempo higher on several ~124-BPM songs). The tempo **octave/metric ambiguity** isn't
  reliably resolvable by a quick metric — a real corrector is a tempogram/beat-tracking mini-project. So a corrector
  that fixes 6 and breaks 7 is WORSE than nothing.
- **Fix (user decision "just require --bpm"):** `estimate_bpm` now prints a loud warning when `--bpm` is omitted;
  `--bpm` documented as strongly recommended (`3d5639d`). No change when passed. The 8 grossly-off (>3%) personal
  charts were re-genned with true BPM; the 2 borderline (~2%: Beam Me Up 2.4%, Crazy Maybe 2.1%) left.

## Failure mode 2 — variable BPM / stops (LEFT AS-IS: unsupported, documented)
`generate.py` assumes ONE constant BPM, so a song whose source has a changing `#BPMS` or a `#STOPS` desyncs
**regardless of `--bpm`**. Detected 2 in the set (attribution correction #3 — my first BPM check read only the FIRST
`#BPMS` value and rubber-stamped these "ok"):
- **Heroes** — 13 distinct BPMs. **Stereo Sayan** — 2 BPMs + 1 stop.
- Neither is truncated, so they weren't in the re-gen. User decision: leave as-is (single-BPM tool; can't fix from
  audio alone). A future nicety: when `--inherit_from` has a source chart, read its `#BPMS` and WARN if variable.

## Failure mode 3 — TRUNCATION → the PE-extend fix (FIXED)
The model's absolute sinusoidal positional encoding was a HARD length cap: songs longer than the trained window
(v2 = `V2_MSL` 5400 frames = 458 beats ≈ 155–215s depending on BPM) were **truncated to silence**. This clipped
**23/34 charts**, some to <half (Lick the Rainbow 210s of 420s; Bye Bye 156s of 364s). Counterintuitively **worse on
v2 than v1**: v2's 48th grid packs 5504 frames into only 458 beats vs v1's 16th grid at 512 beats — the finer grid
buys triplet resolution at the cost of song COVERAGE, and higher-BPM songs pack more beats into the same time.

- **"Longer = harder for the model" is really "the chart just STOPS"** (not quality decay). Density-over-thirds on
  the charted portion mostly RISES (the difficulty-arc / breathe governor), not degrades; only a couple collapse late.
- **Scoping probe — does the model extrapolate past its trained context, or degenerate?** Built the model, then
  swapped in a longer sinusoid PE buffer (fresh; byte-identical over 0–5504 — the checkpoint stores `pe` so a bigger
  BUILD size-mismatches on load, hence load-then-swap) and generated Lick the Rainbow FULL (10752 frames ≈ **2×
  context**). Result: **graceful extrapolation** — panel-usage entropy stayed ~1.0 (full 4-arrow use, no collapse to
  a jack/dead zone) all the way to 410s; density only thinned ~28% (0.246 in-context → 0.179 OOD, recovering to
  0.25–0.27 at the very end; partly the stamina governor doing its job over a long song). Why it holds: the decoder
  leans on **audio cross-attention** (content-based, per-frame) + the periodic `metric_phase` (`t%12`, valid at any
  `t`) far more than on absolute position.
- **Fix (`75cffaf`):** `generate.py` now EXTENDS the PE to cover the whole song (load at checkpoint size, then swap
  in `PositionalEncoding(d, max_len=T+128)`), instead of truncating. `--max_len` default → None (= whole song; an
  optional user truncation cap), with a hard `SAFETY_CAP=24000` frames against OOM/extreme extrapolation. Verified:
  Toulouse 210s → full **251s** (134 measures, 48-row grid intact). **Strictly dominates truncation** (coherent full
  coverage vs a dead half). A re-indexed **sliding window** (positions always in-distribution → erases even the ~28%
  late thinning) is the possible future refinement; not built (cheap fix captured most of the value).

## Status at write time (VERIFY — transient)
- Re-gen of the 23 truncated charts (full-length, PE-extend, preserving BPM corrections) was RUNNING at last check
  (`~7/23`); verify completion by the presence of the `DONE` marker / by re-running the inspection. StepMania cache
  cleared (`~/.stepmania-5.1/Cache`) — launch StepMania only AFTER the re-gen finishes for one clean rescan.
- Deployed model UNCHANGED (still v1 `gen_motif_full_fixed`); these are v2 CLI charts, not the deploy-swap.

Related: [[byo-audio-bpm-footgun]] memory, `notes/HANDOFF.md`, `generation-defaults §0`. Probes were scratchpad-only
(check_bpm_align / validate_bpm_v2 / inspect_charts / probe_extrapolation).
