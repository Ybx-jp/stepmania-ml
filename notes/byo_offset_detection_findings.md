# BYO offset/timing alignment — the "deaf choreography" root cause + auto-offset detector (2026-07-08)

Follow-on to `byo_audio_alignment_findings.md`. The shipped v2 personal charts (`~/sm-generated/v2_personal_hard`)
played "randomly hard / no choreography / like the model is deaf" (user by-ear). Root-caused to **audio↔beat-grid
misalignment** that the prior BYO note dismissed as a "red herring" (it only checked PLAYBACK sync, not CHOREOGRAPHY).

## The disease: generate.py never beat-aligns the audio
- `build_stub_chart` hardcodes `offset=0.0` (`scripts/generate.py:79`) and grids the audio from `t=0` at the
  supplied BPM. The model leans on its `metric_phase` (frame-index beat-phase) channel, so if the audio is shifted
  vs the beat grid, the model choreographs to a **phantom grid** → deaf. Two independent misalignments:
  1. **BPM error.** Toulouse was charted at the librosa estimate **129.199** vs the true **128.000** (from the user's
     reference chart). +0.9% drifts the audio a **full beat every ~51s** against the grid → coherent near the start,
     garbage by the first breakdown. (Corroboration: at 129.199 the `auto_b_trip` meter detector false-fired TRIPLET
     `+0.18`; at true 128 it correctly reads duple `−0.49` — same root cause, wrong grid.)
  2. **Start offset.** frame 0 should be the first beat; generate.py starts at audio t=0.
- The `×4/subdiv` density fix I applied first was a REAL bug (v2 charts ran ~2× real-Hard density; `generate.py:272`
  now scales it, mirroring the exporter) BUT the WRONG axis — count, not placement. Lowering density just exposed the
  deafness. **Lesson: match the fix to the felt property (placement≠count).**

## The oracle: the user's reference charts (`~/sm-personal`)
**`~/sm-personal/` holds the user's OWN hand-authored charts** (`Yb's Home Cooked/`, `Hardcore Xtreme/`) with the
TRUE `#BPMS`/`#OFFSET` measured in ArrowVortex — 26 songs, offsets spanning −1.43s..+1.10s. This is both the ground
truth for BYO timing AND a regression oracle for an auto-detector. Training packs (`data/external`) are hand-authored
too → a bigger oracle. (Recorded to memory `personal-reference-charts` — it had never been written down.)

## Auto-offset detector — can we fill #OFFSET from audio? YES (~80%), given the BPM
Detect the sub-beat phase that aligns the audio to the beat grid at the (user-supplied) BPM. Validated vs the oracle
(circular error mod one beat; latency fit on a train split, reported held-out on the 24 personal songs):

| method | personal (held-out) | train | verdict |
|---|---|---|---|
| **full-band onset pulse-train + 31ms latency cal** | median **6.8ms**, 20/24 ≤40ms, **4 slips** | median 7.1ms, 79% ≤40ms, 21% slips | **WINNER** |
| kick-band pulse-train | worse | 28 slips | oracle-refuted |
| kick-band half-beat tiebreak | HARMFUL (wrecked good answers) | 30 slips | oracle-refuted |
| DFT-phase at beat freq (1/period) | 8 slips, Toulouse 26→83ms | 36 slips | oracle-refuted |
| librosa `beat_track` DP phase | — | — | SEGFAULTS in this env (numba) |

- **Result is BIMODAL:** ~80% nailed at ~7ms (a fifth of a 48th-cell, ~39ms @128); ~20% **slip a half/quarter-beat**
  (a genuine beat-vs-offbeat ambiguity onset energy can't resolve). The two modes are separable (a slip = two
  near-equal pulse peaks a half-beat apart) → a **confidence flag** can auto-fill the confident 80% and flag the rest.
- The **~31ms latency** (onset-strength flux peaks after the transient) is a fixed constant; calibration held out of
  sample (train→personal). Two "principled" upgrades (DFT, kick) both LOST to the plain method — the oracle killed
  confident-but-wrong hypotheses cheaply (experiment-design win).

## Decisions (2026-07-08)
- **Ship the full-band detector as the offset source for ALL generation** (user decision — the product is audio-only;
  reference-chart inheritance was the alternative, rejected) + a confidence flag for the ~20% ambiguous.
- **Anchoring + written #OFFSET must move together** or playback desyncs. Wrinkle: the dataset only skips POSITIVE
  offsets (`audio_features.py:203`), so negative-offset songs (Toulouse −0.281) trained anchored at t=0. Anchor-at-
  true-beat vs anchor-at-t=0 is a BY-EAR call → **A/B on Toulouse IN PROGRESS** (Arm A `toulouse_bpm128` = t=0;
  Arm B `toulouse_anchor_beat` = audio trimmed 0.281s so frame0=downbeat; both at true 128 BPM).
- BPM still must be user-supplied (estimation separately unreliable; `byo_audio_alignment_findings.md`).

## Status (verify — transient)
- `generate.py:272` density `×4/subdiv` fix committed-pending (uncommitted at write). Detector NOT yet wired into
  generate.py. Probes are scratchpad-only (`$CLAUDE_JOB_DIR/tmp/offset_v*.py`, `probe_density_quiet.py`).
- Related: [[personal-reference-charts]], [[byo-audio-bpm-footgun]], `byo_audio_alignment_findings.md`, HANDOFF.
