# M1a-v2 — does the 48th grid LIFT the audio→placement AUC curve? (2026-07-05)

**Thread:** the intersection of the **seq-onset arc** (`experiment_lineage/seq-onset-arc.md` — the ~0.65 audio
placement cap) and the **meter-grid arc** (`experiment_lineage/meter-grid-arc.md` — the data-layer-v2 48th-grid
build). **Probe:** `experiments/generation_typed/probe_seqcontext_frozenh_v2.py` (NEW file; the v1
`probe_seqcontext_frozenh.py` + its numbers are the REFERENCE, left untouched). **Checkpoint:**
`gen_motif_v2_48th_cont/best_val.pt` (val_total 0.7435). **Status:** POSITIVE controls fired → interpretable;
REPRESENTATION only (not drift), same boundary as v1 M1a.

## Question
The seq-onset wall's floor is the AUDIO onset head's fine-placement AUC, capped ~0.65 across four v1 measurements
(0.649 / 0.656 / 0.624) vs a note-context CEILING ~0.89 — but ALL on the hard-4/4 duple-16th grid (`t%4`), which
FLOORS triplet content to the nearest 16th (the confirmed triplet tax). HYPOTHESIS (user's): part of that 0.65 cap
is a GRID ARTIFACT — audio was scored against a target that mis-quantized triplets. On the data-layer-v2 48th grid
(12/beat, `for_v2()` + `highres_v2` beat-sync features) triplets resolve exactly, so audio MAY place better. First
time the question "can audio place a TRIPLET?" is even askable (triplets weren't representable on v1).

## Setup (fair-comparison discipline — experiment-design skill)
- Reuses the v1 arm nets verbatim (`HRead`/`HReadConv`/`Probe`) → byte-identical controls; only the DATA GRID +
  the fine-AUC bands change. 800 real train (all diff) / 140 Hard val. TARGET = real onset. Fresh re-parse
  (`cache_dir=None`) → identity-safe (avoids [[dataset-cache-footgun]]); own npz caches for re-runs.
- **Bands** (canonical `phase_band_positions(12)`): `duple16 = t%12∈{3,9}` (the DIRECT analog of v1's {1,3});
  `triplet = t%12∈{4,8}` (the NEW positions the 16th grid floored); `offbeat = t%12≠0` (aggregate).
- **Grid-robust readout = the anchored BRACKET** (raw AUC is NOT comparable across grids — different base rate /
  frame population): audio reach as a FRACTION of the chance(0.5)→note-context-ceiling gap, ON THE SAME GRID.
- Precompute batch dropped to 2 (O(T²) decoder attention at the 48th-grid T=3072 OOMs the 12GB 3060 at bs=8 — a
  throughput knob, byte-identical per-song `h`).

## Result
| predictor | onset-AUC | duple16 | triplet | offbeat |
|---|---|---|---|---|
| audio | 0.955 | **0.653** | **0.505** | 0.938 |
| both_real (note-context, CEILING) | 0.979 | 0.878 | 0.930 | 0.974 |
| frozen_h (1×1) | 0.802 | 0.782 | 0.791 | 0.743 |
| **frozen_h_conv (capacity-matched)** | 0.981 | **0.891** | **0.939** | 0.977 |

**Bracket (audio reach as % of the chance→ceiling gap):**
- **duple16: 41%** (audio 0.653 / ceiling 0.878) — vs the **v1 REFERENCE 32%** (audio 0.624 / ceiling 0.892).
- **triplet: 1%** (audio 0.505 ≈ chance / ceiling 0.930).
- offbeat: flagged UNDERPOWERED (ceiling 0.974 barely clears audio 0.938 — the aggregate is dominated by the dense
  duple positions where audio is naturally strong; not the informative band). Ignore.

**Stratified by per-song triplet occupancy (median split; triplet-heavy = 31 songs, controls fire):**
| stratum | band | audio | ceiling | reach |
|---|---|---|---|---|
| triplet-heavy | duple16 | 0.675 | 0.852 | 50% |
| triplet-heavy | **triplet** | **0.480** | 0.904 | **−5% (at/below chance)** |
| duple | duple16 | 0.614 | 0.874 | 31% |
| duple | triplet | n/a (too few triplet positives) | | |

## Verdict — two findings, both consistent with the wall
1. **The finer grid LIFTS audio's reach MODESTLY on the OLD (duple-16th) positions: 32% → 41% of the gap.** So the
   0.65 cap was PARTLY a grid artifact — on a clean, beat-synced grid the audio head places duple-16ths somewhat
   better. But it is a ~9-point lift, NOT a breakthrough; placement stays dominantly a chart prior (audio < half
   the gap).
2. **The seq-onset wall EXTENDS to triplets, hard.** Audio is FLAT AT CHANCE for triplet placement (0.505; −5% in
   the triplet-heavy stratum where it's well-powered), while the note-context ceiling nails it (0.930) and the v2
   decoder's `h` encodes it fully (conv 0.939). v2 fixes the TARGET (triplets representable + placeable by the
   trained prior) without moving the SOURCE (audio still can't hear WHERE a triplet goes) — exactly the arc's
   "audio is placement-blind beyond density" result, now shown for a NEW class of positions.
3. **M1a REPLICATES on the 48th grid + extends to triplets:** `frozen_h_conv` ≡ the note-context ceiling on BOTH
   duple16 (0.891≈0.878) and triplet (0.939≈0.930); the 1×1 readout lags (capacity confound, Rule 11, as in v1).
   Build-sizing payoff: the PARKED seq-onset retrain (fork A) would have triplet placement fully available in the
   FROZEN v2 decoder's `h` — a cheap causal-conv onset-head add, now with triplets in scope.

## Honesty caveats (experiment-design Rules 2/7/10)
- **The 32%→41% duple lift co-varies TWO changes** (finer grid + beat-sync audio) on a DIFFERENT val population
  (140 vs v1's 98 Hard). The bracket-fraction controls for population better than raw AUC, but this is NOT a
  controlled A/B. **DISAMBIGUATING CONTROL (untested):** restrict to CONSTANT-BPM songs where beat-sync is a no-op
  → any residual lift is PURE grid. Until then "modest duple lift" is SUGGESTIVE, not proven. (The triplet result
  needs no caveat: audio-at-chance vs a fired ceiling is unambiguous.)
- **REPRESENTATION not DRIFT** (Rule 9): `h` is teacher-forced on REAL notes = the upper bound a frozen readout
  could see; gen-time drift is a separate gate (the v1 M1b collapse). Does NOT prove a gen-time seq-onset head works.
- Nothing here OVERTURNS the wall. It ADDS: a modest duple lift, a clean triplet EXTENSION, and v2 build-sizing.
  It does NOT overwrite any v1 number.

## Does NOT bear on Phase 6 by-ear
This is the AUDIO-head's placement reach (a representation question). The DEPLOYED v2 decode places triplets via
the trained decoder's learned prior (finding 2/3 — that's where triplet placement lives), NOT via the audio head.
So this probe is orthogonal to whether v2 removes the by-ear limp — Phase 6 remains the binding deploy gate.

## Repro
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/generation_typed/probe_seqcontext_frozenh_v2.py
--ckpt checkpoints/gen_motif_v2_48th_cont/best_val.pt` (caches `cache/seqctx_frozenh_v2_{train,val}.npz` gitignored;
present → extraction skipped). `--precompute_bs 2` default (12GB 3060). Log `probe_v2grid.log`.
