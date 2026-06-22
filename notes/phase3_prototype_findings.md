# Phase-3 mask-predict prototype — findings (06-22, branch `gen/phase3-prototype`)

First build of the joint generative paradigm (`diag_maskpredict_proto.py`): onset mask-and-predict (audio +
partial onset context + mask channel), random-mask training, iterative confidence-based unmasking from
all-masked. 40 songs, 10 unmask steps:
```
  teacher-forced 16th-AUC (50% real context): 0.849   (toward the 0.935 ceiling)
  source      density  run-mean   q%   8th%  16th%     (REAL 0.206 / 1.02 / 84/14/3)
  gen(mask)    0.360     1.00      68%   32%    0%
```

## Result — PARTIAL (stability solved; generation-order flaw)
- **POSITIVE: joint generation is STABLE.** run-mean 1.00 ≈ real (1.02) — NO explosion (vs AR 5.7), no
  collapse. And the model LEARNED placement (TF AUC 0.849 >> audio 0.65). Solves the two failures that killed
  AR (explosion) and refinement (needs a good seed) — joint gen is stable AND seed-free.
- **NEGATIVE: naive generation starves 16ths (0% placed) + over-produces density (0.360 vs 0.206).** Cause =
  the UNMASK ORDER: from all-masked there's no note-context, so confidence-greedy unmasking commits the EASY
  audio-confident notes (quarters/8ths) first; 16ths are CONTEXT-dependent (their 0.849 confidence only exists
  once neighbors are present), so they never become "most confident" → never committed. Bootstrap problem in
  a new form (greedy fills the backbone, never reaches the ornaments).

## Fix (next iteration) — STAGED coarse-to-fine generation
Matches WHY placement is sequence-determined (16ths need the backbone as context):
1. Phase 1: generate the BACKBONE (quarters/8ths) — high confidence from audio.
2. Phase 2: with the backbone now as context, place 16ths (their 0.849 confidence is unlocked by neighbors).
+ density-controlled unmask schedule (commit ~target-density notes per phase) to fix the over-production.
This is a generation-PROCEDURE change, not an architecture change — the model already learns placement
(0.849). Implement as a phase-conditioned / two-pass unmask order; re-measure phase dist + density, then add
panels (v4 pattern head via onset_override) + taste-critic eval (vs real 0.844, shuf16 0.524, v4-gen 0.043).

## Caveats
- Onset-only prototype; panels + critic eval pending.
- Density control + staged order are unproven to fully fix the 16th-starvation; the staged re-run is the gate.
- Current best playable stays gen_highres_v4.
