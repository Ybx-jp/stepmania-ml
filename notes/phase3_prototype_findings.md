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

## Staged iteration — STABLE gen, but the critic EVAL is CONFOUNDED (06-22)
`diag_maskpredict_staged.py`: staged backbone→16ths gen with oracle per-phase budget (density/phase = real
by construction), panels via v4, taste-critic verdict:
```
  REAL 0.808 | STAGED 0.320 | v4-gen 0.251 | shuf16 0.743   (30 songs, NO chaos filter)
```
- Staged onset density/phase match real exactly (oracle budget); generation is stable.
- **The critic CANNOT isolate placement here — two eval-design flaws (mine):** (1) panels filled by v4 for
  BOTH staged and v4-gen → both share v4's machine PANEL fingerprint → critic crushes both to ~0.25-0.32
  regardless of onset placement (real/shuf keep REAL panels -> 0.74-0.81; the gap is PANELS, not placement);
  (2) no chaos filter -> most songs have few 16ths -> shuf16 barely changes -> uninformative (0.74 vs the
  objective gate's clean 0.52 on chaotic-filtered songs).
- **DEEPER FINDING (Phase-3 eval problem):** evaluating placement QUALITY on GENERATED charts is unsolved —
  the taste critic confounds placement with panel style, and you can't score against ONE real chart because
  placement is a distribution ([[phase3_generative_design]] divergence: humans agree ~33%). The EVALUATION is
  as hard as the generation (= the project's eval thesis).

## Next — a placement-ISOLATING eval
- Quick: re-run staged-vs-v4 on CHAOTIC songs only (panel-controlled: both use v4 panels, so any critic gap
  IS onset placement; chaos filter so 16ths matter). Tells if staged places 16ths better than v4 at all.
- Deeper: a placement-specific metric/critic that ignores panels (e.g., critic on ONSET-only charts, or
  distributional placement scoring vs the multi-charting pool). This is a real Phase-3 sub-project.
