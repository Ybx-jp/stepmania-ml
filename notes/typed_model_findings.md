# Typed Generator — First Result (holds work, but over-generated)

*Run: 2026-06-18 (seed 42, 20 epochs). Warm-started from focal factorized checkpoint;
panel head = per-panel 5-way symbol; class-weighted CE on onset frames. Eval: 64 val songs.*

## Result

| metric | typed | binary factorized (ref) |
|---|---|---|
| onset_F1 | 0.765 | 0.748 |
| density | 0.206 (real 0.217) | ~0.20 |
| **crit_adj** | **0.531** | 0.927 |
| generated taps | 4,284 | — |
| generated holds | **17,358 heads / 18,128 tails** | 0 |
| hold head/tail orphans | 1,362 (~8%) | — |

Teacher-forced per-symbol recall: none 0.36, tap 0.46, hold-head 0.54, tail 0.96, roll n/a.

## What works

- **The model generates holds** — 17k hold-heads, ~92% properly paired with a tail (orphan
  rate ~8%). The per-panel symbol head + class weighting let the panel head learn holds
  (teacher-forced hold-head recall 0.54, tail 0.96) despite holds being ~20× rarer than taps.
- **Onset alignment preserved** (onset_F1 0.765 ≈ binary factorized 0.748) — warm-starting the
  onset branch from the focal checkpoint carried over cleanly.
- Roll symbol never fires (0 generated) — expected, no training examples.

## The flaw: hold over-generation (a calibration over-correction)

Generated tap:hold ratio is **~1:4**, but real data is **~20:1**. The model produces hold-spam,
which collapses difficulty fidelity (crit_adj 0.93 → 0.53, MAE 1.30). Root cause is the **same
lesson as the onset head**: the panel class weights (hold-head 15.5, tail 15.3 vs tap 0.77,
none 0.28 — inverse-frequency, capped at 20) up-weight the rare classes so much that greedy
argmax at generation over-picks them. Balanced *training* signal → mis-calibrated *generation*,
exactly the weighted-loss distortion seen with the onset pos_weight.

## Fix (next step)

Soften the panel-head objective so generation calibration is preserved — same move that fixed
the onset head:
1. **Inverse-sqrt class weights** (≈4.5× for holds, not 20×) instead of inverse-frequency, or
2. **Focal loss** on the per-panel CE (no explicit class weights; focuses on hard examples), or
3. **Post-hoc calibration** (per-symbol temperature) + density/ratio-matched decoding.

(1) or (2) is a one-line change + retrain; expect the tap:hold ratio to move toward ~20:1 and
crit_adj to recover toward the binary model's ~0.93 while keeping holds. Also worth: drop the
small orphan rate by a head→tail pairing pass at decode (drop unmatched heads/tails).

## Status

Typed generation is structurally working (holds appear, pair up, round-trip to playable .sm via
the typed writer) with onset quality intact. The remaining issue is purely the rare-symbol
calibration over-correction, with a clear fix. Checkpoint: `checkpoints/gen_typed/best_val.pt`.

## Fix attempt 1: focal panel loss + inv_sqrt weights + hold pairing (2026-06-18)

Replaced the heavy weighted-CE panel loss with focal loss + milder inverse-sqrt weights
(hold 3.94 vs the old 15.5), added `pair_holds()` (orphan heads→tap, orphan tails→none → always
playable). Checkpoint `checkpoints/gen_typed_focal/best_val.pt`.

| metric | weighted-CE (w=15) | focal + inv_sqrt | binary (no holds) |
|---|---|---|---|
| onset_F1 | 0.765 | 0.769 | 0.748 |
| crit_adj | 0.531 | 0.688 | 0.927 |
| tap:hold ratio | ~1:4 | ~5.8:1 | real ~20:1 |
| holds (paired) | 17358 | 1591 | 0 |

**Partial success.** Over-generation dropped (1:4 → 5.8:1), crit_adj recovered 0.53 → 0.69 — but
short of binary's 0.93, and **teacher-forced tap recall collapsed to 0.01** (none recall 0.93).
The per-panel head learned to predict "none" for almost everything; generation now leans on the
≥1-non-empty enforcement fallback to place notes.

**Root cause (deeper than weighting):** the per-panel 5-way head conflates *is this panel active*
(mostly no — ~75% of onset-frame panels are none, for single taps) with *what type* — the SAME
imbalance-conflation we solved at the frame level with the onset head. No weighting scheme fixes
a conflated objective.

## Fix attempt 2 (proposed): layered pattern + type head

Stop making the panel head decide none-vs-active per panel. Instead reuse the binary model's
proven structure:
- **pattern head** — which panels are active (the 15-way binary pattern the binary factorized
  model nailed at crit_adj 0.93; fully warm-startable from it), then
- **type head** — for each *active* panel, its type {tap, hold-head, tail, roll} (4-way, no none).

Decode: onset (frame has a step) → pattern (which panels) → per-active-panel type. This keeps the
"which panels" decision at the binary model's quality and layers types on top, instead of
re-introducing the none/active imbalance into the panel head. This is the principled fix.

## Fix attempt 2 result: layered head (SUCCESS)

`LayeredTypedChartGenerator` (onset → 15-way pattern, warm-started from the binary panel_head →
4-way type per active panel). Trained 10 epochs (iterating at 10-epoch granularity to save time).
The "which panels" pattern head removes the none/active conflation; difficulty fidelity recovers.

Decode knobs explored (the type head decides tap-vs-hold among active panels; holds are rare +
intrinsically ambiguous with taps at the head frame, teacher-forced hold-head recall ~0.1):

| config | crit_adj | tap:hold | note |
|---|---|---|---|
| no type weight, greedy | 0.766 | →0 (no holds) | greedy never picks a ~5% hold |
| inv_sqrt type weight, sampled | 0.672 | 1.7:1 | weight inflates + sampling → too many |
| **no type weight, sampled** | **0.844** | 3.5:1 | calibrated; best fidelity |

Then `type_temperature` sweep on the no-weight checkpoint (decode-only, no retrain), real 20.2:1:

| type_temp | onset_F1 | tap:hold | crit_adj |
|---|---|---|---|
| 1.0 | 0.744 | 3.3:1 | 0.750 |
| 0.5 | 0.751 | 6.2:1 | 0.797 |
| 0.35 | 0.766 | 27.5:1 | 0.781 |

**Conclusion / recommended config:** layered head, **no type-head class weighting** (calibrated focal),
**sample the type** at generation with **type_temperature ≈ 0.4** → holds at ~the real rate (≈20:1),
onset_F1 ≈ 0.76, crit_adj ≈ 0.79. Charts are playable (`pair_holds` cleans orphan heads/tails).

Journey: broken weighted-CE per-panel (crit 0.53) → flat focal (0.69) → **layered head (0.79–0.84)**.
The layered factorization (which-panels vs type) was the fix; the type sampling temperature is the
hold-rate knob. Adding holds costs ~0.1 crit_adj vs the hold-free model (0.93) — expected, since
holds are rare and hold-head/tap is ambiguous; ~0.79 with real-rate holds is the working result.

Roll never fires (no training data), as expected. Remaining polish (optional): 20-epoch train for
a sharper type head; hold-state-aware decoding to cut the raw orphan rate before pairing.
