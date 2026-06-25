# Trill A/B — deployed vs clean-retrained model (~/sm-generated/ab_trill_*)

Two sets, **identical songs** (Deja loin, Pound the Alarm, IN BETWEEN, nightbird lost wing, 突撃=japa1),
**identical knob** (trill=+3, guidance 2 — the exact settings of the old `h15_08_motif_trill` you played).
The ONLY difference is the model:

- **`ab_trill_A_old`** = the deployed `gen_motif_full` (trained on the BUGGY representation — trill targets were
  hold-tail-inflated, sub-16th notes dropped).
- **`ab_trill_B_fixed`** = `gen_motif_full_fixed` (retrained after the two repr fixes: attacks-only figure
  detector + sticky converter that recovers sub-16th notes).

A/B the SAME song across the two folders.

## What changed (offline, so you know what to listen for)
- **Trill knob is honestly LOWER on B** (Δself +0.32 vs +0.47 at g1; equal at g3). B learned trill from honest
  targets instead of hold-tail-inflated ones. Q: does B's trill feel *better-judged*, or just *less trilly*?
- **Quality equal** (onset_F1 0.72–0.78, density matched) — neither should feel sloppier.
- **HONEST CAVEAT — your "jack streams during holds" is NOT fixed.** That was a generation behavior, separate
  from the detector bug we corrected. Measured presses-during-holds: A 3.1% vs B **5.4%** (slightly MORE on B,
  though small/noisy). So don't expect B to feel cleaner *there* — if anything watch whether B feels busier
  around holds. This is the open question the offline numbers can't call.

## The verdict to reach
Is `gen_motif_full_fixed` (B) a feel-improvement, a wash, or worse than the deployed model (A)? That decides
whether we swap it in as the default (it's offline strictly-better-or-equal, but H15 is a *feel* thesis).
Pad-playability constraints (max_jack=1, no jump/cross during hold) are ON in both — both are pad-legal.
