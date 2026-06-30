# M1b-3 — note-dropout scheduled sampling BREAKS the free-run collapse (2026-06-29)

**Thread:** seq-aware onset, fork (A) (lineage `seq-onset-arc.md`). **Probe:**
`experiments/generation_typed/probe_seqonset_ss.py` (head saved to `cache/seqonset_ss_head.pt`). **Status:** POSITIVE
— the drift wall that killed fork (A) twice (06-22 explosion; M1b collapse) is BROKEN. The head free-runs coherently
near real density. Two corrections fell out (one to M1b, one methodological). Remaining = placement quality (by-ear /
gen-time AUC) + a steep calibration cliff.

## What was done
M1b found the teacher-forced-trained head COLLAPSES free-run. FIX = **note-dropout scheduled sampling** (cheap,
parallel): per batch drop real notes with prob `d∼U(0,1)`, decode `h` from the corrupted (sparse/empty) context,
train the head to predict the FULL real onsets. `d=0` → M1a teacher-forced (run-coherence); `d→1` → empty context →
the head is FORCED to fire from the audio-in-`h`. 10 epochs, 800 songs, decoder FROZEN (grad through the head only).
Then the M1b drift gate + an absolute-threshold SWEEP (to decouple calibration from drift).

## Result (12 Hard val; the sweep is 6 songs, cap 384)
**Drift gate with the teacher-forced-calibrated tau:** FREE-run density **0.000** (looks like collapse) — BUT this
is a TAU-TRANSFER artifact (see below). Control TF_rollout 0.277 ≈ real 0.272 (harness clean).

**Absolute-threshold sweep (free-run density / run-length; real ≈ 0.27 / 1.05):**
| tau | free_d | free_run |
|---|---|---|
| 0.70 | 0.000 | 0.00 |
| 0.62 | 0.000 | 0.00 |
| 0.58 | 0.213 | 0.67 |
| 0.56 | ≈0.27 | ~0.8 (interpolated — the real-density crossing) |
| 0.55 | 0.332 | 1.00 |
| 0.50 | 0.362–0.390 | 1.00 |
| 0.20 | 0.447 | 1.00 |
| 0.10 | 0.514 | 6.89 (runs begin) |
| 0.02 | 0.890 | 83 (explosion) |

## Two corrections
1. **M1b's "collapse" was CONFOUNDED by tau-transfer (calibration), not pure drift.** The gate calibrated tau on
   TEACHER-FORCED `h` (real, dense context → high logits) and applied it to FREE-RUN (sparse own context → lower
   logits). The teacher-forced tau sat ABOVE the free-run logit range → buried everything → 0.000. This is the same
   class as "tau from unconditioned logits" (conditioning-mechanics §3 / chaos work). The sweep is the fair test:
   at an absolute threshold the head fires. (M1b's harness was otherwise clean — the TF_rollout control held — so
   the WALL conclusion was right that SS was needed; the SEVERITY ("can't fire at all") was the artifact.)
2. **Note-dropout SS genuinely broke the collapse.** A teacher-forced-ONLY head has ~no audio-firing to un-bury
   (its free-run logits are near zero even at tau 0.5); the SS head fires coherently from its own sparse context.

## Verdict — the drift wall is BROKEN (POSITIVE)
- **Stable coherent plateau:** tau 0.2–0.55 → run-length **1.00** (isolated onsets, the real run-structure), NO
  collapse, NO explosion. Explosion only sets in below tau ~0.1 (run → 83 at 0.02). A wide stable operating band.
- **A near-real operating point EXISTS** (tau ≈ 0.56 → density ≈ 0.27, run ~0.8). The head free-runs from its OWN
  context at real density with real run-structure — the thing M1b/06-22 said was unreachable.
- **Caveat — steep calibration cliff:** the free-run logits are concentrated (tau 0.62→0.00, 0.58→0.21, 0.55→0.33),
  so a GLOBAL absolute threshold is density-sensitive. Deployment should use a per-song / self-calibrated tau (from a
  first free-run pass) or the existing density-target / stamina mechanisms — NOT the teacher-forced tau.

## Boundary / what is NOT yet shown (Rule 9)
- The gate measures DENSITY + RUN-LENGTH stability ONLY — NOT placement QUALITY. "run 1.0 at density 0.27" means the
  run-STRUCTURE matches real; whether the onsets land on the musically-right frames is the next question
  (gen-time 16th-AUC of the free-run decisions vs real, and the binding BY-EAR gate).
- Tap-only / greedy emission, governors OFF (pure onset-from-`h` drift). The eventual `generate()` integration must
  use real types + the playability/governor stack.
- dmax=1.0 SS emphasizes audio-firing → the head slightly OVER-fires (the plateau sits at 0.33–0.45 for tau 0.2–0.5);
  a balanced `d` distribution or a real-density training target may tighten it.

## NEXT
1. **Placement quality:** gen-time 16th-AUC of the free-run onsets vs real at the tau≈0.56 operating point; if it
   ranks real 16ths (toward the 0.87 teacher-forced ceiling), the win is real. 2. **Self / per-song tau** (free-run
   calibration) to tame the cliff. 3. **Wire into `generate()`** (opt-in kwarg, decode onset[t] from `h[t]` at
   `typed_model.py:635`, decoder frozen) with real types + governors; `/autotune` before any larger retrain.
   4. **By-ear** = the binding gate. Deployed model stays `gen_motif_full_fixed` until it proves out.

## Repro
`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_ss.py
--epochs 10 --dmax 1.0` (env python directly, NOT `conda run`). Re-sweep without retraining: add `--load_head
--taus 0.62,0.58,0.55,0.52`. Needs the M1a caches. Pairs with `onset_seqrollout_findings.md` (M1b),
`onset_frozenh_findings.md` (M1a).
