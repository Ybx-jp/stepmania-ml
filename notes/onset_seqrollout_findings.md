# M1b — drift gate: the frozen-h onset head COLLAPSES free-run (2026-06-29)

**Thread:** seq-aware onset, fork (A) (lineage `seq-onset-arc.md`). **Probe:**
`experiments/generation_typed/probe_seqonset_rollout.py`. **Status:** controlled NEGATIVE — the teacher-forced-trained
head is NOT drift-stable; **own-output scheduled sampling is mandatory** before any `generate()` integration (the
06-22 prediction, now specific to the frozen-`h` head).

## Question
M1a (`onset_frozenh_findings.md`) showed the frozen decoder's `h` carries the full placement signal TEACHER-FORCED
(conv readout 0.892 ≡ ceiling). But teacher-forcing feeds `h` built from REAL notes. The binding question is DRIFT:
at gen time the head reads `h` built from its OWN emitted notes. Does it survive free-run, or does the
onset→note→`h`→onset loop diverge (06-22 naive AR exploded to density 0.73 vs real 0.18)?

## Setup
Train the M1a conv head (`HReadConv` on `h`, decoder FROZEN, BCE on real onsets, 800 songs). Then a step-by-step
rollout that reuses the DEPLOYED decode (`_decoder_step_cached` + `pattern_head`, the same KV-cached path as
`generate()`), onset decided by the head on a rolling `h`-window. **DRIFT = threshold transfer:** calibrate tau on
TEACHER-FORCED `h` to hit each song's real density, then apply that tau during free-run. Governors OFF, stateless
taps, greedy pattern (pure onset-from-`h` drift — density / run-length, not a playtest). 12 Hard val songs, cap 512.

## Result (MEAN over 12 songs)
| arm | density | read |
|---|---|---|
| real | 0.272 | target |
| tf_parallel @tau (M1a head on parallel teacher-forced `h`) | 0.272 | sanity — tau calibration works |
| **TF_rollout** (incremental `h` + REAL context fed back) | **0.275** | **CONTROL — ≈ real ⇒ no harness bug** |
| **FREE-run** (incremental `h` + OWN emitted notes) | **0.000** | **COLLAPSE to empty** (run 0.00 vs real 1.03) |
| seed32_after (32 REAL frames, then free-run) | 0.026 | NOT cold-start — can't sustain from own context |

## Verdict (controlled NEGATIVE)
- **CONTROL PASSED (Rule 11):** `TF_rollout` 0.275 ≈ real 0.272 → the incremental decode's `h` matches the training
  `h`; the head + tau + sign are all correct. So FREE-run collapse is a real property, NOT a harness artifact
  (experiment-design HARNESS→DATA→MODEL — harness cleared first).
- **FREE-run COLLAPSES to 0.000** on all 12 songs. This is the 06-22 exposure-bias failure, manifesting as COLLAPSE
  (not explosion): trained on teacher-forced `h` (note-context always populated with real notes), the head learned
  "onsets follow recent notes"; at free-run the early/own context is sparse → under-fire → context stays empty →
  self-fulfilling collapse. The audio IS in `h` (cross-attn) but the head leans on note-context (the easier training
  signal), so the audio anchor alone doesn't carry it.
- **NOT cold-start:** a 32-frame REAL warm-seed does NOT rescue (post-seed density 0.026; only 1/12 songs sustained).
  The head cannot SUSTAIN density from its OWN emitted context, even handed a good start → the fix is not a seed/anchor
  tweak; it's learning to read own-context.

## Consequence
M1a (representation) is necessary but NOT sufficient. The readout reads placement perfectly GIVEN good context; the
head must learn to PRODUCE good context under its own rollout. **Next = own-output scheduled sampling (M1b-3):** train
the head on `h` built from its OWN pass-k emitted notes (mix teacher-forced + own context, anneal), so it learns to
fire from the audio-in-`h` when note-context is sparse and to sustain coherent runs. Re-run THIS drift gate after;
KILL fork (A) only if scheduled sampling also fails to stabilize (then BANK + nearest-shippable, or the causal-AR
formulation). `/autotune` before the scheduled-sampling retrain.

## Caveats (Rule 9 — conditional)
- **Tap-only / greedy emission:** the rollout emits tap symbols only (real charts have hold/tail/roll) and greedy
  patterns — a minor own-context simplification. The collapse is to ZERO onsets (an onset-head behavior), not a type
  artifact, and the control fires fine, so this doesn't drive the verdict; but the scheduled-sampling retrain should
  train on the real own-output distribution (full types) anyway.
- This measures DENSITY/run-length drift (the binding stability question), not placement QUALITY or play-feel.

## Repro
`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_rollout.py
--epochs 8 --n_eval 12` (env python directly, NOT `conda run` — it buffers). Needs the M1a caches
(`cache/seqctx_frozenh_{train,val}.npz`). Pairs with `onset_frozenh_findings.md` (M1a representation).
