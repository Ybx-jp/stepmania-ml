# Sequence-aware onset head — the justified program (Phase 2.6)

## Why (the turning point, 06-22)
16th PLACEMENT is SEQUENCE-determined, not audio-ambiguous. `diag_seqcontext_probe.py`: 16th-localization AUC
audio-only **0.649** → causal note-sequence-only **0.935** → both **0.944**. The note sequence (run coherence)
carries the placement signal; audio is nearly secondary once note context is present. Our current onset head
is **audio-only + non-causal** (a deliberate Stage-2 anti-drift choice) → it discards the 0.93 signal → places
isolated audio-salient 16ths instead of coherent runs → the "awkward" play-feel (v7). This supersedes the
"audio-ambiguity ceiling" conclusion ([[chaos_placement_ceiling_SUPERSEDED]], now banner-superseded).
Root insight (user): we measured the wrong objective — global/audio, not local note-sequence coherence.

## The program (3 coordinated pieces; architecture is the de-risked headline)
1. **Architecture — sequence-aware onset head.** Predict onset[t] from `audio[t]` (non-causal, the ANCHOR)
   + `notes[<t]` (causal note context). = the probe's `both` model, integrated into
   `LayeredTypedChartGenerator`'s onset head. **AR-drift mitigation** (Stage-2 onset collapsed to empty):
   (a) AUDIO ANCHOR — audio always signals activity so it can't collapse to empty; (b) scheduled sampling
   (mix teacher-forced + own predictions in training) to close the train/gen gap; (c) KV-cache the causal
   note branch for O(T) decode. GATE: `diag_ar_stability.py` (AR rollout stays stable+coherent?).
2. **Objective — local coherence, not global L1.** Revive the Stage-2 taste critic (corrupted-real negatives;
   REAL>BASE>CHAOS; scores arrow/sequence coherence) as the selection/eval signal. per-song L1 is too coarse
   (v7 matched it yet played awkward). Candidate also: local n-gram / windowed pattern likelihood.
3. **Representation — local chaos + groove manifold.** Chaos as a LOCAL pattern descriptor (sparse-chaos is
   valid; the global scalar conflates chaos with density). Condition on PATTERN properties, not a radar POINT
   — the dims are coupled (jumps→voltage+air CONFIRMED via formulas; stream↔freeze nuance UNVERIFIED — stream
   may count hold occupancy → could be positive not inverse). Radar-point conditioning is WHY chaos went OOD.

## Sequencing (cheap gates before big builds — [[experiment-design]])
- [DONE — YELLOW] AR-stability de-risk (`diag_ar_stability.py`): free-run Bernoulli AR does NOT collapse
  (audio anchor prevents empty) but EXPLODES — density 0.73 (both) / 0.83 (seq) vs real 0.177, mean-run 5.7
  vs real 1.0. Exposure bias the OTHER way: own context says "in a run, continue" → runaway streams. Audio
  anchor moderates (both<seq) but doesn't tame. So: signal real (teacher-forced 0.935), but naive AR
  unstable → **scheduled sampling is MANDATORY, not optional**, with residual risk it doesn't fully tame it.
- [DONE — PARTIAL] Scheduled-sampling de-risk (`--scheduled_sampling`, one-step, eps→0.5): density 0.73→0.66,
  mean-run 5.7→4.0 — HELPED but did NOT tame (still ~3.7× real density, stream-biased). The instability is
  RUNAWAY SELF-FEEDBACK (place note → context says "in a run" → snowball); SS only dampens it. Says: the
  FULLY-AR onset formulation is the wrong shape, not that the signal is bad (still 0.935 teacher-forced).
- [PROPOSED PIVOT] NON-AR note-context / ITERATIVE REFINEMENT: first-pass onset from audio (current stable
  head) → refine with note-context computed over the FIXED first-pass chart (not own streaming output) →
  1-2 passes approximate the real-note context (0.935) with NO runaway loop. De-risk: 2-pass refine probe —
  does it improve placement (run-lengths toward real) while staying stable? Alternatives if it fails:
  density-budgeted AR decode (caps density, may stay stream-biased); full sequential SS with eps→1.
- [ ] Build (only if scheduled-sampling gate passes): causal note-context branch on the onset head
  (warm-start v4 audio branch) + scheduled sampling + KV-cache the note branch. Eval on PLACEMENT (16th-AUC +
  run-length match) + taste critic + playtest. NOT per-song L1.
- [ ] Then objective (taste critic) + representation (local chaos) as follow-on refinements.

## Critic-guided refinement (user idea, 06-22) — the OBJECTIVE layer on top of refinement
generate → critic grades → regenerate-the-weak-parts → keep if better → iterate. Marries the two open
pieces: refinement = the ARCHITECTURE (how to regenerate coherently), taste critic = the OBJECTIVE (what
"better" means). Three flavors, increasing power:
  1. best-of-N + critic (SELECT) — did this (Stage 2b), helped but only picks, never fixes.
  2. critic-guided refinement (this idea, FIX) — critic flags weak bars, regenerate them conditioned on the
     rest (= local inpainting = refinement applied locally), keep if critic improves. Iterate.
  3. critic as a training signal (Stage 2c) — fine-tune toward high-critic outputs.
KEY: (2) NEEDS the refinement mechanism — regenerating a flagged region coherently requires conditioning on
the surrounding FROZEN notes (inpainting); a plain audio-only re-sample just yields another awkward region.
So both the architecture pivot and this idea converge on the SAME foundation (note-context-conditioned
onset); the critic is the guide + stopping rule on top. Caveats: critic grades whole-chart -> need LOCAL
(windowed) scores to target regeneration; iterating toward a critic risks GAMING it (ours is a validated
taste metric REAL>BASE>CHAOS, but cap iterations + keep a playtest in the loop).

## Refinement-mechanism de-risk (DONE — POSITIVE, `diag_refine_probe.py`)
Non-causal frozen-context refiner (denoising: audio + noisy-chart context -> real onset), iterated:
```
  pass        16th-AUC  run-mean  density  Δ      (REAL: density 0.198, run 1.02)
  C0 (input)   0.734     1.12     0.199    —      corrupted rough chart
  refine 1     0.865     1.01     0.198    0.078  big lift, stays real-shaped
  refine 2     0.704     1.01     0.198    0.008  degrades (OOD: own output != training corruption)
  refine 3     0.653     1.01     0.198    0.002
```
- **STABLE — no explosion** (density/run hold at real across passes, Δ->0/converges) — the decisive contrast
  with AR (0.73 density, run 5.7). Breaking the loop into frozen passes works.
- **One pass lifts placement 0.734 -> 0.865** (toward the 0.93 ceiling). Frozen-context refinement recovers
  good placement from a rough chart -> THIS is the architecture.
- Wrinkle: passes 2-3 degrade (refiner only saw synthetic corruption, not its OWN output -> OOD). Fix: train
  on own pass-1 outputs (scheduled-sampling family), or just use ONE pass.
- CAVEAT: the synthetic corruption KEEPS ~half the real 16ths in context, so some of 0.865 reads off
  surviving real notes. The audio-only C0 errs differently (WRONG placement, not drops). -> last cheap gate:
  transfer to a REAL audio-only C0 (train refiner on v4 audio-only outputs -> real) before the full build.

## TRANSFER gate (DONE — SOBERING, `diag_refine_probe.py --real_c0`)
Refiner trained on v4's REAL audio-only C0 (not synthetic corruption), iterated:
```
  pass     16th-AUC  run-mean  density   (REAL: density 0.198, run 1.02)
  C0 (v4)   0.456     1.02     0.198      v4's real rough pass — BELOW chance (its 16ths ANTI-correlate w/ real)
  refine 1  0.666     1.00     0.198      marginal: ~= audio-only ceiling (0.649), NOT the 0.935 ceiling
  refine 2  0.592     ...      ...
```
- STABLE (no explosion, converges) — mechanism robust regardless of C0 source. But placement gain MARGINAL.
- The synthetic test (0.73→0.865) was OPTIMISTIC — it leaked ~half the real 16ths into the context. The REAL
  v4 C0 scores 0.456 (anti-correlated: places 16ths in WRONG spots), so the context is MISLEADING and the
  refiner falls back to ~audio-only (0.666). **Refinement CANNOT bootstrap good placement from a bad C0**, and
  audio-only is our only C0 — circular. The 0.935 ceiling needs good context we can't produce from audio.

## VERDICT (06-22): sequence signal is REAL but NOT cheaply exploitable
AR onset explodes; refinement is stable but can't bootstrap from v4's anti-correlated C0; iterating doesn't
climb. Reaching the 0.935 ceiling requires good context (near-real notes) that audio alone can't produce.
Placement excellence needs a different paradigm (learn the placement DISTRIBUTION from multiple human
chartings; or a much stronger first-pass model), not these levers. **Step back: ship-state = v4 +
amount-control (calib/conditioning); this rigorous bounding is the thesis deliverable.** The critic-guided
loop COULD iteratively improve context but starts from anti-correlated C0 (steep, uncertain) — only if pushing.

## Risks / open questions
- AR-drift survival under generation (the gate above settles direction).
- Cost: causal note branch + scheduled sampling is a real architecture change to the generator (Phase 2.6),
  not a fine-tune. Bigger than anything in the chaos arc.
- The teacher-forced 0.935 includes "continue-the-run" autocorrelation — real and exploitable, but gen-time
  AUC will be lower; the AR-stability + eventual playtest are the honest tests.
- Current best playable model stays **gen_highres_v4** until the new head proves out.
